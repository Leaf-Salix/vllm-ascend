# SPDX-License-Identifier: Apache-2.0
"""复核72核精确初始化时间戳，归档全部样本并统计全局跨度及逐核耗时。"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def distribution(samples):
    return {"median": statistics.median(samples), "min": min(samples), "max": max(samples), "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt")
    parser.add_argument("task_id")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    status = subprocess.check_output(["task-submit", "--status", args.task_id], text=True).strip()
    assert status == "completed (exit=0)", status
    source = here / "runs" / args.attempt / "result.json"
    report = json.loads(source.read_text())
    assert report["passed"] and report["stage"] == "complete" and not report["performance_eligible"]
    assert len(report["checks"]) == 6 and all(report["fixture_checks"].values())
    for check in report["checks"].values():
        assert all(check["guards"].values())
        assert all(value["status"] == "PASS" and value["bit_equal"] for value in check["numerical"].values())
    assert len(report["windows"]) == 20
    sys.path.insert(0, str(here / "frozen"))
    from complete_l1_v2 import summarize

    samples = {side: [] for side in ("reference", "candidate")}
    clocks, raw_windows, csv_rows = [], [], []
    for window in report["windows"]:
        raw = json.loads(Path(window["raw"]).read_text())
        full = summarize(raw, init_layout="fused")
        assert full["task_groups"] == 1 and full["cores"] == 72
        assert full["incore_span_us"] == window["incore_span_us"]
        samples[window["side"]].append(full["incore_span_us"])
        record = {"side": window["side"], "raw_L1": raw}
        if window["side"] == "candidate":
            clock = window["clock"]
            assert clock["guards_passed"] and clock["physical_intervals_validated"] == 72
            storage = json.loads(Path(clock["raw_clock"]).read_text())
            raw_values = storage["storage_int64"]
            assert len(raw_values) == 72 * 8 + 16 and raw_values[:8] == raw_values[-8:] == [37] * 8
            values = [raw_values[8 + 8 * i : 16 + 8 * i] for i in range(72)]
            aic = [v[4] for v in values[:24]]
            assert set(aic) == set(range(24))
            mapping = aic + [24 + 2 * aic[i // 2] + i % 2 for i in range(48)]
            records = {r[0]: r for r in raw["aicore_tasks"]}
            assert set(records) == set(mapping) == set(range(72))
            scale = 1e6 / raw["metadata"]["clock_freq_hz"]
            for core, physical in enumerate(mapping):
                v = values[core]
                assert v[5:] == [core, 1, 37]
                assert v[4] == (physical if core < 24 else -1)
                row = records[physical]
                assert row[3] <= v[0] <= v[1] <= v[2] <= v[3] <= row[4]
                derived = clock["cores"][core]
                assert derived["logical_core"] == core and derived["physical_core"] == physical
                for key, value in {
                    "pre_sync_us": (v[1] - v[0]) * scale,
                    "sync_wait_us": (v[2] - v[1]) * scale,
                    "clear_and_sync_local_us": (v[2] - v[0]) * scale,
                    "adjacent_clock_read_us": (v[3] - v[2]) * scale,
                }.items():
                    assert math.isclose(value, derived[key], abs_tol=1e-9)
                csv_rows.append({"window": len(clocks), **{k: v for k, v in derived.items() if k != "ticks"}})
            first_vector = min(v[0] for v in values[24:])
            arrival, release = max(v[1] for v in values), max(v[2] for v in values)
            derived_globals = {
                "global_clear_and_sync_us": (release - first_vector) * scale,
                "to_last_participant_arrival_us": (arrival - first_vector) * scale,
                "last_arrival_to_last_release_us": (release - arrival) * scale,
                "all_roles_initialization_span_us": (release - min(v[0] for v in values)) * scale,
                "L1_start_to_last_release_us": (release - min(r[3] for r in records.values())) * scale,
            }
            assert all(math.isclose(clock[k], value, abs_tol=1e-9) for k, value in derived_globals.items())
            assert math.isclose(
                clock["global_clear_and_sync_us"],
                clock["to_last_participant_arrival_us"] + clock["last_arrival_to_last_release_us"],
                abs_tol=1e-9,
            )
            clock = {
                **clock,
                "aiv_start_spread_us": (max(v[0] for v in values[24:]) - first_vector) * scale,
                "aiv_clear_to_pre_sync_span_us": (max(v[1] for v in values[24:]) - first_vector) * scale,
                "last_arrival_role": "aic" if max(range(72), key=lambda i: values[i][1]) < 24 else "aiv",
            }
            clocks.append(clock)
            record.update(clock_storage=storage, derived_logical_to_physical=mapping)
        raw_windows.append(record)
    assert len(clocks) == 10 and len(csv_rows) == 720
    for side, values in samples.items():
        assert len(values) == 10 and values == report["statistics_us"][side]["samples"]
        assert statistics.median(values) == report["statistics_us"][side]["median"]
    global_statistics = {
        key: distribution([c[key] for c in clocks])
        for key in (*derived_globals, "aiv_start_spread_us", "aiv_clear_to_pre_sync_span_us")
    }
    local_keys = ("pre_sync_us", "sync_wait_us", "clear_and_sync_local_us", "adjacent_clock_read_us")
    role_statistics = {
        role: {
            key: distribution(
                [statistics.median(c[key] for c in clock["cores"] if c["role"] == role) for clock in clocks]
            )
            for key in local_keys
        }
        for role in ("aic", "aiv")
    }
    per_core = []
    for core in range(72):
        rows = [c["cores"][core] for c in clocks]
        per_core.append(
            {
                "logical_core": core,
                "role": rows[0]["role"],
                "physical_cores_seen": sorted({r["physical_core"] for r in rows}),
                **{key: distribution([r[key] for r in rows]) for key in local_keys},
            }
        )
    control, probe = (statistics.median(samples[s]) for s in ("reference", "candidate"))
    summary = {
        "task_id": args.task_id,
        "device": report["device_id"],
        "case": report["case"],
        "passed": True,
        "sample_windows": 10,
        "verified_core_intervals": 720,
        "strict_check_groups": 6,
        "source_commit": report["reference_commit"],
        "shared_toolchain": report["shared_toolchain"],
        "primary_definition": "max(72核syncall返回时刻)-min(48个AIV清零开始时刻)，排除generation和业务",
        "pre_sync_definition": "AIV为零Tile生成、MTE3写回、原有cacheinvalid/fence；AIC只有原有cacheinvalid/fence",
        "role_statistics_definition": "每窗先对同角色核取中位数，再报告10窗分布；每核自己的10窗数据另存",
        "global_statistics_us": global_statistics,
        "role_statistics_us": role_statistics,
        "per_core": per_core,
        "complete_L1_us": {side: distribution(values) for side, values in samples.items()},
        "instrumented_L1_minus_control_us": probe - control,
        "instrumented_L1_difference_pct": (probe / control - 1) * 100,
        "limits": [
            "计时发生在完整CSA内，额外取时/参数仍可能改变编译和执行时序，属于插桩诊断",
            "区间内不写诊断GM；四个时间戳在函数末尾统一写出，不修改生产源码或SDK",
            "AIC提前到达屏障的等待与AIV清零重叠，各核/各段中位数不能相加",
            "all_roles初始化跨度包含AIC早到后的等待；L1起点到全核释放还包含入口参数处理",
            "背靠背取时用于观察计时粒度，未虚构扣除探针开销；结果不是删除该阶段的净性能收益",
        ],
    }
    evidence = here / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / (args.attempt + ".json")).write_bytes(source.read_bytes())
    (evidence / (args.attempt + "_raw.json")).write_text(json.dumps(raw_windows, ensure_ascii=False, indent=2) + "\n")
    with (here / (args.attempt + "_allcores.csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    (here / (args.attempt + "_summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    manifest_path = here / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status="measurement_completed", task_id=args.task_id, result=args.attempt + "_summary.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "global_medians_us": {k: v["median"] for k, v in global_statistics.items()},
                "role_medians_us": {r: {k: v["median"] for k, v in d.items()} for r, d in role_statistics.items()},
                "control_L1": control,
                "instrumented_L1": probe,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
