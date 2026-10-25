# SPDX-License-Identifier: Apache-2.0
"""用固定MIX的1C2V拓扑恢复AIV映射，验证已有时间戳，不重新执行NPU。"""

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


def save_new(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == content, f"拒绝改写已有不同证据：{path}"
    else:
        path.write_text(content)


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
        assert all(c["status"] == "PASS" and c["bit_equal"] for c in check["numerical"].values())
    assert len(report["windows"]) == 20
    windows, raw_windows = [], []
    for window in report["windows"]:
        clock = window["clock"]
        assert clock["guards_passed"] and not clock["physical_mapping_verified"]
        raw = json.loads(Path(window["raw"]).read_text())
        storage = json.loads(Path(clock["raw_clock"]).read_text())
        values = storage["storage_int64"]
        assert storage["guard_words"] == 8 and values[:8] == values[-8:] == [37] * 8
        values = [values[8 + i * 8 : 16 + i * 8] for i in range(72)]
        assert all(v[5] == i and v[7] == 37 for i, v in enumerate(values))
        assert len(raw["aicore_tasks"]) == 72 and window["task_groups"] == 1
        rows = {r[0]: r for r in raw["aicore_tasks"]}
        assert set(rows) == set(range(72))
        assert len({r[1] for r in rows.values()}) == 1
        aic = [values[i][4] for i in range(24)]
        assert set(aic) == set(range(24))
        # Simpler scheduler_cold_path.cpp:968及scheduler_types.h:214..218。
        # 显式MIX同一逻辑组由固定物理AIC及其两个AIV承载，不使用失败的AIV builtin结果。
        mapping = aic + [24 + 2 * aic[i // 2] + i % 2 for i in range(48)]
        assert set(mapping) == set(range(72))
        scale = 1e6 / raw["metadata"]["clock_freq_hz"]
        lower = min(r[3] for r in rows.values())
        upper = max(r[4] for r in rows.values())
        assert math.isclose((upper - lower) * scale, window["incore_span_us"], abs_tol=1e-8)
        per_core = []
        for logical, physical in enumerate(mapping):
            t0, t1, t2, t3 = values[logical][:4]
            record = rows[physical]
            assert record[3] <= t0 <= t1 <= t2 <= t3 <= record[4], (logical, physical, values[logical], record)
            assert values[logical][:4] == clock["cores"][logical]["ticks"]
            components = {
                "start_skew_us": (record[3] - lower) * scale,
                "wrapper_and_prologue_us": (t0 - record[3]) * scale,
                "initialization_and_generation_us": (t1 - t0) * scale,
                "first_math_and_input_wait_us": (t2 - t1) * scale,
                "remaining_business_us": (t3 - t2) * scale,
                "tail_including_clock_flush_us": (record[4] - t3) * scale,
            }
            assert math.isclose(sum(components.values()), (record[4] - lower) * scale, abs_tol=1e-8)
            per_core.append({"logical_core": logical, "physical_core_derived": physical, **components})
        critical = max(range(72), key=lambda i: rows[mapping[i]][4])
        role_medians = {
            role: {key: statistics.median(c[key] for c in per_core[start:end]) for key in components}
            for role, start, end in (("aic", 0, 24), ("aiv", 24, 72))
        }
        phases = {
            name: (max(v[slot] for v in values) - lower) * scale
            for slot, name in enumerate(
                ("body_enter_max_us", "boot_done_max_us", "first_math_return_max_us", "business_return_max_us")
            )
        }
        windows.append(
            {
                "side": window["side"],
                "incore_span_us_diagnostic_only": window["incore_span_us"],
                "role_medians_us": role_medians,
                "global_phase_offsets_us": phases,
                "last_exit_core": per_core[critical],
                "validated_physical_intervals": 72,
            }
        )
        raw_windows.append(
            {
                "side": window["side"],
                "clock_freq_hz": raw["metadata"]["clock_freq_hz"],
                "aicore_tasks": raw["aicore_tasks"],
                "clock_storage": storage,
                "derived_logical_to_physical": mapping,
            }
        )
    sides = {}
    for side in ("reference", "candidate"):
        selected = [w for w in windows if w["side"] == side]
        assert len(selected) == 10
        sides[side] = {
            "role_median_of_window_medians_us": {
                role: {key: statistics.median(w["role_medians_us"][role][key] for w in selected) for key in components}
                for role in ("aic", "aiv")
            },
            "global_phase_medians_us": {
                key: statistics.median(w["global_phase_offsets_us"][key] for w in selected) for key in phases
            },
        }
    summary = {
        "task_id": args.task_id,
        "attempt": args.attempt,
        "case": report["case"],
        "device": report["device_id"],
        "passed": True,
        "strict_check_groups": 6,
        "performance_eligible": False,
        "validated_physical_intervals": 1440,
        "mapping": "AIC直接探针为0..23置换；AIV通过Simpler固定1C2V拓扑推导，并逐核验证时间包含关系",
        "mapping_source": (
            "simpler 57e499238fd4ee3548947ad860be456aaeeb16b5: "
            "scheduler_cold_path.cpp:968; scheduler_types.h:214..218"
        ),
        "limits": [
            "原AIV物理builtin探针无效；原始报告保持mapping=False，不冒称直接测得AIV物理号",
            "插入参数、标量store和缓存维护会扰动时序，所有耗时只作诊断，禁止计算正式收益",
            "first_math为函数返回边界，不保证流水全部完成；发布中的等待可能计入后段",
            "每角色中位数包含无数学任务的核；不同部分中位数不可相加当作总耗时",
        ],
        "statistics_us_diagnostic_only": report["statistics_us"],
        "sides": sides,
        "windows": windows,
    }
    save_new(here / "evidence" / (args.attempt + ".json"), report)
    save_new(here / "evidence" / (args.attempt + "_raw_clocks.json"), raw_windows)
    name = args.attempt + "_result.json"
    save_new(here / name, summary)
    state_path = here / "current_state.json"
    state = json.loads(state_path.read_text())
    pending = [x for x in state["pending_tasks"] if x["task_id"] == args.task_id]
    if pending:
        assert len(pending) == 1
        task = pending[0]
        state["pending_tasks"].remove(task)
        task.update(status=status, result=name)
        state["completed_tasks"].append(task)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"attempt": args.attempt, "validated_intervals": 1440, "sides": sides}, ensure_ascii=False))


if __name__ == "__main__":
    main()
