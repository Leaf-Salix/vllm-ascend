"""从单卡报告和设备 CSV 重建第二层主对照，首层成本与独立泳道另列。"""

import csv
import json
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def profile_summary(path):
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"缺少设备任务：{path}")
    starts = [Decimal(row["Start Time(us)"].strip()) for row in rows]
    durations = [Decimal(row["Duration(us)"].strip()) for row in rows]
    totals = defaultdict(lambda: {"count": 0, "sum_task_us": 0.0})
    for row, duration in zip(rows, durations):
        item = totals[row["Type"]]
        item["count"] += 1
        item["sum_task_us"] += float(duration)
    return {
        "kernel_records": len(rows),
        "first_to_last_device_us": float(max(s + d for s, d in zip(starts, durations)) - min(starts)),
        "first_kernel": rows[starts.index(min(starts))]["Name"],
        "compact_metadata_tasks": sum(row["Type"] == "CompressorMetadata" for row in rows),
        "kernels_by_task_time": dict(sorted(totals.items(), key=lambda kv: -kv[1]["sum_task_us"])),
        "scope": "独立 profiler 窗口的首末设备任务区间；分项为任务时长，可能重叠，不能相加当层耗时",
    }


def read_pair(root, prefix, policy):
    modes = {}
    fixture = None
    for mode in (1, 2):
        directory = root / f"{prefix}{mode}"
        report = json.loads((directory / "report.json").read_text())
        current = {key: report[key] for key in ("checkpoint", "batch", "history", "seed", "variant",
                                               "deterministic_level", "hccl_deterministic", "pto_reduction")}
        if fixture is not None and fixture != current:
            raise ValueError("两档配置除 NZ mode 外不一致")
        fixture = current
        if report["status"] != "MEASURED" or report["effective_weight_nz_mode"] != mode:
            raise ValueError("缺少有效运行或实际 NZ mode 不符")
        timing = report["timing"]
        if timing["status"] != "MEASURED" or not timing.get("pto_compact_metadata"):
            raise ValueError("计时未完成，或未声明 PTO metadata 范围")
        if policy == "reuse" and timing.get("compact_metadata_policy") != "reuse":
            raise ValueError("第二层主结果必须显式声明复用 metadata")
        if timing.get("compact_metadata_policy", policy) != policy:
            raise ValueError("metadata 计时口径不符")
        entry = {"native_weight_formats": report["native_weight_formats"],
                 "root_layouts": report["root_layouts"], "native_over_pto_p50": timing["native_over_pto_p50"]}
        for backend in ("native", "pto"):
            data = timing[backend]
            stamps = data["start_timestamps_raw"]
            if (len(stamps) != timing["iters"] or len(data["samples_us"]) != timing["iters"]
                    or any(b <= a for a, b in zip(stamps, stamps[1:]))):
                raise ValueError("事件时间戳未逐次更新")
            if any(v["status"] != "PASS" for v in data["guards"].values()):
                raise ValueError("metadata 或保护区检查失败")
            files = list((directory / f"profile/{backend}").rglob("kernel_details.csv"))
            if len(files) != 1:
                raise ValueError(f"设备 CSV 不唯一：{files}")
            profile = profile_summary(files[0])
            expected_metadata = 2 if backend == "native" or policy == "produce" else 0
            if profile["compact_metadata_tasks"] != expected_metadata:
                raise ValueError(f"{directory.name}/{backend} 的 metadata 任务与计时口径不符")
            entry[backend] = {
                **{key: data[key] for key in ("us_min", "us_p50", "us_p95", "us_max")},
                "profile": profile,
            }
        entry["pto_self_x_out"] = report["pto_self"]["x_out"]
        entry["pto_native_x_out"] = report["pto_native"]["x_out"]
        modes[str(mode)] = entry
    return modes, fixture


def main():
    root = Path(__file__).resolve().parent
    modes, fixture = read_pair(root, "following_mode", "reuse")
    first_modes, first_fixture = read_pair(root, "mode", "produce")
    if fixture != first_fixture:
        raise ValueError("首层与第二层除 metadata 策略外的输入配置不一致")
    result = {
        "status": "MEASURED", "fixture": fixture, "modes": modes,
        "compact_metadata_policy": "reuse",
        "scope": "单卡 B16/S6/H8192、model.layers.2 正式权重和合成历史；"
                 "按同一步第二个 CSA 层复用 metadata，Native 保留实际逐层生成；"
                 "不是跨实现数值、整模型或 750 us 验收",
        "first_layer_modes": first_modes,
        "first_layer_scope": "独立任务，PTO 每次在图区间内生成两组 metadata；"
                             "受不同轮次波动影响，不与主结果相减当作 metadata 算子净成本",
    }
    swimlane = root / "pto_mode2_swimlane/dfx"
    raw = json.loads((swimlane / "chip_swimlane_records.json").read_text())
    metadata = raw["metadata"]
    if len(metadata["run_boundaries"]) != 1 or metadata["dropped_run_boundaries"]:
        raise ValueError("DFX 必须只有一个完整调用窗口")
    groups = defaultdict(list)
    trace = json.loads((swimlane / "merged_swimlane.json").read_text())
    for event in trace["traceEvents"]:
        if event.get("ph") == "X" and event.get("cat") == "event" and "kernel-duration-us" in event.get("args", {}):
            groups[event["name"].split("(")[0]].append(event)
    if sum(map(len, groups.values())) != len(raw["aicore_tasks"]):
        raise ValueError("转换后的 worker 样本数与原始记录不符")
    windows = {}
    for name, events in sorted(groups.items(), key=lambda kv: min(v["ts"] for v in kv[1])):
        start = min(v["ts"] for v in events)
        end = max(v["ts"] + v["dur"] for v in events)
        windows[name] = {
            "worker_records": len(events), "first_start_us": round(start, 2),
            "last_finish_us": round(end, 2), "first_to_last_us": round(end - start, 2),
            "max_kernel_us": round(max(v["args"]["kernel-duration-us"] for v in events), 2),
        }
    total = re.search(r"Total Test Time: ([\d.]+) us", (swimlane / "converter_output.txt").read_text())
    if total is None:
        raise ValueError("泳道转换器没有给出完整调度区间")
    result["pto_mode2_swimlane"] = {
        "dispatch_to_finish_us": float(total[1]), "worker_records": len(raw["aicore_tasks"]),
        "scope": "独立 eager DFX 调用，输入与 mode=2 相同；阶段窗口相互重叠，不能相加或替代整模型计时",
        "kernels": windows,
    }
    (root / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for mode, entry in result["modes"].items():
        print(f"mode={mode}: Native {entry['native']['us_p50']:.2f} us, PTO {entry['pto']['us_p50']:.2f} us")


if __name__ == "__main__":
    main()
