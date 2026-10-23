# SPDX-License-Identifier: Apache-2.0
"""只分析已有 Native profiling 与 HCA L1，不发起设备测试。

Native 按算子类型和完整形状分组，避免把 Q_B/O_B 等同名矩阵乘混为一项。
PTO 分别保留 task 首尾包络、kernel-duration-us 核占用/硬件核数，以及
本地设置/提前等待占用；不同定义不相减为已证实的性能收益。
"""

import collections
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def stage(name):
    for prefix, label, cores in (
        ("proj_a_mm", "O_A", 24),
        ("_proj_b_mm", "O_B", 24),
        ("quant", "分组量化", 48),
        ("hca_oproj_hc_post", "O_B归约及HC_post", 48),
        ("hca_unified_attention_aic", "Attention_AIC", 24),
        ("hca_unified_attention_aiv", "Attention_AIV", 48),
        ("hca_qb_stream", "Q_B", 24),
    ):
        if name.startswith(prefix):
            return label, cores
    return None


def audit_pto(directory):
    samples = collections.defaultdict(list)
    for path in sorted(directory.glob("l1_*_control/swimlane.json")):
        groups = collections.defaultdict(list)
        for event in json.loads(path.read_text())["traceEvents"]:
            # Worker View 独占实际每核事件；不重复累加 Task View。
            if event.get("ph") != "X" or event.get("pid") != 4:
                continue
            key = stage(event["name"].split("(")[0])
            if key is not None:
                groups[key].append(event)
        for (label, cores), events in groups.items():
            kernel_sum = sum(e["args"]["kernel-duration-us"] for e in events)
            samples[label].append({
                "source": str(path.relative_to(HERE)),
                "task_envelope_us": max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events),
                "kernel_occupancy_div_cores_us": kernel_sum / cores,
                "setup_and_wait_occupancy_us": sum(e["dur"] for e in events) - kernel_sum,
                "worker_records": len(events),
                "hardware_cores": cores,
            })
    return {label: {
        "median_task_envelope_us": statistics.median(v["task_envelope_us"] for v in rows),
        "median_kernel_occupancy_div_cores_us": statistics.median(v["kernel_occupancy_div_cores_us"] for v in rows),
        "windows": rows,
    } for label, rows in samples.items()}


def main():
    native_root = HERE / "runs/h131072_b16_eager_profile_v1/profile_native"
    csv_path, = native_root.rglob("kernel_details.csv")
    grouped = collections.defaultdict(list)
    for row in csv.DictReader(csv_path.open()):
        if row["Type"] in ("Fill", "TensorMove"):
            # 这里只列计算分项；全层跨度仍使用 profile_eager 的边界判定，不以名字删任务。
            continue
        grouped[(row["Type"], row["Input Shapes"])].append(float(row["Duration(us)"]))
    result = {
        "scope": "已有数据的瓶颈定位；无新设备执行，不是同窗 Native/PTO 收益验收",
        "aclgraph_replay": False,
        "native_source": str(csv_path.relative_to(HERE)),
        "native": [{"type": kind, "input_shapes": shape, "durations_us": durations,
                    "median_us": statistics.median(durations)}
                   for (kind, shape), durations in grouped.items()],
        "pto": {name: audit_pto(HERE / "runs" / name)
                for name in ("qb_24_b16_v1", "qb_24_b28_v1")},
        "limits": [
            "只选 Q_B 24核实验中的原版 control；候选未采用。",
            "Native Duration 与 PTO 核占用/24或48不是相同定义，不能直接相减。",
            "多个 task 的包络有重叠；分项不能相加为完整关键路径。",
            "这些样本的 SDK 是 csa-shared-20261021-upstream-fixes，不能与新 SDK 跨版计算收益。",
        ],
    }
    destination = HERE / "evidence/NATIVE_KERNEL_AUDIT.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
