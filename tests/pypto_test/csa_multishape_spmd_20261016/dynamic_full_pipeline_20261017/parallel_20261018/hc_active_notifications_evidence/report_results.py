# SPDX-License-Identifier: Apache-2.0
"""汇总 HC 有效消费者通知的现成设备结果，不编译、不测量、不访问 NPU。

只把原完整检查、三方矩阵和四阶段信号读回全部通过的作业计入性能结论。
参考源固定为 d56c9eb6；候选间不得跨作业相减或将小幅收益当成30%达标。
原始 result.json/trace 保持不变，本脚本仅更新本目录派生报告。
"""

import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    registry = json.loads((here / "device_registry.json").read_text())
    rows = []
    for case in registry["cases"]:
        directory = Path(case["output"])
        result_path = directory / "result.json"
        matrix_path = directory / "source_matrix/result.json"
        slots_path = directory / "hc_phase_slots.json"
        row = {key: case[key] for key in ("history", "batch", "task_id", "output")}
        row["complete"] = False
        if all(path.exists() for path in (result_path, matrix_path, slots_path)):
            result = json.loads(result_path.read_text())
            matrix = json.loads(matrix_path.read_text())
            slots = json.loads(slots_path.read_text())
            row["complete"] = result.get("stage") == "complete"
            row["passed"] = bool(result.get("passed") and matrix.get("passed") and slots.get("passed"))
            if row["complete"] and row["passed"]:
                stats = matrix["statistics_us"]
                candidate = stats["spmd_candidate"]["samples_us"]
                reference = stats["spmd_reference"]["samples_us"]
                assert len(candidate) == len(reference) == 5
                row["statistics_us"] = stats
                row["candidate_vs_reference"] = matrix["candidate_vs_reference"]
                row["paired_reference_minus_candidate_us"] = [
                    round(ref - new, 5) for ref, new in zip(reference, candidate)
                ]
                row["comparison_to_original_task"] = matrix["comparison"]["spmd_candidate"]
                row["skipped_notifications"] = slots["skipped_publications_per_call"]
        rows.append(row)
    report = {
        "reference_commit": "d56c9eb6",
        "complete": all(row["complete"] for row in rows),
        "passed": all(row.get("passed", False) for row in rows),
        "new_device_execution": False,
        "scope": "正式W8A8/S6/同进程三方交替五窗完整L1；原PTO精确检查，不是Native/token验收",
        "rows": rows,
    }
    (here / "DEVICE_RESULTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# HC 有效消费者通知：完整设备结果",
        "",
        "对照为 d56c9eb6。以下均为同一作业内三方交替五窗完整 L1 的中位数，单位 μs。",
        "原分 task PTO 是最终30%目标的分母；相对 d56 的改善只表示本候选的增量收益。",
        "",
        "| 历史 / B | 原 task | d56 SPMD | 候选 | 对 d56 降低 | 每次少发通知 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        label = f"{row['history']} / B{row['batch']}"
        if not row.get("passed") or not row["complete"]:
            lines.append(f"| {label} | 待完成 | | | | |")
            continue
        stats = row["statistics_us"]
        values = [stats[name]["median_us"] for name in ("dynamic_pto", "spmd_reference", "spmd_candidate")]
        reduction = row["candidate_vs_reference"]["latency_reduction_pct"]
        lines.append(f"| {label} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {reduction:.3f}% | {row['skipped_notifications']} |")
    lines.extend([
        "",
        "只有完整数值/Top-K/cache/state、A→B→A、图与历史/页表重放、保护区、72核代际",
        "以及本候选四阶段实际发布集合全部通过后才计入上表。原始五窗、任务和路径见",
        "DEVICE_RESULTS.json 与 device_registry.json；本轮未新增 Native 或整模型测试。",
        "未补齐全部七档，不据三个代表点宣称全区间性能验收通过。",
        "",
    ])
    (here / "DEVICE_RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"complete": report["complete"], "passed": report["passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
