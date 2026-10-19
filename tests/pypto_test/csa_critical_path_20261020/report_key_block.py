# SPDX-License-Identifier: Apache-2.0
"""只汇总已终态且完整验证通过的三方配对L1，不挑选跨轮最优值。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import subprocess


def summarize(folder, task):
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        return {"task": task, "status": status, "performance_eligible": False}
    result = json.loads((folder / "result.json").read_text())
    matrix = json.loads((folder / "source_matrix/result.json").read_text())
    if not (result["stage"] == "complete" and result["passed"]
            and result["full_csa_validated"] and result["performance_measured"]
            and result["graph_replay_validated"] and result["post_l1_protocol"]
            and matrix["passed"] and matrix["same_process"] and matrix["same_root_addresses"]):
        raise ValueError(f"完整数值/状态/同轮验证未通过: {folder}")
    names = ("dynamic_pto", "spmd_reference", "spmd_candidate")
    # 测试壳每轮轮换三方顺序。先核对三项一组，下面的zip才是真实同轮配对，
    # 不能把每方的最小样本重新组合成所谓配对收益。
    if len(matrix["windows"]) != 15 or any(
        {row["side"] for row in matrix["windows"][start:start + 3]} != set(names)
        for start in range(0, 15, 3)
    ):
        raise ValueError("五轮各三方的完整轮换记录不符")
    windows = defaultdict(list)
    for row in matrix["windows"]:
        if row["clock_freq_hz"] != 50000000 or row["cores"] != 72:
            raise ValueError("L1频率或参与核数改变，须检查原始记录")
        if row["side"].startswith("spmd_") and row["task_count"] != 1:
            raise ValueError("候选或参考不再是一个完整SPMD")
        windows[row["side"]].append(row["incore_span_us"])
    if set(windows) != set(names) or any(len(windows[name]) != 5 for name in names):
        raise ValueError("必须保留三方各五个完整窗口")
    values = {name: median(windows[name]) for name in names}
    baseline, reference, candidate = (values[name] for name in names)
    paired_savings = [r - c for r, c in zip(windows["spmd_reference"], windows["spmd_candidate"])]
    return {"task": task, "status": status, "performance_eligible": True,
            "windows_us": dict(windows), "median_us": values,
            "net_saved_vs_current_us": reference - candidate,
            "case_meets_20us_gate": reference - candidate >= 20,
            "paired_savings_us": paired_savings,
            "median_paired_savings_us": median(paired_savings),
            "faster_window_pairs": sum(value > 0 for value in paired_savings),
            "review_note": "20μs字段只核对本档中位数；须结合五窗稳定性及另一档无明确退步，不能据单字段自动扩档。",
            "goal_latency_reduction_percent": 100 * (baseline - candidate) / baseline,
            "display_speedup_percent": 100 * (baseline - candidate) / candidate,
            "goal_30_percent_met": candidate <= baseline * 0.7,
            "root_result": str(folder / "result.json"),
            "raw_matrix": str(folder / "source_matrix/result.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    jobs = {16: "task_20261020_014546_26630633015", 28: "task_20261020_014433_25715589889"}
    reports = {str(batch): summarize(args.root / f"key_block_h131072_b{batch}_v1", task)
               for batch, task in jobs.items()}
    (args.root / "KEY_BLOCK_PERFORMANCE.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    for batch, report in reports.items():
        print(batch, report.get("median_us", report["status"]))
