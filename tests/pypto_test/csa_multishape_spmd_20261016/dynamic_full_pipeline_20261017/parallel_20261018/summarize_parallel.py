# SPDX-License-Identifier: Apache-2.0
"""只读取本轮隔离源码矩阵的已落盘结果，不初始化设备或重新测量。

三方耗时均取同一进程轮换的完整 L1 窗口中位数。候选相对旧 SPMD 的
收益与相对原分 task 的 30% 验收分开记录；运行中或失败结果不生成收益。
候选间的运行环境可以不同，不能横跨任务相减得到组合优化的收益。
"""

import json
from pathlib import Path


def summarize(directory):
    rows = []
    for path in sorted(directory.glob("*/result.json")):
        result = json.loads(path.read_text())
        matrix_path = path.parent / "source_matrix/result.json"
        if not matrix_path.exists():
            continue
        matrix = json.loads(matrix_path.read_text())
        row = {
            "case": path.parent.name,
            "history": result.get("history"),
            "batch": result.get("batch"),
            "query_tokens": result.get("query_tokens"),
            "stage": result.get("stage"),
            "full_csa_validated": result.get("full_csa_validated", False),
            "passed": bool(result.get("passed") and matrix.get("passed")),
            "result": str(path.relative_to(directory)),
            "matrix_result": str(matrix_path.relative_to(directory)),
        }
        if row["passed"]:
            row["statistics_us"] = matrix["statistics_us"]
            row["candidate_vs_reference"] = matrix["candidate_vs_reference"]
            row["comparison_to_original_task_pto"] = matrix["comparison"]
            row["target_met"] = matrix["comparison"]["spmd_candidate"]["target_met"]
        rows.append(row)
    return {
        "scope": "CSA S6，正式 W8A8，同进程原分 task/旧 SPMD/候选五窗口完整 L1",
        "native_precision_or_token_acceptance": False,
        "cross_run_combination_speedup_claimed": False,
        "results": rows,
    }


def main():
    directory = Path(__file__).resolve().parent
    report = summarize(directory)
    (directory / "result_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
