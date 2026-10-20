# SPDX-License-Identifier: Apache-2.0
"""汇总已完成的 v5 配对结果；未产生结果的原任务保留 pending 状态。"""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[2]
    tasks = []
    for name in ("PAYLOAD_FULL_TASKS.json", "PAYLOAD_EXPANSION_TASKS.json"):
        tasks.extend(json.loads((root / name).read_text()))
    rows = []
    for task in tasks:
        output = Path(task["output"])
        output = output if output.is_absolute() else repo / output
        result_path = output / "result.json"
        row = {
            "batch": task["batch"], "history": task["history"],
            "task": task["task"], "result": str(result_path.relative_to(repo)),
        }
        if not result_path.exists():
            row["result_available"] = False
        else:
            result = json.loads(result_path.read_text())
            matrix = result["source_matrix"]
            stats = matrix["statistics_us"]
            row.update(
                result_available=True, passed=result["passed"],
                full_csa_validated=result["full_csa_validated"],
                graph_replay_validated=result["graph_replay_validated"],
                statistics_us=stats,
                saved_us=round(stats["spmd_reference"]["median_us"]
                               - stats["spmd_candidate"]["median_us"], 4),
                paired_saved_us=[
                    round(a - b, 4)
                    for a, b in zip(stats["spmd_reference"]["samples_us"],
                                    stats["spmd_candidate"]["samples_us"], strict=True)
                ],
            )
        rows.append(row)
    report = {
        "scope": "每档同进程、同输入、同工具链，五个交替完整L1窗口；未跨轮拼接",
        "formal_adoption": False,
        "reason": "B4/B24扩档回退，尚缺采用所需S1/S5等边界验证",
        "rows": rows,
    }
    (root / "PAYLOAD_PERFORMANCE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    for row in rows:
        print(row["history"], row["batch"], row.get("passed"), row.get("saved_us"))


if __name__ == "__main__":
    main()
