# SPDX-License-Identifier: Apache-2.0
"""汇总当前QK提前FIX/独立WS Acc候选的已终态完整三方五窗。"""
import json
from pathlib import Path

from report_key_block import summarize

HERE = Path(__file__).resolve().parent
JOBS = {
    16: "task_20261020_024554_121710728666",
    28: "task_20261020_024441_116755019471",
}

if __name__ == "__main__":
    reports = {str(batch): summarize(HERE / f"score_fix_acc_h131072_b{batch}_v1", task)
               for batch, task in JOBS.items()}
    (HERE / "SCORE_FIX_ACC_PERFORMANCE.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    for batch, report in reports.items():
        print(batch, report.get("median_us", report["status"]))
