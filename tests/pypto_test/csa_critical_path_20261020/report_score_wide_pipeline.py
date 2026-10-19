# SPDX-License-Identifier: Apache-2.0
"""保留宽面板两代表档的同轮完整三方五窗，显式记录真实私有N128。"""

import json
from pathlib import Path

from report_key_block import summarize

HERE = Path(__file__).resolve().parent


def main():
    jobs = json.loads((HERE / "SCORE_WIDE_TASKS.json").read_text())
    reports = {}
    for batch, task in jobs.items():
        report = summarize(HERE / task["folder"], task["task"])
        report.update(
            reference="source_after_score_split/13259f92",
            candidate="source_score_wide_pipeline",
            reference_cube_panel_cols=64, actual_candidate_cube_panel_cols=128,
            public_qk_panel_selection_condition=64,
            adopted=False,
            note="公共QK_PANEL只保留原区间选择；候选私有Cube为N128，不据指令数宣称收益",
        )
        reports[batch] = report
        print(batch, report.get("median_us",report["status"]), report.get("net_saved_vs_current_us"))
    (HERE / "SCORE_WIDE_PERFORMANCE.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
