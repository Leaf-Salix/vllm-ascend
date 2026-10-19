# SPDX-License-Identifier: Apache-2.0
"""分开呈现叶平衡v1与区间v2；明确标注B16执行代码相同而复用整轮结果。"""
import json
from pathlib import Path

from report_key_block import summarize

HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    first = {
        str(b): summarize(HERE / f"leaf_balanced_h131072_b{b}_v1", task)
        for b, task in ((16, "task_20261020_031327_23202592313"),
                        (28, "task_20261020_031111_223008730987"))
    }
    equality = json.loads((HERE / "LEAF_INTERVAL_DISABLED_COMPARISON.json").read_text())
    assert equality["B16_equals_prior_fix_acc_after_comments_whitespace"]
    reused = summarize(HERE / "score_fix_acc_h131072_b16_v1", "task_20261020_024554_121710728666")
    reused.update(measurement_reused=True, equivalence_evidence="LEAF_INTERVAL_DISABLED_COMPARISON.json",
                  measurement_note="沿用相同执行C++对应的完整配对五窗，不是本轮新测，不挑选旧轮最小值")
    second = {"16": reused,
              "28": summarize(HERE / "leaf_interval_h131072_b28_v2", "task_20261020_033233_30933866282")}
    result = {"v1": first, "interval_v2": second}
    (HERE / "LEAF_BALANCE_PERFORMANCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for version, reports in result.items():
        for batch, report in reports.items():
            print(version, batch, report.get("median_us", report["status"]),
                  "reused" if report.get("measurement_reused") else "")
