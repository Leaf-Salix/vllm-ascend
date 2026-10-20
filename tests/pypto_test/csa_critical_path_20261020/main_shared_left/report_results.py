# SPDX-License-Identifier: Apache-2.0
"""共享Left代表档完整三方五窗汇总；门槛仍需结合另一档及逐窗稳定性。"""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location("main_left_paired", HERE.parent / "report_key_block.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jobs = json.loads((HERE / "TASKS_V2.json").read_text())
    reports = {}
    for batch, row in jobs.items():
        report = module.summarize(HERE / row["folder"], row["task"])
        report.update(reference="source_after_score_split/13259f92",
                      candidate="main_shared_left/source_candidate_v2", adopted=False,
                      note="保持原每输出K256链；完整L1收益不能由Left搬运次数直接推算")
        reports[batch] = report
        print(batch, report.get("median_us", report["status"]), report.get("net_saved_vs_current_us"))
    (HERE / "PERFORMANCE_V2.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
