# SPDX-License-Identifier: Apache-2.0
"""汇总跨页多ND候选的完整三方五窗；基础搬运probe不混入CSA性能。"""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location("paired_report", HERE.parent / "report_key_block.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reports = {}
    for batch, job in json.loads((HERE / "TASKS.json").read_text()).items():
        if batch == "probe":
            continue
        report = module.summarize(HERE / job["folder"], job["task"])
        report.update(
            reference="source_after_score_split/13259f92",
            candidate="key_multind/source_candidate",
            adopted=False,
            note="新编译器下三方同轮运行；Mat父双槽改为直接N64载入，不能声称只改变DMA条数",
        )
        reports[batch] = report
        print(batch, report.get("median_us", report["status"]), report.get("net_saved_vs_current_us"))
    (HERE / "PERFORMANCE.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
