# SPDX-License-Identifier: Apache-2.0
"""汇总当前数学wave候选的完整三方五窗，不自动采用或扩大验证范围。"""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location(
        "wave_paired_report", HERE.parent / "report_key_block.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jobs = json.loads((HERE / "DEVICE_TASKS_V1.json").read_text())
    reports = {}
    for batch, task in jobs.items():
        report = module.summarize(HERE / task["folder"], task["task"])
        report.update(
            reference="source_after_score_split/13259f92",
            candidate="wave_current_math/source_candidate",
            adopted=False,
            timing="完整L1首个incore开始到最后incore结束，包含内部等待",
            gate="重点档至少净省20μs，另一档无明确退步；五窗稳定性仍须人工判读",
        )
        reports[batch] = report
        print(batch, report.get("median_us", report["status"]),
              report.get("net_saved_vs_current_us"))
    (HERE / "PERFORMANCE.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
