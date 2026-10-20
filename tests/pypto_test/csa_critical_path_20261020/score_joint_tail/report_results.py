# SPDX-License-Identifier: Apache-2.0
"""当前双QK与列池下生产/消费联合短尾的完整三方五窗汇总，不跨轮拼接。"""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location("pool_paired", HERE.parent / "report_key_block.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reports = {}
    for batch, job in json.loads((HERE / "TASKS.json").read_text()).items():
        report = module.summarize(HERE / job["folder"], job["task"])
        report.update(reference="source_after_score_split/13259f92",
                      candidate="score_joint_tail/source_candidate", adopted=False,
                      note="Cube与AIV共同裁剪实际短尾；保持完整20μs门槛与另一档无明确退步")
        reports[batch] = report
        print(batch, report.get("median_us", report["status"]), report.get("net_saved_vs_current_us"))
    (HERE / "PERFORMANCE.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
