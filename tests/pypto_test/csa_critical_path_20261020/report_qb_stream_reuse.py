# SPDX-License-Identifier: Apache-2.0
"""汇总Q_B候选原三方五窗；关闭路径只报告代码相同，不编造B16新测量。"""

import json
from pathlib import Path

from report_key_block import summarize

HERE = Path(__file__).resolve().parent


def main():
    b28 = summarize(HERE / "qb_stream_h131072_b28_v3", "task_20261020_041024_37973320637")
    same = json.loads((HERE / "QB_STREAM_DISABLED_COMPARISON.json").read_text())
    assert same["passed"]
    report = {
        "28": b28,
        "16": {
            "comparison": "策略关闭，完整生成C++去行注释和空白后与正式版本相同",
            "evidence": "QB_STREAM_DISABLED_COMPARISON.json",
            "new_device_measurement": False,
            "latency_us": None,
            "note": "这是代码一致性证据，不声称本轮新测B16性能或量出0μs差值",
        },
        "formal_source_changed": False,
        "note": "是否扩档须审阅全部配对窗口及20μs门槛，不用单窗口或不同轮差值替代",
    }
    (HERE / "QB_STREAM_PERFORMANCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(b28.get("median_us", b28["status"]))


if __name__ == "__main__":
    main()
