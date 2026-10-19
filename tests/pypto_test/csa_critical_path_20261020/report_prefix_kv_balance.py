# SPDX-License-Identifier: Apache-2.0
"""保存KV分配候选的原三方全部窗口，门槛相对采用Score后的新正式版本。"""

import json
from pathlib import Path

from report_key_block import summarize

HERE = Path(__file__).resolve().parent


def main():
    task = json.loads((HERE / "PREFIX_KV_BALANCE_TASK.json").read_text())
    result = summarize(HERE / task["folder"], task["task"])
    result.update(
        reference="source_after_score_split/13259f92",
        candidate="source_prefix_kv_balance",
        b16_new_device_measurement=False,
        b16_evidence="PREFIX_KV_BALANCE_CPU_AUDIT.json：关闭路径完整C++与当前正式相同",
        adopted=False,
        note="只按此轮新正式比较，不将候选与早一轮1166.04跨轮相减计收益",
    )
    (HERE / "PREFIX_KV_BALANCE_PERFORMANCE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
