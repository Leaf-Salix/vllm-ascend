#!/usr/bin/env python3
"""只读复算Attention地址guard候选；不导入设备库、不执行内核。"""

import json
from pathlib import Path

from report_prefetch_quant_round import summarize


def main():
    evidence = Path(__file__).resolve().parent / "attention_guard_elision_evidence"
    actual = summarize(evidence)
    archived = json.loads((evidence / "DEVICE_RESULTS.json").read_text())
    # 决策文字由审阅记录维护，统计字段必须能从原始五窗完整复算。
    assert all(archived[key] == value for key, value in actual.items())
    print(json.dumps(actual, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
