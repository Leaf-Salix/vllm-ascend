# SPDX-License-Identifier: Apache-2.0
"""检查提前KV通知的实际生成代码，禁止未改变提交顺序的候选上卡。"""

import argparse
import json
import re
from pathlib import Path

from qk_phase_clock import close_brace


def audit(directory):
    source = (Path(directory) / "csa_static_group.cpp").read_text()
    function = re.search(r"^AICORE void csa_static_aiv\([^\n]+\) \{", source, re.M)
    if function is None:
        raise ValueError("未找到AIV入口")
    body = source[function.end():close_brace(source, function.end() - 1)]
    blocks = []
    for match in re.finditer(r"^  if \([^\n]+\) \{", body, re.M):
        part = body[match.start():close_brace(body, match.end() - 1) + 1]
        if "// pto: %done_30__tile" in part:
            blocks.append(part)
    if len(blocks) != 1:
        raise ValueError("QK/PV阶段不唯一")
    segment = blocks[0]
    waits = list(re.finditer(r"wait_flag_dev\(", segment))
    notifications = list(re.finditer(r"ffts_cross_core_sync\(", segment))
    if len(waits) != 2 or len(notifications) != 2:
        raise ValueError("原有两个等待/两个通知位点改变")
    rowmax = segment.index("TROWMAX(")
    rowsum = segment.index("TROWSUM(")
    if not (waits[0].start() < notifications[0].start() < rowmax < rowsum
            < notifications[1].start() < waits[1].start()):
        raise ValueError("未保持消费旧Score→通知新KV→softmax→通知Prob→等待PV顺序")
    if segment.count("pipe_barrier(PIPE_ALL)") != 9:
        raise ValueError("原有完整流水屏障数改变")
    points = {"wait_previous_score": waits[0].start(), "notify_current_kv": notifications[0].start(),
              "softmax_rowmax": rowmax, "softmax_rowsum": rowsum,
              "notify_probability": notifications[1].start(), "wait_pv": waits[1].start()}
    return {"artifact": str(Path(directory).resolve()),
            "ordered_sites_line_in_stage": {k: segment.count("\n", 0, v) + 1 for k, v in points.items()},
            "ffts_wait_sites": 2, "ffts_notify_sites": 2, "pipe_all_sites": 9,
            "scope": "提交顺序和位点检查，不是设备正确性或性能证据"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.artifact)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
