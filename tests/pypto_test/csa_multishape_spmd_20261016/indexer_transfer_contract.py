# SPDX-License-Identifier: Apache-2.0
"""比较原1024×2与新2048交接的有效排序段，并枚举两槽通知计数模型。"""

import argparse
import collections
import json
from pathlib import Path


def check():
    checked = 0
    odd_tail = 0
    for valid in range(1, 8193):
        original_steps = (valid + 1023) // 1024
        transfer_steps = (valid + 2047) // 2048
        written = [(big * 2048 + sub * 1024, 1024)
                   for big in range(transfer_steps) for sub in range(2)
                   if big * 2048 + sub * 1024 < valid]
        assert written == [(step * 1024, 1024) for step in range(original_steps)]
        visible_cases = {0, 1, max(valid - 2, 0), max(valid - 1, 0), valid}
        # 满叶时query还可看见下一叶。此时必须由叶边界限制，而非只靠query长度。
        if valid % 1024 == 0:
            visible_cases.add(valid + 4096)
        for visible in visible_cases:
            original = []
            for step in range(original_steps):
                if step % 2 == 1 or step + 1 == original_steps:
                    begin = step // 2 * 2048
                    count = max(min(visible - begin, (step % 2 + 1) * 1024), 0)
                    original.append((begin, count, step < 2))
            merged = []
            for step in range(transfer_steps):
                begin = step * 2048
                count = max(min(visible - begin, min(2048, valid - begin)), 0)
                merged.append((begin, count, step == 0))
                assert begin + count <= written[-1][0] + written[-1][1]
            assert original == merged, (valid, visible, original, merged)
            checked += 1
        odd_tail += original_steps % 2
    models = []
    for steps in range(1, 5):
        start = (0, 0, 0)
        todo = collections.deque([start])
        seen = {start}
        while todo:
            produced, aiv0, aiv1 = state = todo.popleft()
            successors = []
            if produced < steps and produced - min(aiv0, aiv1) < 2:
                successors.append((produced + 1, aiv0, aiv1))
            if aiv0 < produced:
                successors.append((produced, aiv0 + 1, aiv1))
            if aiv1 < produced:
                successors.append((produced, aiv0, aiv1 + 1))
            if not successors:
                assert state == (steps, steps, steps), state
            assert produced - min(aiv0, aiv1) <= 2
            for nxt in successors:
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        models.append({"steps": steps, "reachable_states": len(seen), "terminal": [steps] * 3})
    return {"leaf_valid_range": [1, 8192], "effective_segment_cases": checked,
            "odd_subblock_leaf_counts": odd_tail,
            "all_1024_subblock_writes_match_baseline": True,
            "topk_segment_order_and_effective_counts_match": True,
            "root_first_segment_initialization_matches": True,
            "two_slot_counter_models": models,
            "scope": "原/新有效段对照与通知计数模型；不代替硬件FFTS、DMA顺序、精度验证"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
