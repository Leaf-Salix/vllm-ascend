# SPDX-License-Identifier: Apache-2.0
"""按消费者核旋转等待表，分散最先轮询的生产者；不改变依赖集合。"""

import collections


def rotate_waits(layout):
    if layout["protocol"]["record_width"] != 6:
        raise ValueError("等待次序旋转目前只支持普通六字段记录")
    records = layout["workspace"]["records"]
    waits = layout["workspace"]["waits"]
    covered = set()
    changed = 0
    total = 0
    for batches in layout["batches"].values():
        for batch in batches:
            for row in batch["members"]:
                start = batch["record_offset_words"] + row["core"] * 6
                base, count = records[start + 3:start + 5]
                before = [tuple(waits[2 * i:2 * i + 2]) for i in range(base, base + count)]
                expected = [(int(core), seq) for core, seq in row["wait_for_core_progress"].items()]
                if collections.Counter(before) != collections.Counter(expected):
                    raise ValueError("等待表与已证明的依赖清单不一致")
                indices = set(range(base, base + count))
                if indices & covered:
                    raise ValueError("等待表片段被多个成员共享，不能原地旋转")
                covered.update(indices)
                if count:
                    shift = row["core"] % count
                    after = before[shift:] + before[:shift]
                    if collections.Counter(after) != collections.Counter(before):
                        raise ValueError("旋转丢失或重复了前驱")
                    waits[2 * base:2 * (base + count)] = [v for pair in after for v in pair]
                    changed += int(shift != 0)
                total += count
    if covered != set(range(len(waits) // 2)):
        raise ValueError("存在未审查的等待表片段")
    return {"wait_edges": total, "changed_members": changed,
            "same_dependency_multisets": True, "device_code_unchanged": True,
            "strategy": "按消费者核号循环旋转，每轮仍等待所有原前驱，不清除信号"}
