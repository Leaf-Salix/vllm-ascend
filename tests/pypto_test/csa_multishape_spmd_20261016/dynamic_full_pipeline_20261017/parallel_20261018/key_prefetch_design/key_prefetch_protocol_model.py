# SPDX-License-Identifier: Apache-2.0
"""CPU 协议模型：遍历一组 AIC/双 AIV 的有限交错，验证双槽生命周期。

没有 PyPTO/NPU 代码、没有编译。prepare 原子动作抽象 MTE3 完成后 KEY_READY；
consume 抽象 SCORE_READY 后载入 scale/score 再发送 MTE2 ACK。
不模拟真实 DMA 时长、缓存一致性或编译器指令重排，不能代替设备验证。
"""

import argparse
import json
from collections import deque
from pathlib import Path


def programs(lengths):
    cube, vector, labels = [], [], {}
    token = 0
    for leaf, count in enumerate(lengths):
        current = list(range(token, token + count))
        for step, item in enumerate(current):
            labels[item] = (leaf, step)
            cube.append(("key", item, step % 2, current[step - 2] if step >= 2 else -1))
            cube.append(("score", item, step % 2, -1))
        if count:
            cube.append(("drain", current[-1], -1, current[-2] if count > 1 else -1))
            vector.append(("prepare", current[0], 0, -1))
        for step, item in enumerate(current):
            if step + 1 < count:
                vector.append(("prepare", current[step + 1], (step + 1) % 2, -1))
            vector.append(("consume", item, step % 2, -1))
        token += count
    return (cube, vector, vector), labels


def explore(lengths, shared_scale):
    code, labels = programs(lengths)
    # state: PCs, key-tag[2 lanes * 2 slots], score-tag[2], prepared[2],
    #        key-read mask, score-ready mask, score-read/ACK mask[2].
    initial = ((0, 0, 0), (-1,) * 4, (-1,) * 2, (0, 0), 0, 0, (0, 0))
    queue = deque([initial])
    parents = {initial[0]: None}
    terminal = 0
    max_pending_key = [0, 0]

    def trace(pcs, actor, action):
        result = [(actor, action)]
        while parents[pcs] is not None:
            previous, old_actor, old_action = parents[pcs]
            result.append((old_actor, old_action))
            pcs = previous
        result.reverse()
        return [dict(actor=("AIC", "AIV0", "AIV1")[a], action=x[0], leaf_step=labels[x[1]]) for a, x in result]

    while queue:
        state = queue.popleft()
        pcs, key_tags, score_tags, prepared, key_read, ready, ack = state
        if all(pcs[a] == len(code[a]) for a in range(3)):
            terminal += 1
            continue
        advanced = 0
        for actor in range(3):
            if pcs[actor] == len(code[actor]):
                continue
            action = kind, item, slot, previous = code[actor][pcs[actor]]
            bit = 1 << item
            keys, scores, prep, acks = list(key_tags), list(score_tags), list(prepared), list(ack)
            reads, produced = key_read, ready
            hazard = None
            if kind == "key":
                if not all(p & bit for p in prep):
                    continue
                if previous >= 0 and not all(a & (1 << previous) for a in acks):
                    continue
                if any(keys[lane * 2 + slot] != item for lane in range(2)):
                    hazard = "AIC 读取被提前覆盖的 Key"
                reads |= bit
            elif kind == "score":
                old = scores[slot]
                if old >= 0 and not all(a & (1 << old) for a in acks):
                    hazard = "AIC 覆写尚未由双 AIV 读取的 Score"
                scores[slot] = item
                produced |= bit
            elif kind == "drain":
                needed = bit | (1 << previous if previous >= 0 else 0)
                if not all(a & needed == needed for a in acks):
                    continue
            elif kind == "prepare":
                lane = actor - 1
                old = keys[lane * 2 + slot]
                if old >= 0 and not (reads & (1 << old)):
                    hazard = "AIV 覆写 AIC 尚未读完的 Key"
                keys[lane * 2 + slot] = item
                prep[lane] |= bit
            else:
                lane = actor - 1
                if not (ready & bit):
                    continue
                if scores[slot] != item:
                    hazard = "AIV 读取被提前覆盖的 Score"
                scale_lanes = range(2) if shared_scale else (lane,)
                if any(keys[source * 2 + slot] != item for source in scale_lanes):
                    hazard = "AIV 读取的 scale 已被快 lane 下一轮覆盖"
                acks[lane] |= bit
            if hazard:
                return dict(
                    passed=False,
                    shared_scale=shared_scale,
                    leaf_lengths=lengths,
                    states_seen=len(parents),
                    hazard=hazard,
                    trace=trace(pcs, actor, action),
                )
            advanced += 1
            next_pcs = list(pcs)
            next_pcs[actor] += 1
            next_pcs = tuple(next_pcs)
            for lane in range(2):
                max_pending_key[lane] = max(max_pending_key[lane], (prep[lane] & ~reads).bit_count())
            if next_pcs not in parents:
                parents[next_pcs] = (pcs, actor, action)
                queue.append((next_pcs, tuple(keys), tuple(scores), tuple(prep), reads, produced, tuple(acks)))
        if not advanced:
            return dict(
                passed=False,
                shared_scale=shared_scale,
                leaf_lengths=lengths,
                states_seen=len(parents),
                hazard="无可推进角色且未完成：协议死锁",
            )
    return dict(
        passed=True,
        shared_scale=shared_scale,
        leaf_lengths=lengths,
        states_seen=len(parents),
        terminal_states=terminal,
        max_pending_key_per_lane=max_pending_key,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [[count] for count in range(1, 9)] + [[1, 1], [1, 2], [2, 1], [3, 3], [1, 3, 1], [0, 2, 0, 3], [6, 5]]
    private = [explore(case, shared_scale=False) for case in cases]
    shared = [explore(case, shared_scale=True) for case in ([3], [1, 1])]
    assert all(row["passed"] for row in private)
    assert all(not row["passed"] and "scale" in row["hazard"] for row in shared)
    result = dict(
        passed=True,
        scope="单 AIC/双 AIV 的有限交错协议模型；没有编译和设备验证",
        private_scale=private,
        shared_scale_counterexamples=shared,
        assumptions=[
            "两 AIV 与 AIC 逐项使用同一 device group/leaf/step 计划；无效 leaf 全部跳过",
            "KEY_READY 来自 MTE3 完成；SCORE_READY 来自 FIX 完成；ACK 来自 Score/scale 读取完成",
            "FFTS mode2 的双 lane 汇合沿用现有 Score ACK 合同；模型不模拟硬件实现",
            "prepare 和consume分别抽象完成后的可见操作，不模拟缓存/别名/编译器顺序",
            "叶边界也代表已排空的代际边界；外层运行时仍要求前次调用完成",
        ],
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "private_cases": len(private),
                "states": sum(r["states_seen"] for r in private),
                "shared_scale_hazards": [r["hazard"] for r in shared],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
