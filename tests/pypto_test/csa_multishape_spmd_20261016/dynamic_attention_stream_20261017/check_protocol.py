# SPDX-License-Identifier: Apache-2.0
"""三角色随机推进模型：验证事件顺序、三槽数据寿命、失效块及尾排空。

模型假设每次FFTS通知是有序可计数的token，mode2是双AIV聚合/广播，
生产者事件使其之前写入对消费者可见。它验证软件协议，不证明这些A3硬件前提。
固定模型常量3槽/5块，query个数与有效mask变化；不模拟浮点数学或缓存行为。
"""

import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path


def traces(mask, early):
    n = len(mask)
    result = [[], [], []]

    def add(role, kind, *args):
        result[role].append((kind, *args))

    def rw(role, kind, area, w, lanes):
        for lane in lanes:
            add(role, kind, area, w % 3, lane, w)

    for tick in range(n + 2):
        if tick < n and mask[tick]:
            w = tick
            add(0, "wait", "KV", w)
            rw(0, "read", "KV", w, [0, 1])
            rw(0, "write", "L1", w, [0])
            rw(0, "write", "SCORE", w, [0, 1])
            add(0, "set", "SCORE", w)
        if tick >= 2:
            w = tick - 2
            if mask[w]:
                add(0, "wait", "PROB", w)
                rw(0, "read", "PROB", w, [0, 1])
                rw(0, "read", "L1", w, [0])
                rw(0, "write", "PV", w, [0, 1])
                add(0, "set", "PV", w)
    for lane in range(2):
        role = lane + 1
        for tick in range(n + 3):
            current = tick < n and mask[tick]
            previous = 0 < tick <= n and mask[tick - 1]
            if current:
                rw(role, "write", "KV", tick, [lane])
            if early:
                if previous:
                    add(role, "wait", "SCORE", tick - 1)
                if current:
                    add(role, "set", "KV", tick)
            if previous:
                w = tick - 1
                if not early:
                    add(role, "wait", "SCORE", w)
                rw(role, "read", "SCORE", w, [lane])
                for area in ["PROB", "MI", "LI"]:
                    rw(role, "write", area, w, [lane])
                add(role, "set", "PROB", w)
            if current and not early:
                add(role, "set", "KV", tick)
            if tick >= 3:
                w = tick - 3
                if mask[w]:
                    add(role, "wait", "PV", w)
                    for area in ["PV", "MI", "LI"]:
                        rw(role, "read", area, w, [lane])
                if w % 5 == 4:
                    add(role, "publish", w // 5)
    return result


def run(mask, early, seed):
    streams = traces(mask, early)
    pc = [0, 0, 0]
    queues = defaultdict(deque)
    mem = {}
    maxdepth = defaultdict(int)
    published = [[], [], []]
    rng = random.Random(seed)

    def keys(role, event):
        if event in ("KV", "PROB"):
            return [(event, lane) for lane in ([0, 1] if role == 0 else [role - 1])]
        return [(event, lane) for lane in ([0, 1] if role == 0 else [role - 1])]

    while any(pc[i] < len(streams[i]) for i in range(3)):
        ready = []
        for role in range(3):
            if pc[role] == len(streams[role]):
                continue
            op = streams[role][pc[role]]
            if op[0] != "wait" or all(queues[k] for k in keys(role, op[1])):
                ready.append(role)
        assert ready, ("deadlock", mask, early, seed, pc)
        role = rng.choice(ready)
        op = streams[role][pc[role]]
        pc[role] += 1
        if op[0] in ("set", "wait"):
            kind, event, work = op
            for key in keys(role, event):
                if kind == "set":
                    queues[key].append(work)
                    maxdepth[event] = max(maxdepth[event], len(queues[key]))
                else:
                    assert queues[key].popleft() == work, ("event_order", op, role, seed)
        elif op[0] in ("read", "write"):
            kind, area, slot, lane, work = op
            key = (area, slot, lane)
            if kind == "write":
                mem[key] = work
            else:
                assert mem.get(key) == work, ("ring_overwrite", op, mem.get(key), role, seed, pc)
        elif op[0] == "publish":
            published[role].append(op[1])
    assert all(not queue for queue in queues.values()), ("unconsumed_event", mask)
    assert published[1] == published[2] == list(range(len(mask) // 5))
    return maxdepth


def check():
    rng = random.Random(20261017)
    cases = []
    depths = defaultdict(int)
    # 两档T分支和0~8个query/core：覆盖空核、小尾和跨query槽复用，非全shape测试。
    for count in [0, 1, 2, 3, 6, 8]:
        n = count * 5
        patterns = [("all", [True] * n), ("none", [False] * n)]
        if n:
            patterns += [("first", [i == 0 for i in range(n)]), ("last", [i == n - 1 for i in range(n)])]
            patterns += [("random" + str(k), [rng.random() < 0.6 for _ in range(n)]) for k in range(4)]
        for early in [False, True]:
            for name, mask in patterns:
                for seed in range(12):
                    result = run(mask, early, seed)
                    for key, value in result.items():
                        depths[key] = max(depths[key], value)
                cases.append(
                    {
                        "queries_per_core": count,
                        "early_kv": early,
                        "mask": name,
                        "interleavings": 12,
                        "valid_blocks": sum(mask),
                    }
                )
    # 实际T→每组query计数：不假定整除24，也不让空组碰plan或publish。
    ownership = []
    for t in [0, 1, 23, 24, 25, 96, 143, 144, 145, 168, 192]:
        counts = [(t + 23 - core) // 24 for core in range(24)]
        assigned = [core + q * 24 for core in range(24) for q in range(counts[core])]
        assert sorted(assigned) == list(range(t))
        ownership.append({"T": t, "counts": counts, "early_kv": t >= 144})
    return {
        "passed": True,
        "case_count": len(cases),
        "random_executions": len(cases) * 12,
        "cases": cases,
        "ownership": ownership,
        "maximum_pending_tokens": dict(depths),
        "model_assumptions": [
            "FFTS mode2 aggregates both AIV producers and broadcasts AIC producer to both lanes",
            "channels are ordered counted notifications, not binary booleans",
            "event wait makes producer prior stores visible",
        ],
        "limitations": [
            "CPU protocol model, not A3 FFTS capacity/cache/pipe validation",
            "No NPU precision or performance",
            "Model retains canonical repeated-event ordering; maximum pending counts require hardware support",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "ownership")}, ensure_ascii=False))
