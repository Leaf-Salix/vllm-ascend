# SPDX-License-Identifier: Apache-2.0
"""直接L1交接的CPU偏序模型。不能替代真实设备缓存/pipe顺序验证。"""

import json
import random
from collections import defaultdict, deque
from pathlib import Path


def programs(valid):
    count = len(valid)
    cube = [("wait_plan", -1)]
    vector = [[("plan", -1)] for _ in range(2)]
    for tick in range(count + 2):
        if tick >= 3 and valid[tick - 3]:
            cube.append(("ack", tick - 3))
        if tick < count and valid[tick]:
            cube.append(("qk", tick))
        if tick >= 2 and valid[tick - 2]:
            cube.append(("pv", tick - 2))
    if count and valid[-1]:
        cube.append(("ack", count - 1))
    for tick in range(count + 3):
        for lane in range(2):
            if 0 < tick <= count and valid[tick - 1]:
                vector[lane].append(("sf", tick - 1))
            if tick >= 3 and valid[tick - 3]:
                vector[lane].append(("merge", tick - 3))
    return [cube] + vector


def simulate(valid, seed, generations=2):
    rng = random.Random(seed)
    events = defaultdict(deque)
    slots = {"kv": {}, "score": {}, "prob": {}, "pv": {}}
    consumed = defaultdict(set)
    actions = 0
    for generation in range(generations):
        queues = [deque(program) for program in programs(valid)]
        while any(queues):
            possible = []
            for actor, queue in enumerate(queues):
                if not queue:
                    continue
                operation, work = queue[0]
                lanes = (0, 1) if actor == 0 else (actor - 1,)
                event = {"wait_plan": 4, "ack": 0, "sf": 1, "pv": 2, "merge": 3}.get(operation)
                if event is None or all(events[(event, lane)] for lane in lanes):
                    possible.append(actor)
            if not possible:
                raise AssertionError(f"死锁：{valid}, {list(map(list, queues))}")
            actor = rng.choice(possible)
            operation, work = queues[actor].popleft()
            token = (generation, work)
            slot = work % 3
            lanes = (0, 1) if actor == 0 else (actor - 1,)
            event = {"wait_plan": 4, "ack": 0, "sf": 1, "pv": 2, "merge": 3}.get(operation)
            if event is not None:
                for lane in lanes:
                    assert events[(event, lane)].popleft() == token, (operation, token, lane)
            if operation == "plan":
                events[(4, actor - 1)].append(token)
            elif operation == "qk":
                for kind in ("score", "pv"):
                    old = slots[kind].get(slot)
                    if old is not None:
                        assert consumed[(kind, old)] == {0, 1}, (kind, "过早复用", token, old)
                slots["kv"][slot] = token
                slots["score"][slot] = token
                for lane in range(2):
                    events[(1, lane)].append(token)
            elif operation == "sf":
                assert slots["score"][slot] == token
                consumed[("score", token)].add(actor - 1)
                slots["prob"][(slot, actor - 1)] = token
                events[(2, actor - 1)].append(token)
            elif operation == "pv":
                assert slots["kv"][slot] == token
                for lane in range(2):
                    assert slots["prob"][(slot, lane)] == token
                slots["pv"][slot] = token
                for lane in range(2):
                    events[(3, lane)].append(token)
            elif operation == "merge":
                assert slots["pv"][slot] == token
                consumed[("pv", token)].add(actor - 1)
                events[(0, actor - 1)].append(token)
            actions += 1
        assert not any(events.values()), "调用结束仍有未消费事件"
    return actions


def address_coverage():
    # 五段必须恰好覆盖128行；真实行与补零行互斥，DMA有效范围不得越页。
    positions = list(range(257)) + [8191, 8192, 131071, 131072, 131077]
    for position in positions:
        length = min(position + 1, 128)
        begin = position - length + 1
        head = begin % 32
        for negative_mask in (0, 1, 3, 10, 31):
            writes = [0] * 128
            for segment in range(5):
                lo, hi = max(segment * 32 - head, 0), min(segment * 32 - head + 32, 128)
                if lo < hi:
                    written = lo
                    if lo < length and not (negative_mask >> segment & 1):
                        written = min(hi, length)
                        assert 0 < written - lo <= 32
                        assert (begin + lo) // 32 == (begin + written - 1) // 32
                        for row in range(lo, written):
                            writes[row] += 1
                    if written < hi:
                        assert 0 < hi - written <= 32
                        for row in range(written, hi):
                            writes[row] += 1
            assert writes == [1] * 128
    return len(positions) * 5


def main():
    actions, cases = 0, 0
    for mask in range(1024):
        valid = [bool(mask >> bit & 1) for bit in range(10)]
        for seed in range(3):
            actions += simulate(valid, seed)
            cases += 1
    rng = random.Random(17)
    for case in range(256):
        valid = [rng.random() < 0.8 for _ in range(20)]
        actions += simulate(valid, case, generations=3)
        cases += 1
    report = {"passed": True, "schedules": cases, "actions": actions,
              "exhaustive_two_query_valid_masks": 1024, "swa_cases": address_coverage(),
              "two_aiv_independent_progress": True, "all_event_credits_drained_each_generation": True,
              "slot_overwrite_checked": True, "device_cache_or_dma_proof": False}
    Path("qk_direct_l1_protocol_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
