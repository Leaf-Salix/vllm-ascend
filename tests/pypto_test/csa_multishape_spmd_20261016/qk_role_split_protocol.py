# SPDX-License-Identifier: Apache-2.0
"""QK双AIV分工的事件/槽复用模型，不模拟DMA或缓存可见性。"""

import json
import random
from collections import defaultdict, deque
from pathlib import Path


def programs(valid):
    count = len(valid)
    cube, gather, math = [], [], []
    for tick in range(count + 2):
        if tick < count and valid[tick]:
            cube.append(("qk", tick))
        if tick >= 2 and valid[tick - 2]:
            cube.append(("pv", tick - 2))
    for tick in range(count + 3):
        if tick < count and valid[tick]:
            gather.append(("gather", tick))
        if 0 < tick <= count and valid[tick - 1]:
            gather.append(("score_ack", tick - 1))
            math.append(("score_ack", tick - 1))
        if tick < count and valid[tick]:
            gather.append(("kv_notify", tick))
            math.append(("kv_notify", tick))
        if 0 < tick <= count and valid[tick - 1]:
            gather.append(("prob_dummy", tick - 1))
            math.append(("softmax", tick - 1))
        if tick >= 3 and valid[tick - 3]:
            gather.append(("pv_dummy", tick - 3))
            math.append(("merge", tick - 3))
    return [cube, gather, math]


def simulate(valid, seed, generations=2):
    rng = random.Random(seed)
    events = defaultdict(deque)
    slots = {kind: {} for kind in ("gm_kv", "l1_kv", "score", "prob", "pv")}
    consumed, peak = defaultdict(set), defaultdict(int)
    actions = 0
    for generation in range(generations):
        queues = [deque(program) for program in programs(valid)]
        while any(queues):
            possible = []
            for actor, queue in enumerate(queues):
                if not queue:
                    continue
                operation, work = queue[0]
                event = {"qk": 0, "pv": 2, "score_ack": 1,
                         "pv_dummy": 3, "merge": 3}.get(operation)
                lanes = (0, 1) if actor == 0 else (actor - 1,)
                if event is None or all(events[event, lane] for lane in lanes):
                    possible.append(actor)
            if not possible:
                raise AssertionError(f"事件死锁：{valid}, {list(map(list, queues))}")
            actor = rng.choice(possible)
            operation, work = queues[actor].popleft()
            token, slot = (generation, work), work % 3
            event = {"qk": 0, "pv": 2, "score_ack": 1,
                     "pv_dummy": 3, "merge": 3}.get(operation)
            lanes = (0, 1) if actor == 0 else (actor - 1,)
            if event is not None:
                for lane in lanes:
                    assert events[event, lane].popleft() == token

            def write(kind):
                old = slots[kind].get(slot)
                if old is not None:
                    assert consumed[kind, old], (kind, "过早复用", token, old)
                slots[kind][slot] = token

            def read(kind):
                assert slots[kind][slot] == token, (kind, token, slots[kind].get(slot))
                consumed[kind, token].add(actor)

            def notify(event_id, lane):
                queue = events[event_id, lane]
                queue.append(token)
                peak[event_id, lane] = max(peak[event_id, lane], len(queue))

            if operation == "gather":
                write("gm_kv")
            elif operation == "kv_notify":
                notify(0, actor - 1)
            elif operation == "qk":
                read("gm_kv")
                write("l1_kv")
                write("score")
                notify(1, 0)
                notify(1, 1)
            elif operation == "softmax":
                read("score")
                write("prob")
                notify(2, 1)
            elif operation == "prob_dummy":
                notify(2, 0)
            elif operation == "pv":
                read("prob")
                read("l1_kv")
                write("pv")
                notify(3, 0)
                notify(3, 1)
            elif operation == "merge":
                read("pv")
            actions += 1
        assert not any(events.values()), "调用末尾残留事件"
    return actions, max(peak.values(), default=0)


def main():
    actions = cases = peak = 0
    for mask in range(1024):
        valid = [bool(mask >> bit & 1) for bit in range(10)]
        for seed in range(3):
            steps, credits = simulate(valid, seed)
            actions += steps
            peak = max(peak, credits)
            cases += 1
    rng = random.Random(20261017)
    for seed in range(256):
        valid = [rng.random() < 0.8 for _ in range(20)]
        steps, credits = simulate(valid, seed, generations=3)
        actions += steps
        peak = max(peak, credits)
        cases += 1
    report = {"passed": True, "schedules": cases, "actions": actions,
              "max_queued_event_credits_in_model": peak,
              "exhaustive_two_query_masks": 1024, "random_four_query_masks": 256,
              "all_five_buffer_slot_reuses_checked": True, "events_drained_each_generation": True,
              "limitation": "原事件采用有序credit抽象；不证明硬件计数容量、DMA完成或缓存可见性"}
    output = Path(__file__).with_name("qk_role_split_protocol_result_v1.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
