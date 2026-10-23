# SPDX-License-Identifier: Apache-2.0
"""Check phase-level waits and three-slot reuse for early PV reads.

This models phase completion and event pairing, not DMA duration or generated
local fences. Codegen and real-device checks remain necessary.
"""

import collections
import itertools
import random


def check(mask, early_kv, prefetch=True, prefetch_delta=3, omit_pv_wait=False):
    edges, ids, last = [], {}, {}

    def add(role, action, work):
        key = role, action, work
        assert key not in ids
        index = len(edges)
        ids[key] = index
        edges.append(set())
        if role in last:
            edges[last[role]].add(index)
        last[role] = index
        return index

    def valid(work):
        return 0 <= work < len(mask) and mask[work]

    for tick in range(len(mask) + 2):
        if valid(tick):
            for action in ("wait_kv0", "wait_kv1", "read_kv", "write_score", "notify_score"):
                add("C", action, tick)
        work = tick - 2
        if valid(work):
            for action in ("wait_prob0", "wait_prob1", "read_prob", "write_pv", "notify_pv"):
                add("C", action, work)
    for lane in range(2):
        role = f"V{lane}"
        for tick in range(len(mask) + 3):
            if valid(tick):
                add(role, "write_kv", tick)
            if prefetch and valid(tick - prefetch_delta):
                work = tick - prefetch_delta
                if not omit_pv_wait:
                    add(role, "wait_pv", work)
                add(role, "read_pv", work)
            work = tick - 1
            if early_kv:
                if valid(work):
                    add(role, "wait_score", work)
                if valid(tick):
                    add(role, "notify_kv", tick)
            if valid(work):
                if not early_kv:
                    add(role, "wait_score", work)
                for action in ("read_score", "write_stats", "write_prob", "notify_prob"):
                    add(role, action, work)
            if not early_kv and valid(tick):
                add(role, "notify_kv", tick)
            work = tick - 3
            if valid(work):
                if not prefetch:
                    add(role, "wait_pv", work)
                    add(role, "read_pv", work)
                add(role, "use_pv", work)
    for work, active in enumerate(mask):
        if not active:
            continue
        for lane in range(2):
            role = f"V{lane}"
            for producer, consumer in (
                ((role, "notify_kv", work), ("C", f"wait_kv{lane}", work)),
                (("C", "notify_score", work), (role, "wait_score", work)),
                ((role, "notify_prob", work), ("C", f"wait_prob{lane}", work)),
            ):
                edges[ids[producer]].add(ids[consumer])
            if (role, "wait_pv", work) in ids:
                edges[ids["C", "notify_pv", work]].add(ids[role, "wait_pv", work])
    indegree = [0] * len(edges)
    for children in edges:
        for child in children:
            indegree[child] += 1
    queue = collections.deque(i for i, value in enumerate(indegree) if value == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in edges[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    assert len(order) == len(edges), "Event dependency cycle"
    reachable = [0] * len(edges)
    for node in reversed(order):
        for child in edges[node]:
            reachable[node] |= (1 << child) | reachable[child]

    def before(first, second):
        assert reachable[ids[first]] & (1 << ids[second]), (first, second)

    active = [work for work, flag in enumerate(mask) if flag]
    for work in active:
        for lane in range(2):
            role = f"V{lane}"
            before((role, "write_kv", work), ("C", "read_kv", work))
            before(("C", "write_score", work), (role, "read_score", work))
            before((role, "write_prob", work), ("C", "read_prob", work))
            before(("C", "write_pv", work), (role, "read_pv", work))
            before((role, "read_pv", work), (role, "use_pv", work))
            before((role, "write_stats", work), (role, "use_pv", work))
        successors = [other for other in active if other > work and other % 3 == work % 3]
        if successors:
            following = successors[0]
            for lane in range(2):
                role = f"V{lane}"
                before(("C", "read_kv", work), (role, "write_kv", following))
                before((role, "read_score", work), ("C", "write_score", following))
                before(("C", "read_prob", work), (role, "write_prob", following))
                before((role, "use_pv", work), ("C", "write_pv", following))
                before((role, "use_pv", work), (role, "write_stats", following))
    return len(edges)


def verify_all():
    patterns = [(), (False,), (True,)]
    patterns.extend(itertools.product((False, True), repeat=10))
    randomizer = random.Random(20261023)
    patterns.extend(tuple(bool(randomizer.getrandbits(1)) for _ in range(20)) for _ in range(64))
    patterns.append((True,) * 20)
    nodes = runs = 0
    for mask in patterns:
        for early_kv in (False, True):
            for prefetch in (False, True):
                nodes += check(mask, early_kv, prefetch)
                runs += 1
    controls = 0
    for kwargs in ({"prefetch_delta": 2}, {"omit_pv_wait": True}):
        try:
            check((True,) * 5, False, **kwargs)
        except AssertionError:
            controls += 1
        else:
            raise AssertionError(f"Unsafe negative control passed: {kwargs}")
    return dict(
        patterns=len(patterns),
        phase_graphs=runs,
        total_nodes=nodes,
        negative_controls=controls,
        event_cycles_absent=True,
        phase_level_raw_war_and_stats_checks=True,
        limitation="Phase/event model only; async fences and UB lifetimes still need codegen/device checks",
    )


if __name__ == "__main__":
    import json

    print(json.dumps(verify_all(), indent=2))
