# SPDX-License-Identifier: Apache-2.0
"""两query/全部mask下检查FFTS有向依赖与三槽复用；不替代真实硬件验证。"""
import json
from collections import defaultdict, deque
from pathlib import Path


def check(valid, early_kv, pv_first):
    sequences = [[] for _ in range(3)]
    total = len(valid)

    def append(role, kind, work):
        sequences[role].append((role, kind, work))

    def pv(tick):
        work = tick - 2
        if 0 <= work < total and valid[work]:
            for kind in ("wait_prob", "pv", "notify_pv"):
                append(0, kind, work)

    def qk(tick):
        if tick < total and valid[tick]:
            for kind in ("wait_kv", "load_key", "qk", "notify_score"):
                append(0, kind, tick)

    for tick in range(total + 2):
        if pv_first:
            pv(tick)
            qk(tick)
        else:
            qk(tick)
            pv(tick)
    for lane in (1, 2):
        for tick in range(total + 3):
            current = tick < total and valid[tick]
            previous = 0 < tick <= total and valid[tick - 1]
            if current:
                append(lane, "gather_kv", tick)
            if early_kv and previous:
                append(lane, "wait_score", tick - 1)
            if early_kv and current:
                append(lane, "notify_kv", tick)
            if previous:
                if not early_kv:
                    append(lane, "wait_score", tick - 1)
                append(lane, "softmax", tick - 1)
                append(lane, "notify_prob", tick - 1)
            if not early_kv and current:
                append(lane, "notify_kv", tick)
            if 3 <= tick < total + 3 and valid[tick - 3]:
                append(lane, "merge_pv", tick - 3)

    nodes = [node for sequence in sequences for node in sequence]
    index = {node: number for number, node in enumerate(nodes)}
    successors = defaultdict(set)
    indegree = {node: 0 for node in nodes}

    def edge(a, b):
        if b not in successors[a]:
            successors[a].add(b)
            indegree[b] += 1

    for sequence in sequences:
        for a, b in zip(sequence, sequence[1:]):
            edge(a, b)
    for work, active in enumerate(valid):
        if active:
            for lane in (1, 2):
                edge((lane, "notify_kv", work), (0, "wait_kv", work))
                edge((0, "notify_score", work), (lane, "wait_score", work))
                edge((lane, "notify_prob", work), (0, "wait_prob", work))
                edge((0, "notify_pv", work), (lane, "merge_pv", work))
    ready = deque(node for node, count in indegree.items() if count == 0)
    ancestors = {node: 0 for node in nodes}
    seen = 0
    while ready:
        node = ready.popleft()
        seen += 1
        for child in successors[node]:
            ancestors[child] |= ancestors[node] | (1 << index[node])
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    assert seen == len(nodes), (valid, early_kv, pv_first, "event_cycle")

    def before(a, b):
        assert ancestors[b] & (1 << index[a]), (valid, early_kv, pv_first, "reuse_without_order", a, b)

    for slot in range(3):
        uses = [work for work in range(slot, total, 3) if valid[work]]
        for old, new in zip(uses, uses[1:]):
            before((0, "pv", old), (0, "load_key", new))
            for lane in (1, 2):
                before((0, "load_key", old), (lane, "gather_kv", new))
                before((lane, "softmax", old), (0, "qk", new))
                before((0, "pv", old), (lane, "softmax", new))
                before((lane, "merge_pv", old), (0, "pv", new))


def main():
    cases = 0
    for bits in range(1 << 10):
        valid = [bool(bits & (1 << i)) for i in range(10)]
        for early in (False, True):
            for pv_first in (False, True):
                check(valid, early, pv_first)
                cases += 1
    result = dict(passed=True, cases=cases, queries=2, blocks_per_query=5,
                  scope="原/候选顺序、两种AIV交接、全部10位mask；FFTS DAG及三槽先读后复用",
                  device_validated=False, note="抽象操作按完成顺序建边，不能替代硬件pipe/cache语义或真机精度")
    (Path(__file__).resolve().parent / "FIFO_ORDER_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
