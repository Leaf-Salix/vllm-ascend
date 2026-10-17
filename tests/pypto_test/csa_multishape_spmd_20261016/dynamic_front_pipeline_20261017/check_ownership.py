# SPDX-License-Identifier: Apache-2.0
"""纯 CPU：独立枚举真实逻辑分块，验证前段 owner 集合与发布/等待无环。

不模拟 Scalar 缓存、流水完成、数值或硬件调度，不能作为设备同步证明。
"""
import ast
import json
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent


def actual_qa(tokens):
    dense = 32 if tokens < 128 else 64
    groups = min(3, max(1, tokens // dense))
    full = tokens // dense * dense
    padded = (tokens + 15) // 16 * 16
    owned = defaultdict(list)
    coverage = set()
    for worker in range(24):
        for logical in range(worker, 8 * groups, 24):
            col = logical % 8
            group = logical // 8
            spans = [(row, dense) for row in range(group * dense, full, groups * dense)]
            spans += [(row, 16) for row in range(full + group * 16, padded, groups * 16)]
            for first, rows in spans:
                owned[worker].append((first, rows, col))
                for row in range(first, first + rows):
                    assert (row, col) not in coverage
                    coverage.add((row, col))
    assert coverage == {(row, col) for row in range(padded) for col in range(8)}
    return set(owned), sum(map(len, owned.values()))


def check(tokens):
    tiles8 = (tokens + 7) // 8
    tiles16 = (tokens + 15) // 16
    true_owners = {
        1: {24 + tile % 48 for tile in range(tiles8)},
        2: {logical % 24 for logical in range(tiles16 * 4)},
        3: {24 + tile % 48 for tile in range(tiles16)},
        4: {24 + tile % 48 for tile in range(tiles8)},
        5: {24 + tile % 48 for tile in range(tiles8)},
        6: {24 + tile % 48 for tile in range(tiles8)},
        7: actual_qa(tokens)[0],
        8: {24 + tile % min(tiles8, 48) for tile in range(tiles8)},
    }
    expected = {
        1: set(range(24, 24 + min(tiles8, 48))),
        2: set(range(min(tiles16 * 4, 24))),
        3: set(range(24, 24 + min(tiles16, 48))),
        4: set(range(24, 24 + min(tiles8, 48))),
        6: set(range(24, 24 + min(tiles8, 48))),
        7: set(range(min(8 * min(3, max(1, tokens // (32 if tokens < 128 else 64))), 24))),
    }
    assert all(expected[phase] == true_owners[phase] for phase in expected)
    # 每个节点表示本核 phase 已发布。数学阶段只消费严格更早的逻辑 phase。
    edges = set()
    nodes = set()
    for core in range(72):
        phases = [0, 2, 7] if core < 24 else [0, 1, 3, 4, 5, 6, 8]
        nodes.update((core, phase) for phase in phases)
        edges.update(((core, a), (core, b)) for a, b in zip(phases, phases[1:]))
        waits = [(2, 1), (7, 6)] if core < 24 else [(3, 2), (4, 3), (4, 1), (6, 4), (8, 7)]
        for consumer_phase, producer_phase in waits:
            for producer in true_owners[producer_phase]:
                edges.add(((producer, producer_phase), (core, consumer_phase)))
    degrees = dict.fromkeys(nodes, 0)
    successors = defaultdict(list)
    for before, after in edges:
        assert before in nodes and after in nodes
        degrees[after] += 1
        successors[before].append(after)
    ready = deque(node for node, degree in degrees.items() if degree == 0)
    visited = 0
    while ready:
        current = ready.popleft()
        visited += 1
        for after in successors[current]:
            degrees[after] -= 1
            if degrees[after] == 0:
                ready.append(after)
    assert visited == len(nodes), (tokens, visited, len(nodes))
    return {"tokens": tokens, "owner_counts": {str(k): len(v) for k, v in true_owners.items()},
            "nodes": len(nodes), "edges": len(edges), "acyclic": True,
            "qa_tiles": actual_qa(tokens)[1]}


def main():
    tree = ast.parse((HERE / "probe.py").read_text())
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert calls.count("begin_invocation") == 2
    assert calls.count("pl.spmd") == 1 and calls.count("pl.cluster") == 1
    assert not any("syncall" in name or "submit" in name for name in calls)
    counts = list(range(1, 385)) + [511, 512, 513, 769, 1024, 1537]
    result = [check(tokens) for tokens in counts]
    examples = [row for row in result if row["tokens"] in (6, 24, 48, 96, 144, 168, 192, 513, 1537)]
    report = {"passed": True, "cpu_only": True, "device_verified": False,
              "scope": "真实tile覆盖、活跃owner与抽象phase依赖无环；不证明硬件可见性",
              "token_case_count": len(result), "token_range": "1..384,511,512,513,769,1024,1537",
              "entry_spmd_count": 1, "cluster_count": 1, "per_role_begin_count": 1,
              "begin_cores": 72, "target_S6_tokens": [24, 48, 96, 144, 168, 192],
              "example_cases": examples}
    (HERE / "ownership.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "example_cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
