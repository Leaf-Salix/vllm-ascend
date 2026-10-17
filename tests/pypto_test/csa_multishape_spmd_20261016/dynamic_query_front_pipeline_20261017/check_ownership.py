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


def query_work(tokens):
    """独立枚举每个N256列块、有效M128行块以及每个token/head的唯一写者。"""
    qb = set()
    qb_owners = set()
    for worker in range(24):
        for column in range(worker, 128, 24):
            for row in range(0, tokens, 128):
                key = (row, column)
                assert key not in qb
                qb.add(key)
                qb_owners.add(worker)
    assert qb == {(row, column) for row in range(0, tokens, 128) for column in range(128)}
    head_tile = 64 if tokens >= 384 and tokens % 384 == 0 else (
        16 if tokens >= 96 and tokens % 96 == 0 else 4)
    groups = 64 // head_tile
    output = set()
    dq_owners = set()
    for worker in range(48):
        for work in range(worker, ((tokens + 7) // 8) * groups, 48):
            row0 = work // groups * 8
            head0 = work % groups * head_tile
            if row0 + 8 > tokens:
                assert head_tile == 4, "尾分支只处理4 heads必须由策略保证"
            for row in range(row0, min(row0 + 8, tokens)):
                for head in range(head0, head0 + head_tile):
                    assert (row, head) not in output
                    output.add((row, head))
                    dq_owners.add(worker)
    assert output == {(row, head) for row in range(tokens) for head in range(64)}
    return qb_owners, dq_owners, head_tile, len(qb)


def check(tokens):
    tiles8 = (tokens + 7) // 8
    tiles16 = (tokens + 15) // 16
    qb_owners, dq_owners, head_tile, qb_tiles = query_work(tokens)
    true_owners = {
        1: {24 + tile % 48 for tile in range(tiles8)},
        2: {logical % 24 for logical in range(tiles16 * 4)},
        3: {24 + tile % 48 for tile in range(tiles16)},
        4: {24 + tile % 48 for tile in range(tiles8)},
        5: {24 + tile % 48 for tile in range(tiles8)},
        6: {24 + tile % 48 for tile in range(tiles8)},
        7: actual_qa(tokens)[0],
        8: {24 + tile % min(tiles8, 48) for tile in range(tiles8)},
        9: {24 + tile % min(tiles8, 48) for tile in range(tiles8)},
        10: qb_owners,
        11: {24 + worker for worker in dq_owners},
    }
    expected = {
        1: set(range(24, 24 + min(tiles8, 48))),
        2: set(range(min(tiles16 * 4, 24))),
        3: set(range(24, 24 + min(tiles16, 48))),
        4: set(range(24, 24 + min(tiles8, 48))),
        6: set(range(24, 24 + min(tiles8, 48))),
        7: set(range(min(8 * min(3, max(1, tokens // (32 if tokens < 128 else 64))), 24))),
    }
    expected.update({9: set(range(24, 24 + min(tiles8, 48))), 10: set(range(24)),
                     11: set(range(24, 24 + min(tiles8 * (64 // head_tile), 48)))})
    assert all(expected[phase] == true_owners[phase] for phase in expected)
    # 每个节点表示本核 phase 已发布。数学阶段只消费严格更早的逻辑 phase。
    edges = set()
    nodes = set()
    for core in range(72):
        phases = [0, 2, 7, 10] if core < 24 else [0, 1, 3, 4, 5, 6, 8, 9, 11]
        nodes.update((core, phase) for phase in phases)
        edges.update(((core, a), (core, b)) for a, b in zip(phases, phases[1:]))
        waits = [(2, 1), (7, 6), (10, 8)] if core < 24 else [
            (3, 2), (4, 3), (4, 1), (6, 4), (8, 7), (11, 10), (11, 9), (11, 8)]
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
            "qa_tiles": actual_qa(tokens)[1], "qb_tiles": qb_tiles, "dequant_head_tile": head_tile}


def main():
    tree = ast.parse((HERE / "probe.py").read_text())
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "entry")
    annotations = {arg.arg: ast.unparse(arg.annotation) for arg in entry.args.args}
    assert len(annotations) == 35
    assert "QR_ROWS" in annotations["qr_matmul"] and "P8" in annotations["qr_scale_pad"]
    assert "QPROJ_ROWS" in annotations["q_proj"]
    assert "pl.NZ" in annotations["wq_a"] and "pl.NZ" in annotations["wq_b"]
    assert calls.count("begin_invocation") == 2
    assert calls.count("pl.spmd") == 1 and calls.count("pl.cluster") == 1
    assert not any("syncall" in name or "submit" in name for name in calls)
    counts = list(range(1, 385)) + [511, 512, 513, 769, 1024, 1537]
    result = [check(tokens) for tokens in counts]
    examples = [row for row in result if row["tokens"] in (6, 24, 48, 96, 144, 168, 192, 513, 1537)]
    report = {"passed": True, "cpu_only": True, "device_verified": False,
              "scope": "HC/QA/QB和query有效token/head唯一覆盖、真实owner和phase无环；不证明硬件可见性",
              "token_case_count": len(result), "token_range": "1..384,511,512,513,769,1024,1537",
              "entry_spmd_count": 1, "cluster_count": 1, "per_role_begin_count": 1,
              "begin_cores": 72, "phase_count_including_generation": 12,
              "target_S6_tokens": [24, 48, 96, 144, 168, 192],
              "example_cases": examples}
    (HERE / "ownership.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "example_cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
