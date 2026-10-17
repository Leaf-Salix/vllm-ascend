"""有界只读分工模型：不导入PyPTO，不编译，不模拟硬件时延或FFTS。"""

import json
from collections import Counter
from pathlib import Path


here = Path(__file__).resolve().parent
groups, work = [], []
for g in range(8):
    members = []
    for p in range(3):
        members.append({"kind": "pair", "owner": 3*g+p, "group": g,
                        "n128_blocks": [2*p, 2*p+1]})
    for s in range(2):
        members.append({"kind": "single", "owner": 2*g+s, "group": g,
                        "n128_blocks": [6+s]})
    assert sorted(n for m in members for n in m["n128_blocks"]) == list(range(8))
    owners = sorted({m["owner"] for m in members})
    old = sorted({(8*g+n) % 24 for n in range(8)})
    groups.append({"group": g, "members": members, "new_wait_owners": owners,
                   "old_wait_owners": old,
                   "old_wait_misses_new_producers": sorted(set(owners)-set(old)),
                   "old_waits_for_nonpublishing_workers": sorted(set(old)-set(owners))})
    work.extend(members)
per_core = []
for worker in range(24):
    mine = [m for m in work if m["owner"] == worker]
    assert sum(len(m["n128_blocks"]) for m in mine) == (3 if worker < 16 else 2)
    assert sum(m["kind"] == "pair" for m in mine) == 1
    per_core.append({"worker": worker, "members": sorted(mine, key=lambda m: m["group"]),
                     "n128_output_count": sum(len(m["n128_blocks"]) for m in mine)})
cases = []
for tokens in range(1, 97):
    # 模型不分配Tile，只验证各真实token/列块恰好有一个写者。
    counts = Counter((g, t, n) for g in range(8) for t in range(tokens)
                     for m in groups[g]["members"] for n in m["n128_blocks"])
    assert len(counts) == 8*tokens*8 and set(counts.values()) == {1}
    q = (tokens+31)//32
    quant_workers = {(g, row): (g*q+row) % 48 for g in range(8) for row in range(q)}
    # 量化任务和最后padding owner不变；所有消费者读整1024列，须等完整owner并集。
    for g in range(8):
        required = {m["owner"] for m in groups[g]["members"]}
        assert required == set(groups[g]["new_wait_owners"])
        for row in range(q):
            assert 0 <= quant_workers[g,row] < 48
    cases.append({"tokens": tokens, "row_blocks": 1, "quant_per_group": q,
                  "ob_rows": 32 if tokens <= 32 else 96,
                  "last_quant_owner_by_group": [(g*q+q-1) % 48 for g in range(8)],
                  "coverage_unique": True, "wait_union_complete": True})

# 分层DAG：这里仅证明业务phase没有环，不替代Scalar缓存协议或硬件事件验证。
nodes = ["packed"]
edges = []
for g in range(8):
    for owner in groups[g]["new_wait_owners"]:
        a = f"OA_g{g}_w{owner}"
        nodes.append(a); edges.append(("packed", a)); edges.append((a, f"quant_g{g}"))
    nodes.extend([f"quant_g{g}", f"OB_g{g}"])
    edges.extend([(f"quant_g{g}", f"OB_g{g}"), (f"OB_g{g}", "post")])
nodes.append("post")
for core in per_core:
    owned_groups = sorted({m["group"] for m in core["members"]})
    edges.extend((f"OA_g{a}_w{core['worker']}", f"OA_g{b}_w{core['worker']}")
                 for a,b in zip(owned_groups, owned_groups[1:]))
    # 当前源码只在本核全部OA组完成后进入OB，保留这一顺序。
    for g in range(8):
        if core["worker"] in {(g*8+n) % 24 for n in range(8)}:
            edges.append((f"OA_g{owned_groups[-1]}_w{core['worker']}", f"OB_g{g}"))
incoming = {n: 0 for n in nodes}
adj = {n: [] for n in nodes}
for a,b in set(edges):
    incoming[b] += 1; adj[a].append(b)
ready = [n for n in nodes if incoming[n] == 0]
seen = []
while ready:
    n = ready.pop(); seen.append(n)
    for dest in adj[n]:
        incoming[dest] -= 1
        if incoming[dest] == 0: ready.append(dest)
assert len(seen) == len(nodes)
(here / "owner_model.json").write_text(json.dumps({
    "scope": "源码合同有限模型，无编译/NPU/硬件时延证明", "groups": groups,
    "per_core": per_core, "cases": cases, "phase_dag_acyclic": True,
    "phase_nodes": len(nodes), "phase_edges": len(set(edges)),
    "all_64_n128_outputs_covered_once": True,
    "T0_behavior": "正式policy拒绝；helper须无任务不发布，不纳入本候选有效区间",
    "T_ge97_behavior": "保持原OA区间/数学/owner，不能外推本pair映射",
}, ensure_ascii=False, indent=2) + "\n")
print("T=1..96 coverage/owner/DAG model PASS; no compiler or device")
