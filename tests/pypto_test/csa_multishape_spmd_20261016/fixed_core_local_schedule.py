# SPDX-License-Identifier: Apache-2.0
"""离线局部重排：固定分核，以相邻交换改善已证明成员图的排队。

与member_schedule的三种从头贪心分配不同，本脚本从现选顺序开始；
不改变数学、核组映射或MIX事件。MIX的三个成员按共同开始时刻建模，
防止把组内互相等待误当成可独立执行；每条资源边仍使用真实角色的成本。
结果只是CPU成本模型，不生成设备代码，不替代精度或实际L1结果。
"""

import argparse
import collections
import heapq
import json
import time
from pathlib import Path

from member_schedule import _costs, _estimate, _problem, _resource_parents, ancestor_sets


def run(root, output, seconds=120, rounds=100, defer_main_q=False):
    root, output = Path(root), Path(output)
    schedule = json.loads((root / "b16_cached_row_deps_v1/specialization/bindings/h131072_b16_prepared.json").read_text())
    row_audit = json.loads((root / "b16_cached_row_deps_v1/specialization/control_variants/h131072_b16/row_dependencies/audit.json").read_text())
    nodes, math_parents, bundles, bundle_of, _, _, mixed, names = _problem(schedule, row_audit)
    costs = _costs(schedule, nodes, root / "b16_dynamic_l1_cost.json")
    full_parents = _resource_parents(nodes, math_parents, names)
    member_baseline = _estimate(full_parents, costs)

    # 静态数学边和原物理组的FFTS复用边不动；同核串行边由候选队列重建。
    static = [dict() for _ in bundles]
    indexer = next(task for task, name in names.items() if name == "indexer_score_topk_native_pair")
    qk = next(task for task, name in names.items() if name == "qk_pv")
    indexer_groups = collections.defaultdict(list)
    for i, node in enumerate(nodes):
        if node["task"] == indexer:
            group = node["core"] if node["core"] < 24 else (node["core"] - 24) // 2
            indexer_groups[group].append(i)
    ffts_edges = 0
    for i, original in enumerate(math_parents):
        parents = set(original)
        if nodes[i]["task"] == qk:
            core = nodes[i]["core"]
            group = core if core < 24 else (core - 24) // 2
            parents.update(indexer_groups[group])
            ffts_edges += len(indexer_groups[group])
        for p in parents:
            u, v = bundle_of[p], bundle_of[i]
            if u == v:
                raise ValueError("MIX内部出现外层依赖")
            static[v][u] = max(static[v].get(u, 0.0), costs[p])
    assert ffts_edges == 216
    _, math_ancestors = ancestor_sets([set(p) for p in static])
    queues = collections.defaultdict(list)
    member_on_core = {}
    for i, node in enumerate(nodes):
        queues[node["core"]].append(i)
    for core, ids in queues.items():
        ids.sort(key=lambda i: nodes[i]["dispatch_seq"])
        queues[core] = [bundle_of[i] for i in ids]
        for i in ids:
            member_on_core[bundle_of[i], core] = i
    maximum_cost = [max(costs[i] for i in ids) for ids in bundles]

    def estimate():
        parents = [p.copy() for p in static]
        for core, sequence in queues.items():
            for u, v in zip(sequence, sequence[1:]):
                lag = costs[member_on_core[u, core]]
                parents[v][u] = max(parents[v].get(u, 0.0), lag)
        children = [[] for _ in bundles]
        degree = [len(p) for p in parents]
        for v, ps in enumerate(parents):
            for u, lag in ps.items():
                children[u].append((v, lag))
        ready = [u for u, count in enumerate(degree) if not count]
        heapq.heapify(ready)
        starts = [0.0] * len(bundles)
        order = []
        while ready:
            u = heapq.heappop(ready)
            order.append(u)
            for v, lag in children[u]:
                starts[v] = max(starts[v], starts[u] + lag)
                degree[v] -= 1
                if degree[v] == 0:
                    heapq.heappush(ready, v)
        if len(order) != len(bundles):
            return None
        ends = [starts[u] + maximum_cost[u] for u in range(len(bundles))]
        # 同时改善最长路径和队列的整体等待，允许走过多个等长关键路径的平台。
        objective = (round(max(ends), 7), round(sum(ends), 7))
        return objective, starts, ends, parents, order

    initial = estimate()
    assert initial is not None
    best = initial
    cohort_moves = []
    if defer_main_q:
        # 主Q后处理仅生产attention使用的Q，不生产Indexer query。
        # 成组移动可越过“单核交换不改变全局最长路径”的平台；必须重建整个图验环。
        for core, sequence in queues.items():
            if core < 24:
                continue
            indexer_units = [u for u in sequence if nodes[bundles[u][0]]["task"] == indexer]
            if len(indexer_units) != 1:
                raise ValueError("每条AIV必须有且只有一份原Indexer成员")
            anchor = indexer_units[0]
            moving = [u for u in sequence[:sequence.index(anchor)] if
                      schedule["call_bindings"][nodes[bundles[u][0]]["task"]]["logical_function"]
                      == "qproj_dequant_rms_nope_rope"]
            for u in moving:
                if len(bundles[u]) != 1:
                    raise ValueError("不能单独移动MIX成员")
                sequence.remove(u)
            insertion = sequence.index(anchor) + 1
            sequence[insertion:insertion] = moving
            cohort_moves.extend({"core": core, "unit": u, "after": anchor} for u in moving)
        best = estimate()
        if best is None:
            raise ValueError("整组延后主Q后处理产生依赖环，不得生成设备版本")
    start_time = time.monotonic()
    transitions = []
    evaluations, cycles, blocked = 0, 0, 0
    for round_id in range(rounds):
        chosen = None
        for core in sorted(queues):
            sequence = queues[core]
            for j, (u, v) in enumerate(zip(sequence, sequence[1:])):
                # 两个MIX不能只在一条物理核上反序；数学祖先也不允许反序。
                if (len(bundles[u]) > 1 and len(bundles[v]) > 1
                        or math_ancestors[v] >> u & 1):
                    blocked += 1
                    continue
                sequence[j], sequence[j + 1] = v, u
                trial = estimate()
                sequence[j], sequence[j + 1] = u, v
                evaluations += 1
                if trial is None:
                    cycles += 1
                elif trial[0] < best[0] and (chosen is None or trial[0] < chosen[0]):
                    chosen = trial[0], core, j, trial
                if time.monotonic() - start_time >= seconds:
                    break
            if time.monotonic() - start_time >= seconds:
                break
        if chosen is None:
            break
        _, core, j, best = chosen
        u, v = queues[core][j:j + 2]
        queues[core][j:j + 2] = [v, u]
        transitions.append({"round": round_id, "core": core, "before": [u, v],
                            "objective": list(best[0])})
        print(json.dumps(transitions[-1]), flush=True)
        if time.monotonic() - start_time >= seconds:
            break

    verified = estimate()
    assert verified is not None and verified[0] == best[0]
    # 每个原成员恰好出现一次，物理分核及MIX映射未动。
    actual = [member_on_core[u, core] for core, sequence in queues.items() for u in sequence]
    assert sorted(actual) == list(range(len(nodes)))
    report = {
        "scope": "固定分核相邻交换的CPU成本模型；未生成二进制/未上卡",
        "member_baseline_us": member_baseline,
        "common_mix_start_baseline_us": initial[0][0],
        "candidate_estimate_us": best[0][0],
        "model_reduction_us": initial[0][0] - best[0][0],
        "members": len(nodes), "units": len(bundles), "ffts_reuse_member_edges": ffts_edges,
        "evaluations": evaluations, "cycle_rejections": cycles,
        "math_or_mixed_rejections": blocked, "accepted_moves": len(transitions),
        "elapsed_seconds": time.monotonic() - start_time,
        "mapping_and_math_unchanged": True, "acyclic": True,
        "common_mix_start_required": True, "device_validated": False,
        "cohort_moves": cohort_moves,
        "transitions": transitions,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    plan = {
        "units": [[{key: nodes[i][key] for key in
                    ("core", "task", "logical_block", "original_subslot", "dispatch_seq")}
                   for i in ids] for ids in bundles],
        "unit_queues": queues,
        "unit_parents_with_lags": best[3],
        "unit_start_us": best[1],
    }
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--defer-main-q-dequant", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output, args.seconds, args.rounds,
                         args.defer_main_q_dequant), ensure_ascii=False))
