# SPDX-License-Identifier: Apache-2.0
"""固定物理分核、允许整体换序的混合整数排程模型。

不重跑相邻交换或从头贪心分配。数学依赖、逻辑分片及MIX核组固定；
各MIX成员共用开始变量，同核任意两个区间必须互斥。只使用已安装的
SciPy/HiGHS，不修改共享环境。模型结果不代表设备性能或同步正确性。
"""

import argparse
import collections
import copy
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from member_schedule import _costs, _estimate, _problem, _resource_parents, _waits, ancestor_sets


def problem(root):
    root = Path(root)
    schedule = json.loads((root / "b16_cached_row_deps_v1/specialization/bindings/h131072_b16_prepared.json").read_text())
    row_audit = json.loads((root / "b16_cached_row_deps_v1/specialization/control_variants/h131072_b16/row_dependencies/audit.json").read_text())
    nodes, math_parents, bundles, bundle_of, _, _, mixed, names = _problem(schedule, row_audit)
    durations = _costs(schedule, nodes, root / "b16_dynamic_l1_cost.json")
    indexer = next(t for t, n in names.items() if n == "indexer_score_topk_native_pair")
    qk = next(t for t, n in names.items() if n == "qk_pv")
    groups = collections.defaultdict(list)
    for i, row in enumerate(nodes):
        if row["task"] == indexer:
            core = row["core"]
            groups[core if core < 24 else (core - 24) // 2].append(i)
    fixed = [dict() for _ in bundles]
    for i, original in enumerate(math_parents):
        parents = set(original)
        if nodes[i]["task"] == qk:
            core = nodes[i]["core"]
            parents.update(groups[core if core < 24 else (core - 24) // 2])
        for p in parents:
            u, v = bundle_of[p], bundle_of[i]
            if u == v:
                raise ValueError("MIX内部不能存在外层依赖")
            fixed[v][u] = max(fixed[v].get(u, 0), durations[p])
    queues = collections.defaultdict(list)
    member_on_core = {}
    for i, row in enumerate(nodes):
        queues[row["core"]].append(i)
        member_on_core[bundle_of[i], row["core"]] = i
    for core, ids in queues.items():
        ids.sort(key=lambda i: nodes[i]["dispatch_seq"])
        queues[core] = [bundle_of[i] for i in ids]
    maximum = [max(durations[i] for i in ids) for ids in bundles]
    baseline_parents = [p.copy() for p in fixed]
    for core, ids in queues.items():
        for u, v in zip(ids, ids[1:]):
            lag = durations[member_on_core[u, core]]
            baseline_parents[v][u] = max(baseline_parents[v].get(u, 0), lag)
    order, _ = ancestor_sets([set(p) for p in baseline_parents])
    baseline_starts = [0.0] * len(bundles)
    for v in order:
        baseline_starts[v] = max((baseline_starts[u] + lag for u, lag in baseline_parents[v].items()), default=0)
    horizon = max(baseline_starts[u] + maximum[u] for u in range(len(bundles)))
    return (schedule, nodes, math_parents, bundles, bundle_of, names, durations,
            fixed, queues, member_on_core, maximum, baseline_starts, horizon)


def run(root, output, seconds=120, max_order_changes=None, reduce_precedence=False,
        feasibility_target_us=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (schedule, nodes, math_parents, bundles, bundle_of, names, durations, fixed,
     queues, member_on_core, maximum, baseline_starts, horizon) = problem(root)
    reduction = None
    closure = None
    if reduce_precedence:
        from weighted_precedence import reduce

        fixed, closure, reduction = reduce(fixed)
    count = len(bundles)
    order, ancestors = ancestor_sets([set(p) for p in fixed])
    children = collections.defaultdict(dict)
    earliest = [0.0] * count
    for v in order:
        earliest[v] = max((earliest[u] + lag for u, lag in fixed[v].items()), default=0)
        for u, lag in fixed[v].items():
            children[u][v] = lag
    remaining = maximum.copy()
    for u in reversed(order):
        remaining[u] = max([maximum[u], *[lag + remaining[v] for v, lag in children[u].items()]])
    latest = [horizon - value for value in remaining]
    lower, upper = earliest + [0.0], latest + [horizon]
    integrality = [0] * (count + 1)
    baseline_values = baseline_starts + [horizon]
    coefficients, rows, cols, lower_constraints = [], [], [], []

    def constraint(terms, bound):
        row = len(lower_constraints)
        for column, value in terms.items():
            if value:
                rows.append(row)
                cols.append(column)
                coefficients.append(value)
        lower_constraints.append(bound)

    for v, parents in enumerate(fixed):
        for u, lag in parents.items():
            constraint({v: 1.0, u: -1.0}, lag)
    for u, elapsed in enumerate(maximum):
        constraint({count: 1.0, u: -1.0}, elapsed)
    # 同一个MIX对可能在三条物理核上相遇。合并为同一顺序变量，
    # 但两个方向分别保留相应共享资源上的最大占用长度。
    pairs = {}
    for core, ids in queues.items():
        for p, a in enumerate(ids):
            for b in ids[p + 1:]:
                u, v = sorted((a, b))
                du, dv = (durations[member_on_core[x, core]] for x in (u, v))
                old = pairs.get((u, v), (0.0, 0.0))
                pairs[u, v] = max(old[0], du), max(old[1], dv)
    decisions = []
    fixed_orders = 0
    implied_resource_orders = 0
    for (u, v), (du, dv) in sorted(pairs.items()):
        if ancestors[v] >> u & 1:
            if closure is not None and closure[v, u] >= du - 1e-9:
                implied_resource_orders += 1
            else:
                constraint({v: 1.0, u: -1.0}, du)
            fixed_orders += 1
        elif ancestors[u] >> v & 1:
            if closure is not None and closure[u, v] >= dv - 1e-9:
                implied_resource_orders += 1
            else:
                constraint({u: 1.0, v: -1.0}, dv)
            fixed_orders += 1
        else:
            variable = len(lower)
            lower.append(0.0)
            upper.append(1.0)
            integrality.append(1)
            uv = max(0.0, latest[u] + du - earliest[v])
            vu = max(0.0, latest[v] + dv - earliest[u])
            # z=1表示u在v前，z=0表示v在u前。M由上下界逐对计算。
            constraint({v: 1.0, u: -1.0, variable: -uv}, du - uv)
            constraint({u: 1.0, v: -1.0, variable: vu}, dv)
            before = baseline_starts[v] >= baseline_starts[u] + du - 1e-7
            baseline_values.append(float(before))
            decisions.append({"u": u, "v": v, "variable": variable, "du": du, "dv": dv,
                              "big_m_uv": uv, "big_m_vu": vu})
    if max_order_changes is not None:
        if max_order_changes < 1:
            raise ValueError("邻域必须允许实际换序")
        signs = {d["variable"]: 1.0 if baseline_values[d["variable"]] else -1.0
                 for d in decisions}
        ones = sum(baseline_values[d["variable"]] for d in decisions)
        constraint(signs, ones - max_order_changes)
    matrix = coo_matrix((coefficients, (rows, cols)),
                        shape=(len(lower_constraints), len(lower))).tocsc()
    matrix.indices = matrix.indices.astype(np.int32)
    matrix.indptr = matrix.indptr.astype(np.int32)
    baseline_values = np.asarray(baseline_values)
    bounds_lower, bounds_upper = np.asarray(lower), np.asarray(upper)
    rhs = np.asarray(lower_constraints)
    violation = max(float(np.max(rhs - matrix @ baseline_values)),
                    float(np.max(bounds_lower - baseline_values)),
                    float(np.max(baseline_values - bounds_upper)), 0.0)
    if violation > 1e-7:
        raise ValueError(f"原安排不是模型的合法见证：{violation}")
    before = {
        "units": count, "members": len(nodes), "variables": len(lower),
        "order_binaries": len(decisions), "constraints": len(lower_constraints),
        "fixed_resource_order_pairs": fixed_orders, "baseline_common_mix_start_us": horizon,
        "implied_resource_orders_omitted": implied_resource_orders,
        "weighted_precedence_reduction": reduction, "max_order_changes": max_order_changes,
        "feasibility_target_us": feasibility_target_us,
        "baseline_witness_max_violation": violation,
        "scipy_version": scipy.__version__, "shared_environment_modified": False,
        "scope": "固定分核的CPU MILP模型；不代表硬件下界或设备性能",
    }
    (output / "model.json").write_text(json.dumps(before, ensure_ascii=False, indent=2) + "\n")
    objective = np.zeros(len(lower))
    if feasibility_target_us is None:
        objective[count] = 1.0
    else:
        if not 0 < feasibility_target_us < horizon:
            raise ValueError("可行性目标必须比基线短")
        # 先验证过未加此条件的基线见证，再增加候选门槛。零目标函数
        # 表示寻找任意满足门槛的解；不能将返回的fun=0当成运行时间。
        bounds_upper[count] = feasibility_target_us
    started = time.monotonic()
    result = milp(objective, integrality=np.asarray(integrality),
                  bounds=Bounds(bounds_lower, bounds_upper),
                  constraints=LinearConstraint(matrix, rhs, np.inf),
                  options={"time_limit": float(seconds), "presolve": True, "mip_rel_gap": 0.001,
                           "threads": 8, "disp": True})
    report = dict(before, solver_status=int(result.status), solver_message=result.message,
                  solver_wall_seconds=time.monotonic() - started,
                  solver_success=bool(result.success), device_validated=False,
                  incumbent_found=result.x is not None,
                  mip_gap=getattr(result, "mip_gap", None),
                  mip_dual_bound=getattr(result, "mip_dual_bound", None),
                  mip_node_count=getattr(result, "mip_node_count", None))
    if result.x is None:
        (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return report
    values = result.x
    maximum_violation = max(float(np.max(rhs - matrix @ values)),
                            float(np.max(bounds_lower - values)),
                            float(np.max(values - bounds_upper)), 0.0)
    int_error = max((abs(values[d["variable"]] - round(values[d["variable"]])) for d in decisions), default=0)
    if maximum_violation > 1e-5 or int_error > 1e-5:
        raise ValueError("求解器返回值未通过独立约束/整数检查")
    candidate = copy.deepcopy(nodes)
    candidate_queues = {}
    for core, ids in queues.items():
        ordered = sorted(ids, key=lambda u: (values[u], values[u] + maximum[u], u))
        for u, v in zip(ordered, ordered[1:]):
            if values[v] < values[u] + durations[member_on_core[u, core]] - 1e-5:
                raise ValueError("实际同核成员区间重叠")
        members = [member_on_core[u, core] for u in ordered]
        candidate_queues[str(core)] = [candidate[i] for i in members]
        for sequence, i in enumerate(members, 1):
            start = float(values[bundle_of[i]])
            candidate[i].update(dispatch_seq=sequence, estimated_start_us=start,
                                estimated_end_us=start + durations[i],
                                start=start * 50, reference_end=(start + durations[i]) * 50)
    required = _resource_parents(candidate, math_parents, names)
    # 将共同MIX开始约束落实为真实成员的前驱集合，不能只存在于成本模型中。
    mixed_ready_edges = 0
    for ids in bundles:
        if len(ids) == 3:
            parents = set().union(*(required[i] for i in ids))
            if parents.intersection(ids):
                raise ValueError("核组就绪前驱包含本组成员")
            for i in ids:
                mixed_ready_edges += len(parents - required[i])
                required[i].update(parents)
    estimate = _estimate(required, durations)
    candidate_span = float(values[count])
    if estimate > candidate_span + 1e-5:
        raise ValueError("实现核组就绪边后成本超出求解器计划")
    wait_plan, translated = _waits(candidate, required)
    report.update(candidate_model_us=candidate_span, earliest_reconstructed_model_us=estimate,
                  improvement_us=horizon - estimate, maximum_constraint_violation=maximum_violation,
                  integer_error=int_error, mixed_ready_member_edges_added=mixed_ready_edges,
                  wait_plan=wait_plan, mapping_and_math_unchanged=True,
                  global_barriers_added=0, original_core_order_preserved=False,
                  common_mix_readiness_in_required_parents=True)
    schedule["per_core_sequences"] = candidate_queues
    schedule["wait_plan"] = wait_plan
    schedule["source_kind"] = "shape_fixed_core_milp_schedule"
    schedule["fixed_core_milp"] = report
    schedule["balanced_schedule"].update(original_placement_superseded=True, fixed_core_milp=report)
    schedule["row_dependencies"].update(original_core_order_and_mapping_preserved=False,
                                        physical_mapping_preserved=True,
                                        placement_and_wait_counts_superseded_by="fixed_core_milp")
    (output / "schedule.json").write_text(json.dumps(schedule, ensure_ascii=False, indent=2) + "\n")
    graph = {
        "nodes": [{k: row[k] for k in ("task", "logical_block", "original_subslot", "core", "dispatch_seq")}
                  for row in candidate],
        "math_parents": [sorted(p) for p in math_parents],
        "required_parents": [sorted(p) for p in required],
        "translated_parents": [sorted(p) for p in translated],
        "unit_starts_us": [float(v) for v in values[:count]],
    }
    (output / "graph.json").write_text(json.dumps(graph, ensure_ascii=False) + "\n")
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--max-order-changes", type=int)
    parser.add_argument("--reduce-precedence", action="store_true")
    parser.add_argument("--feasibility-target-us", type=float)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output, args.seconds,
                         args.max_order_changes, args.reduce_precedence,
                         args.feasibility_target_us), ensure_ascii=False))
