# SPDX-License-Identifier: Apache-2.0
"""Resource-constrained screening of whole-leaf/query interleaving, not a speed forecast.

Each cluster is one serialized service resource in this deliberately coarse
model. Score steps cost one unit; an Attention query costs a dimensionless
parameter. This excludes complementary AIC/AIV/pipe overlap, so conclusions
apply only to moving the existing jobs between phases on that resource model.
"""

import ast
import json
import runpy
from pathlib import Path

from analyze_score_owners import count, describe


def function_ast(path, name):
    tree = ast.parse(path.read_text())
    return ast.dump(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name))


def schedule(cores, groups, leaves, query_cost):
    """Keep every owner's leaf order; insert queries after their last root round.

    A round is an owner's ordinal in its original Score item list. A query
    waits for all its root leaves, not for an invented global barrier. Every
    cluster executes its local Score item before its Attention jobs in a round.
    This is acyclic even when different clusters finish a round at different
    times. TopK/plan/publication and repeated pipeline draining cost zero here.
    """
    tokens = groups * 6
    roots = {}
    for core, row in enumerate(cores):
        for ordinal, item in enumerate(row["work_items"]):
            key = (item["group"], item["leaf"])
            assert key not in roots
            roots[key] = (core, ordinal)
    assert set(roots) == {(group, leaf) for group in range(groups) for leaf in range(leaves)}
    ready_round = [max(roots[group, leaf][1] for leaf in range(leaves)) for group in range(groups)]
    available = [0.0] * 24
    root_ends = {}
    jobs = []
    query_seen = set()
    rounds = max(len(row["work_items"]) for row in cores)
    for ordinal in range(rounds):
        for core, row in enumerate(cores):
            if ordinal < len(row["work_items"]):
                item = row["work_items"][ordinal]
                start = available[core]
                end = start + item["score_steps"]
                root_ends[item["group"], item["leaf"]] = end
                available[core] = end
                jobs.append(dict(kind="score", core=core, start=start, end=end, **item))
        for core in range(24):
            for query in range(core, tokens, 24):
                if ready_round[query // 6] == ordinal:
                    assert query not in query_seen
                    query_seen.add(query)
                    root_ready = max(root_ends[query // 6, leaf] for leaf in range(leaves))
                    start = max(available[core], root_ready)
                    end = start + query_cost
                    available[core] = end
                    jobs.append(
                        dict(kind="attention", core=core, query=query, start=start, end=end, root_ready=root_ready)
                    )
    assert query_seen == set(range(tokens))
    # Validate the actual schedule, including both resource exclusion and every
    # root dependency. Zero-duration jobs are permitted for the diagnostic limit.
    for core in range(24):
        intervals = sorted((job["start"], job["end"]) for job in jobs if job["core"] == core)
        assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    assert all(job["start"] >= job["root_ready"] for job in jobs if job["kind"] == "attention")
    score_work = sum(row["score_steps"] for row in cores)
    service_work = score_work + tokens * query_cost
    lower_bound = service_work / 24
    assert max(available) >= lower_bound
    baseline = max(row["score_steps"] for row in cores) + (tokens // 24) * query_cost
    assert tokens % 24 == 0 and baseline == lower_bound
    return dict(
        query_cost_in_score_step_units=query_cost,
        query_ready_after_owner_round=[value + 1 for value in ready_round],
        serialized_resource_capacity_bound=lower_bound,
        phase_order_makespan=baseline,
        interleaved_makespan=max(available),
        modeled_change_pct=100 * (max(available) / baseline - 1),
        total_score_step_work=score_work,
        total_query_work=tokens * query_cost,
        idle_resource_units=max(available) * 24 - service_work,
        jobs=jobs,
    )


def main():
    here = Path(__file__).resolve().parent
    source = here / "runs/physical_capacity_production_v1"
    baseline_source = here / "runs/baseline"
    unchanged = {}
    for name in ("indexer_long_leaf_plan", "indexer_work_item", "indexer_leaf_range"):
        unchanged[name] = function_ast(source / "indexer_group_stages.py", name) == function_ast(
            baseline_source / "indexer_group_stages.py", name
        )
    assert all(unchanged.values())
    select_policy = runpy.run_path(str(source / "policy.py"))["select_policy"]
    cases = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
    historical = json.loads((here / "score_owner_work_model.json").read_text())["cases"]
    report = dict(
        device_execution=False,
        performance_eligible=False,
        source="physical_capacity_production_v1.json",
        source_plan_functions_equal_frozen_baseline=unchanged,
        scope="C04/C05 uniform S6 fixtures; query//6 is offline identity, not proposed Host dispatch",
        resource_model="One serialized whole-leaf/query service resource per cluster; 24 clusters",
        limitations=[
            "Dimensionless sensitivity analysis; no phase duration is fitted as pure service cost",
            "Does not model simultaneous use of separate hardware pipes or AIC/AIV roles",
            "TopK, plan, publication, acquisition and context-switch costs are optimistically zero",
            "Current Attention pipelines across queries; splitting calls adds startup/drain costs excluded here",
            "No conclusion that every asynchronous ready scheduler or hardware overlap scheme must lose",
            "A better real scheme must exploit resource complementarity, reduce work, "
            "or fix measured service imbalance",
        ],
        cases={},
    )
    for case in cases:
        if case["id"] not in ("C04", "C05"):
            continue
        policy = select_policy(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        ).constexprs()
        assert policy["QUERY_GROUP"] == 6 and policy["BALANCED"] and not policy["SCORE_GROUP_ORDER"]
        groups, leaves, leaf_tiles, extra = describe(case, policy)
        assert all(group["live"] == 6 for group in groups)
        work = count(groups, leaves, leaf_tiles, extra, policy, list(range(len(groups))))
        assert work == historical[case["id"]]["original"]
        samples = [schedule(work["cores"], len(groups), leaves, cost) for cost in (0, 0.25, 0.5, 1, 2, 4, 8)]
        report["cases"][case["id"]] = dict(
            groups=len(groups),
            leaves=leaves,
            samples=samples,
            decision="Do not implement mandatory whole-leaf-round Attention insertion with the existing ownership",
        )
        print(
            case["id"],
            [(row["query_cost_in_score_step_units"], round(row["modeled_change_pct"], 3)) for row in samples],
        )
    with (here / "coarse_interleaving_resource_model.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
