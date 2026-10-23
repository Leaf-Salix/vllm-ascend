# SPDX-License-Identifier: Apache-2.0
"""Dependency-only readiness model in abstract Score-step units, never microseconds.

This optimistic model ignores prefix arrivals, sorting/plan costs, resource
contention and the current requirement that each AIV finish its own Score loop.
It can expose dependency barriers; it cannot predict end-to-end speedup.
"""

import json
from pathlib import Path


def downstream(root_ready, tokens, per_query_topk=False, first_query_only=False):
    query_ready = [root_ready[q // 6] for q in range(tokens)]
    topk_phase = [max((query_ready[q] for q in range(worker, tokens, 48)), default=0) for worker in range(48)]
    plan_phase = []
    for worker in range(48):
        rows = [q for begin in range(worker * 8, tokens, 48 * 8) for q in range(begin, min(begin + 8, tokens))]
        plan_phase.append(max((query_ready[q] if per_query_topk else topk_phase[q % 48] for q in rows), default=0))
    attention_ready = []
    for core in range(24):
        queries = list(range(core, tokens, 24))
        if first_query_only:
            queries = queries[:1]
        attention_ready.append(max((plan_phase[(q // 8) % 48] for q in queries), default=0))
    return dict(topk_phase=topk_phase, plan_phase=plan_phase, attention_ready=attention_ready)


def main():
    here = Path(__file__).resolve().parent
    work = json.loads((here / "score_owner_work_model.json").read_text())
    report = dict(
        device_execution=False,
        performance_eligible=False,
        source="score_owner_work_model.json",
        assumptions=[
            "Only C04/C05 uniform S6 fixtures; q//6 is an offline fixture identity, never proposed Host dispatch",
            "All Score workers start together; every 1024-candidate step costs one abstract unit",
            "Leaf service order is the actual per-core item order in the existing work model",
            "Zero sorting, plan, publication and synchronization costs; no prefix or resource contention",
            "Hypothetical early TopK work ignores its current same-core Score predecessor",
            "These are readiness constraints, not a schedulable implementation or a latency forecast",
        ],
        cases={},
    )
    for case_id, tokens in [("C04", 96), ("C05", 144)]:
        case = work["cases"][case_id]
        assert len(case["groups"]) == tokens // 6 and all(g["live"] == 6 for g in case["groups"])
        leaf_end, phase_end = {}, {}
        for core in case["original"]["cores"]:
            elapsed = 0
            for item in core["work_items"]:
                key = (item["group"], item["leaf"])
                assert key not in leaf_end
                elapsed += item["score_steps"]
                leaf_end[key] = elapsed
                phase_end[key] = core["score_steps"]
            assert elapsed == core["score_steps"]
        expected = {(g, leaf) for g in range(len(case["groups"])) for leaf in range(case["leaves"])}
        assert set(leaf_end) == expected
        early = [max(leaf_end[g, leaf] for leaf in range(case["leaves"])) for g in range(len(case["groups"]))]
        current = [max(phase_end[g, leaf] for leaf in range(case["leaves"])) for g in range(len(case["groups"]))]
        row = dict(
            whole_owner_root_ready=current,
            hypothetical_per_leaf_root_ready=early,
            current_phase_dependencies=downstream(current, tokens),
            per_leaf_only_keep_downstream_phases=downstream(early, tokens),
            per_leaf_and_per_query_topk_keep_whole_attention_query_set=downstream(early, tokens, True),
            per_leaf_and_per_query_topk_and_first_attention_query_only=downstream(early, tokens, True, True),
        )
        report["cases"][case_id] = row
        print(case_id, "root readiness", sorted(set(current)), "hypothetical leaf readiness", sorted(set(early)))
        for name, values in row.items():
            if isinstance(values, dict):
                print(" ", name, sorted(set(values["attention_ready"])))
    with (here / "score_readiness_model.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
