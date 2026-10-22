# SPDX-License-Identifier: Apache-2.0
"""Model the frozen Score owner's padded QK/WS work, not measured utilization."""

import json
import math
import runpy
from pathlib import Path


def describe(case, policy):
    group_size = policy["QUERY_GROUP"]
    groups = []
    for request, (query, history) in enumerate(zip(case["query_lengths"], case["history_lengths"])):
        for begin in range(0, query, group_size):
            live = min(group_size, query - begin)
            groups.append(dict(request=request, first=begin, live=live, visible=(history + begin + live) // 4))
    maximum = max(case["seq_lens"]) // 4
    leaves, leaf_tiles, extra = max(math.ceil(maximum / 8192), 1), 8, 0
    balanced = policy["BALANCED"]
    if balanced:
        if group_size == 2:
            leaves *= 2
            leaf_tiles = 4
        else:
            multiple = 3 if len(groups) in (8, 16) else 6 if len(groups) == 4 else 1
            proposed = math.ceil(leaves / multiple) * multiple
            if multiple > 1 and proposed <= 32:
                leaves = proposed
                leaf_tiles = max(math.ceil(maximum / 1024) // leaves, 1)
                extra = math.ceil(maximum / 1024) % leaves
    return groups, leaves, leaf_tiles, extra


def count(groups, leaves, leaf_tiles, extra, policy, order):
    cores = [dict(score_steps=0, padded_qk_rows_times_steps=0, query_steps=[0, 0], work_items=[]) for _ in range(24)]
    for item in range(len(groups) * leaves):
        slot, leaf = (item % len(groups), item // len(groups)) if len(groups) < 24 else divmod(item, leaves)
        group = order[slot]
        info = groups[group]
        visible = info["visible"]
        if policy["BALANCED"] and policy["QUERY_GROUP"] == 2:
            original_begin = (leaf // 2) * 8192
            original_valid = max(min(visible - original_begin, 8192), 0)
            capacity = min(math.ceil(original_valid / 1024) * 512, 4096)
            begin = original_begin + (leaf % 2) * capacity
        else:
            begin = (leaf * leaf_tiles + min(leaf, extra)) * 1024
            capacity = (leaf_tiles + int(leaf < extra)) * 1024
        valid = max(min(visible - begin, capacity), 0)
        steps = math.ceil(valid / policy["SCORE_TILE"])
        padded_rows = policy["QUERY_GROUP"] * 64
        if policy["SPLIT_SCORE_ROWS"] and info["live"] <= 3:
            padded_rows = 192
        row = cores[item % 24]
        row["score_steps"] += steps
        row["padded_qk_rows_times_steps"] += padded_rows * steps
        for lane in range(2):
            live = max(min(info["live"] - lane * (policy["QUERY_GROUP"] // 2), policy["QUERY_GROUP"] // 2), 0)
            row["query_steps"][lane] += steps * live
        row["work_items"].append(dict(group=group, leaf=leaf, score_steps=steps, padded_rows=padded_rows))
    costs = [c["padded_qk_rows_times_steps"] for c in cores]
    return dict(cores=cores, max=max(costs), mean=sum(costs) / 24, max_over_mean=max(costs) / (sum(costs) / 24))


def main():
    here = Path(__file__).resolve().parent
    select_policy = runpy.run_path(str(here / "runs/baseline/policy.py"))["select_policy"]
    cases = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
    report = dict(
        device_execution=False,
        performance_eligible=False,
        source="baseline.json",
        limitations=[
            "Padded QK row-work model; excludes memory, Scalar, FIFO waits, scheduling and other stages",
            "Sorted groups are a hypothetical permutation only; no operator code has been changed",
            "A changed owner requires matching inverse mapping in TopK root dependency waits",
            "Physical leaf boundaries and their tie-sensitive merge tree must remain unchanged",
        ],
        cases={},
    )
    for case in cases:
        policy = select_policy(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        ).constexprs()
        if not (policy["BALANCED"] and policy["SPLIT_SCORE_ROWS"]):
            report["cases"][case["id"]] = dict(scope="Excluded: this analysis models the long-G6 split path only")
            continue
        groups, leaves, leaf_tiles, extra = describe(case, policy)
        original = count(groups, leaves, leaf_tiles, extra, policy, list(range(len(groups))))
        # Heavy groups first is a bounded candidate, not an optimal scheduling claim.
        order = sorted(range(len(groups)), key=lambda g: (-(1 + int(groups[g]["live"] > 3)), -groups[g]["visible"], g))
        proposed = count(groups, leaves, leaf_tiles, extra, policy, order)
        assert original["mean"] == proposed["mean"]
        report["cases"][case["id"]] = dict(
            groups=groups,
            leaves=leaves,
            leaf_tiles=leaf_tiles,
            extra_leaves=extra,
            original=original,
            hypothetical_heavy_first=proposed,
            hypothetical_order=order,
        )
        print(
            case["id"],
            "max/mean",
            round(original["max_over_mean"], 4),
            "hypothetical",
            round(proposed["max_over_mean"], 4),
        )
    (here / "score_owner_work_model.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
