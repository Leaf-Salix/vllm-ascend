# SPDX-License-Identifier: Apache-2.0
"""Inspect completed Score ownership plans outside all performance windows."""


def inspect_score_owner_plan(entry, workspace, base):
    """Compare device-written maps with independently enumerated work ownership.

    This diagnostic may copy tensors to CPU. It is test-only, after completion;
    it is never used to choose production work, launch kernels, or time L1.
    """
    groups_total, vector_count = 24, 48
    score_tile, maximum_leaves = 1024, 32
    tokens = entry.buffers["positions"].shape[0]
    requests = entry.buffers["idx_lengths"].shape[0]
    plan = entry.buffers["group_plan"].cpu().tolist()
    assert len(plan) == 8 and workspace.numel() == base + 4 * groups_total
    groups, leaves, leaf_tiles, extra = plan[0], plan[2], plan[3], plan[4]
    words = workspace[base:].tolist()
    costs, prefix, mapping, inverse = [words[index:index + groups_total] for index in range(0, 96, 24)]
    assert sorted(mapping) == sorted(inverse) == list(range(groups_total)), "mapping is not a permutation"
    assert all(inverse[logical] == physical for physical, logical in enumerate(mapping)), "inverse mismatch"
    eligible = (
        groups_total <= groups <= vector_count and 0 < leaves <= maximum_leaves
        and leaf_tiles > 0 and 0 <= extra < leaves and 128 < tokens <= 256
        and plan[5] == 6 and plan[6] != 0
    )
    expected_cost = [0] * groups_total
    descriptor = (requests + 16) // 16 * 16
    if eligible:
        for group in range(groups):
            visible = int(workspace[descriptor + 4 * group + 3])
            for leaf in range(leaves):
                first_tile = leaf * leaf_tiles + min(leaf, extra)
                count = leaf_tiles + int(leaf < extra)
                # Count actual nonempty panels, independent of ceil(valid/tile).
                panels = sum(tile * score_tile < visible for tile in range(first_tile, first_tile + count))
                expected_cost[(group * leaves + leaf) % groups_total] += panels
    assert costs == expected_cost, "device Score costs disagree with actual visible panels"
    expected_prefix = [0] * groups_total
    expected_mapping = list(range(groups_total))
    should_permute = eligible and max(expected_cost) - min(expected_cost) >= 2
    if should_permute:
        branch_groups, qb_columns, kv_columns = 4, 128, 4
        full_end = tokens // 64 * 64
        padded_end = (tokens + 15) // 16 * 16
        row_groups = min(3, max(1, tokens // 64))
        for physical in range(groups_total):
            logical = (physical - branch_groups) % groups_total
            rows = 0
            if logical < kv_columns * row_groups:
                row_group = logical % row_groups
                rows += sum(64 for _ in range(row_group * 64, full_end, row_groups * 64))
                rows += sum(16 for _ in range(full_end + row_group * 16, padded_end, row_groups * 16))
            columns = 0
            if physical >= branch_groups:
                columns = len(range(physical % (groups_total - branch_groups), qb_columns,
                                    groups_total - branch_groups))
            expected_prefix[physical] = rows * (qb_columns + 1) + columns
        physical_order = sorted(range(groups_total), key=lambda item: (-expected_prefix[item], item))
        logical_order = sorted(range(groups_total), key=lambda item: (expected_cost[item], item))
        for physical, logical in zip(physical_order, logical_order):
            expected_mapping[physical] = logical
    assert prefix == expected_prefix, "device KV/QB prefix ranks disagree with actual tile traversal"
    assert mapping == expected_mapping, "device assignment does not follow stable cost ordering"
    return {
        "passed": True, "groups": groups, "leaves": leaves, "eligible": eligible,
        "costs": costs, "prefix_ranks": prefix, "physical_to_logical": mapping,
        "logical_to_physical": inverse, "identity": mapping == list(range(groups_total)),
        "scope": "Post-completion ownership diagnostic; no production readback or timing",
    }
