# SPDX-License-Identifier: Apache-2.0
"""Device group permutation in unused descriptor capacity; leaves remain logical.

Only the existing 16-group balanced geometry is considered. With an even number
of waves, group g and g+8 share a physical worker. If a heavy/heavy pair exists
and at most half the groups need both QK halves, placing heavy groups first can
reduce that row-work tail. Unequal histories and other geometries keep identity.
This is a performance hypothesis; notifications and inverse owners must agree.
"""

import pypto.language as pl

ORDER_WORDS = pl.dynamic("CSA_SCORE_ORDER_WORDS")


@pl.jit.inline(auto_scope=False)
def order_storage_base(
    groups: pl.Scalar[pl.INDEX],
    requests: pl.Scalar[pl.INDEX],
    tokens: pl.Scalar[pl.INDEX],
):
    base = pl.cast(-1, pl.INDEX)
    if 3 * groups == 2 * 24 and tokens >= 32 and tokens < 6 * requests and 1 + 6 * groups <= 4 * tokens:
        # Original descriptors use 4 words per actual group in a 4*T region.
        # Never touch the scale-plan workspace that starts after that region.
        base = ((requests + 1 + 15) // 16) * 16 + 4 * groups
    return base


@pl.jit.inline(auto_scope=False)
def plan_score_group_order(
    offsets: pl.InOut[pl.Tensor[[ORDER_WORDS], pl.INT32]],
    groups: pl.Scalar[pl.INDEX],
    requests: pl.Scalar[pl.INDEX],
    tokens: pl.Scalar[pl.INDEX],
    leaves: pl.Scalar[pl.INDEX],
):
    base = order_storage_base(groups, requests, tokens)
    if base >= 0:
        pl.write(offsets, [base], pl.cast(0, pl.INT32))
        descriptors = ((requests + 1 + 15) // 16) * 16
        heavy_count = pl.cast(0, pl.INDEX)
        heavy_pairs = pl.cast(0, pl.INDEX)
        minimum_visible = pl.cast(2147483647, pl.INDEX)
        maximum_visible = pl.cast(0, pl.INDEX)
        for pair in pl.range(groups // 2):
            first_row = descriptors + 4 * pair
            second_row = descriptors + 4 * (pair + groups // 2)
            first_live = pl.cast(pl.read(offsets, [first_row + 2]), pl.INDEX)
            second_live = pl.cast(pl.read(offsets, [second_row + 2]), pl.INDEX)
            first_visible = pl.cast(pl.read(offsets, [first_row + 3]), pl.INDEX)
            second_visible = pl.cast(pl.read(offsets, [second_row + 3]), pl.INDEX)
            heavy_count = heavy_count + pl.cast(first_live > 3, pl.INDEX) + pl.cast(second_live > 3, pl.INDEX)
            heavy_pairs = heavy_pairs + pl.cast(first_live > 3 and second_live > 3, pl.INDEX)
            minimum_visible = pl.min(minimum_visible, pl.min(first_visible, second_visible))
            maximum_visible = pl.max(maximum_visible, pl.max(first_visible, second_visible))
        if (
            leaves % 3 == 0
            and heavy_pairs > 0
            and heavy_count <= groups // 2
            and maximum_visible > 0
            and maximum_visible - minimum_visible <= 1
        ):
            heavy_slot = pl.cast(0, pl.INDEX)
            light_slot = heavy_count
            for group in pl.range(groups):
                live = pl.cast(pl.read(offsets, [descriptors + 4 * group + 2]), pl.INDEX)
                slot = light_slot
                if live > 3:
                    slot = heavy_slot
                    heavy_slot = heavy_slot + 1
                else:
                    light_slot = light_slot + 1
                pl.write(offsets, [base + 1 + slot], pl.cast(group, pl.INT32))
                pl.write(offsets, [base + 1 + groups + group], pl.cast(slot, pl.INT32))
            pl.write(offsets, [base], pl.cast(1, pl.INT32))
    return offsets


@pl.jit.inline(auto_scope=False)
def active_score_order_base(
    offsets: pl.Tensor[[ORDER_WORDS], pl.INT32],
    groups: pl.Scalar[pl.INDEX],
    requests: pl.Scalar[pl.INDEX],
    tokens: pl.Scalar[pl.INDEX],
):
    base = order_storage_base(groups, requests, tokens)
    active = pl.cast(-1, pl.INDEX)
    if base >= 0:
        if pl.read(offsets, [base]) == 1:
            active = base
    return active
