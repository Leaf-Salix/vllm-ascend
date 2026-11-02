# SPDX-License-Identifier: Apache-2.0
"""复制到独立算子包使用；保留Native相邻对顺序的物理行MGather。"""

import pypto.language as pl

from .attention_native_stream_stages import COMPRESSED_ROWS_DYN, HEAD_DIM, PLAN_CAP_DYN

PHYSICAL_ROWS_DYN = pl.dynamic("NATIVE_PHYSICAL_PLAN_ROWS")


@pl.jit.inline(auto_scope=False)
def native_gather_physical_pairs(
    cache: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    raw_indices: pl.Tensor[[PLAN_CAP_DYN, 512], pl.INT32],
    physical_indices: pl.Tensor[[PHYSICAL_ROWS_DYN, 512], pl.INT32],
    token: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    limit: pl.Scalar[pl.INDEX],
):
    # 调用方证明position<2**24，使plan逻辑上界与Native整数上界一致。
    # physical plan保证物理行<=2**29-1；下列地址/距离比较全部用INT32，不经浮点。
    physical = pl.load(physical_indices, [token, column], [1, 64])
    raw = pl.load(raw_indices, [token, column], [1, 64])
    even = pl.tile.muls(pl.tile.ci(0, [1, 32], dtype=pl.INT32), 2)
    odd = pl.tile.adds(even, 1)
    tmp32 = pl.create_tile([1, 32], dtype=pl.INT32)
    p0 = pl.tile.gather(physical, even, tmp32)
    p1 = pl.tile.gather(physical, odd, tmp32)
    r0 = pl.tile.maximums(pl.tile.gather(raw, even, tmp32), 0)
    r1 = pl.tile.maximums(pl.tile.gather(raw, odd, tmp32), 0)
    valid0 = pl.tile.minimums(pl.tile.maximums(pl.tile.adds(p0, 1), 0), 1)
    valid1 = pl.tile.minimums(pl.tile.maximums(pl.tile.adds(p1, 1), 0), 1)
    upper = pl.cast(limit - 1, pl.INT32)
    last0 = pl.tile.minimums(pl.tile.maximums(pl.tile.adds(pl.tile.muls(r0, -1), upper), 0), 1)
    last1 = pl.tile.minimums(pl.tile.maximums(pl.tile.adds(pl.tile.muls(r1, -1), upper), 0), 1)
    difference = pl.sub(p0, p1)
    descending = pl.tile.minimums(pl.tile.maximums(difference, 0), 1)
    # Native条件：(row0-row1-1)*1024 < INT32_MAX，等价于正差<=2**21。
    near = pl.tile.minimums(pl.tile.maximums(pl.tile.adds(pl.tile.muls(difference, -1), 2097153), 0), 1)
    swap = pl.mul(pl.mul(pl.mul(valid0, valid1), pl.mul(last0, last1)), pl.mul(descending, near))
    lanes = pl.tile.ci(0, [1, 64], dtype=pl.INT32)
    pair_ids = pl.tile.shrs(lanes, 1)
    parity = pl.tile.sub(lanes, pl.tile.muls(pair_ids, 2))
    tmp64 = pl.create_tile([1, 64], dtype=pl.INT32)
    swap64 = pl.tile.gather(swap, pair_ids, tmp64)
    permutation = pl.tile.add(lanes, pl.tile.sub(swap64, pl.tile.muls(pl.tile.mul(swap64, parity), 2)))
    ordered = pl.tile.gather(physical, permutation, tmp64)
    return pl.tile.mgather(cache, ordered, coalesce="row", gather_oob="zero")
