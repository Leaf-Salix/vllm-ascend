# SPDX-License-Identifier: Apache-2.0
"""Batched scale addressing over a zero-copy 64-byte-row view of Native cache.

The caller keeps the original scalar path for non-64-byte page strides or
indices that cannot be represented in INT32. This helper preserves the old
logical-page clamp and uses explicit invalid-page masks before multiplying.
"""

import pypto.language as pl

SCALE_ROWS = pl.dynamic("CSA_EIGHT_SCALE_ROWS")
TABLE_ROWS = pl.dynamic("CSA_EIGHT_SCALE_TABLE_ROWS")
TABLE_COLS = pl.dynamic("CSA_EIGHT_SCALE_TABLE_COLS")


@pl.jit.inline(auto_scope=False)
def gather_scale_pages(
    scale_rows: pl.Tensor[[SCALE_ROWS, 64], pl.INT8],
    table: pl.Tensor[[TABLE_ROWS, TABLE_COLS], pl.INT32],
    batch: pl.Scalar[pl.INDEX],
    score_begin: pl.Scalar[pl.INDEX],
    cache_len: pl.Scalar[pl.INDEX],
    cache_pages: pl.Scalar[pl.INDEX],
    page_bytes: pl.Scalar[pl.INDEX],
    SCORE_TILE: pl.constexpr,
):
    """Load the same 64 scale bytes per selected page, including repeated tail pages."""
    columns = pl.tensor.dim(table, 1)
    lane = pl.tile.ci(0, [1, SCORE_TILE // 32], dtype=pl.INT32)
    logical = pl.add(lane, pl.cast(score_begin // 32, pl.INT32))
    logical = pl.minimum(logical, pl.cast(pl.max((cache_len - 1) // 32, 0), pl.INT32))
    logical_valid = pl.and_(pl.cmp(logical, 0, cmp_type=5), pl.cmp(logical, pl.cast(columns, pl.INT32), cmp_type=2))
    linear = pl.add(logical, pl.cast(batch * columns, pl.INT32))
    # A2/A3 TSELS needs one physical source row in the same element dtype.
    tmp = pl.tile.create([1, SCORE_TILE // 32], dtype=pl.INT32)
    safe_linear = pl.tile.sels(logical_valid, linear, tmp, -1)
    physical = pl.tile.mgather(table, safe_linear, coalesce="elem", gather_oob="zero")
    physical_valid = pl.and_(
        logical_valid,
        pl.and_(pl.cmp(physical, 0, cmp_type=5), pl.cmp(physical, pl.cast(cache_pages, pl.INT32), cmp_type=2)),
    )
    safe_physical = pl.tile.sels(physical_valid, physical, tmp, 0)
    scale_row = pl.add(pl.mul(safe_physical, pl.cast(page_bytes // 64, pl.INT32)), 4096 // 64)
    safe_scale_row = pl.tile.sels(physical_valid, scale_row, tmp, -1)
    scales = pl.tile.mgather(scale_rows, safe_scale_row, coalesce="row", gather_oob="zero")
    return pl.tile.reshape(pl.tile.reinterpret_view(scales, pl.FP16), [1, SCORE_TILE])
