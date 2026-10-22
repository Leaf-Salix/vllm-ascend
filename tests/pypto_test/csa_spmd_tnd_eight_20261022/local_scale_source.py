# SPDX-License-Identifier: Apache-2.0
"""Prove one full scale block contiguous, otherwise execute the legacy page loop."""

import pypto.language as pl

CACHE_PAGES = pl.dynamic("CSA_LOCAL_SCALE_CACHE_PAGES")
PAGE_BYTES = pl.dynamic("CSA_LOCAL_SCALE_PAGE_BYTES")
TABLE_ROWS = pl.dynamic("CSA_LOCAL_SCALE_TABLE_ROWS")
TABLE_COLS = pl.dynamic("CSA_LOCAL_SCALE_TABLE_COLS")


@pl.jit.inline(auto_scope=False)
def contiguous_scale_block(
    table: pl.Tensor[[TABLE_ROWS, TABLE_COLS], pl.INT32],
    batch: pl.Scalar[pl.INDEX],
    score_begin: pl.Scalar[pl.INDEX],
    cache_len: pl.Scalar[pl.INDEX],
    cache_pages: pl.Scalar[pl.INDEX],
    SCORE_TILE: pl.constexpr,
):
    first_physical = pl.cast(-1, pl.INDEX)
    logical_begin = score_begin // 32
    pages = SCORE_TILE // 32
    if (
        logical_begin >= 0
        and logical_begin + pages <= pl.tensor.dim(table, 1)
        and logical_begin + pages - 1 <= (cache_len - 1) // 32
        and cache_pages <= 16777216
    ):
        first = pl.cast(pl.read(table, [batch, logical_begin]), pl.INDEX)
        if first >= 0 and first + pages <= cache_pages:
            actual = pl.load(table, [batch, logical_begin], [1, SCORE_TILE // 32])
            expected = pl.add(
                pl.cast(pl.tile.ci(0, [1, SCORE_TILE // 32], dtype=pl.INT32), pl.FP32),
                pl.cast(pl.cast(first, pl.INT32), pl.FP32),
            )
            difference = pl.abs(pl.sub(pl.cast(actual, pl.FP32), expected))
            # Preserve the established physical reduction geometry, but use one
            # valid row. FP32 is exact for every permitted physical page number.
            ones = pl.set_validshape(pl.tile.full([8, SCORE_TILE // 32], dtype=pl.FP32, value=1.0), 1, SCORE_TILE // 32)
            repeated = pl.set_validshape(pl.col_expand_mul(ones, difference), 1, SCORE_TILE // 32)
            scratch = pl.create_tile([8, SCORE_TILE // 32], dtype=pl.FP32)
            maximum = pl.row_max(repeated, scratch)
            if pl.tile.read(maximum, [0, 0]) == 0.0:
                first_physical = first
    return first_physical


@pl.jit.inline(auto_scope=False)
def read_local_scales(
    cache: pl.Tensor[[CACHE_PAGES, PAGE_BYTES], pl.INT8],
    table: pl.Tensor[[TABLE_ROWS, TABLE_COLS], pl.INT32],
    batch: pl.Scalar[pl.INDEX],
    score_begin: pl.Scalar[pl.INDEX],
    cache_len: pl.Scalar[pl.INDEX],
    SCORE_TILE: pl.constexpr,
):
    first = contiguous_scale_block(table, batch, score_begin, cache_len, pl.tensor.dim(cache, 0), SCORE_TILE)
    result = pl.tile.reinterpret_view(
        pl.fillpad(pl.tile.full([1, SCORE_TILE], dtype=pl.FP16, value=0.0), pad_value=pl.PadValue.zero), pl.INT8
    )
    if first >= 0 and pl.tensor.dim(cache, 1) >= 4160:
        packed = pl.load(cache, [first, 4096], [SCORE_TILE // 32, 64])
        halves = pl.tile.reinterpret_view(packed, pl.FP16)
        flat = pl.tile.reshape(pl.fillpad(halves, pad_value=pl.PadValue.zero), [1, SCORE_TILE])
        result = pl.tile.reinterpret_view(flat, pl.INT8)
    else:
        for lane in pl.range(SCORE_TILE // 32):
            logical = pl.min(score_begin // 32 + lane, pl.max((cache_len - 1) // 32, 0))
            if logical >= 0 and logical < pl.tensor.dim(table, 1):
                physical = pl.cast(pl.read(table, [batch, logical]), pl.INDEX)
                if physical >= 0 and physical < pl.tensor.dim(cache, 0):
                    result = pl.gather_row(result, cache, [0, lane * 64], [physical, 4096], [1, 64])
    return result
