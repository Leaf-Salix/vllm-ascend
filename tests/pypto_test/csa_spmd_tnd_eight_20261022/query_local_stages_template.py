# SPDX-License-Identifier: Apache-2.0
"""Experimental main-query preparation by its Attention cluster.

This file is copied into a frozen operator package, not imported as a test.
One AIV owns 32 heads of one query. The arithmetic still reduces 512 columns
and preserves activation scale -> weight scale -> RMS -> RoPE -> BF16.
"""

import pypto.language as pl

from .query_projection_stages import (
    EPS,
    HEAD_DIM,
    NOPE_DIM,
    PAD_ROWS_DYN,
    Q_ROPE_T_TILE,
    QPROJ_ROWS_DYN,
    ROPE_DIM,
    ROPE_DIM_SCALE,
    T_DYN,
    H,
)

QUERY_HEAD_TILE = 8
QUERY_ROWS_DYN = pl.dynamic("CSA_QUERY_LOCAL_HEAD_ROWS")


@pl.jit.inline(auto_scope=False)
def local_query_prepare_stage(
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    query_heads: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, HEAD_DIM], pl.BF16]],
    qr_scale: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    q_proj: pl.Tensor[[QPROJ_ROWS_DYN, H * HEAD_DIM], pl.INT32],
    query: pl.Scalar[pl.INDEX],
    lane: pl.Scalar[pl.INDEX],
):
    tokens = pl.tensor.dim(query_heads, 0) // H
    activation_scale = pl.read(qr_scale, [query, 0])
    cos_row = pl.load(rope_cos, [query, 0], [1, ROPE_DIM], target_memory=pl.MemorySpace.Vec)
    sin_row = pl.load(rope_sin_signed, [query, 0], [1, ROPE_DIM], target_memory=pl.MemorySpace.Vec)
    swap_row = pl.load(rope_swap, [query, 0], [1, ROPE_DIM], target_memory=pl.MemorySpace.Vec)
    swaps = pl.col_expand(pl.tile.create([QUERY_HEAD_TILE, ROPE_DIM], dtype=pl.INT32), swap_row)
    head_offsets = pl.reshape(
        pl.mul(pl.cast(pl.tile.arange(0, [1, QUERY_HEAD_TILE], dtype=pl.INT32), pl.FP32), ROPE_DIM_SCALE),
        [QUERY_HEAD_TILE, 1],
    )
    absolute_swap = pl.reshape(
        pl.cast(pl.row_expand_add(pl.cast(swaps, pl.FP32), head_offsets), pl.INT32),
        [1, QUERY_HEAD_TILE * ROPE_DIM],
    )
    for chunk in pl.range(H // 2 // QUERY_HEAD_TILE):
        column = (lane * (H // 2) + chunk * QUERY_HEAD_TILE) * HEAD_DIM
        acc = pl.reshape(
            pl.load(q_proj, [query, column], [1, QUERY_HEAD_TILE * HEAD_DIM], target_memory=pl.MemorySpace.Vec),
            [QUERY_HEAD_TILE, HEAD_DIM],
        )
        weight_scale = pl.reshape(
            pl.load(wq_b_scale, [column], [QUERY_HEAD_TILE * HEAD_DIM], target_memory=pl.MemorySpace.Vec),
            [QUERY_HEAD_TILE, HEAD_DIM],
        )
        activation_scaled = pl.mul(pl.cast(acc, pl.FP32, mode="none"), activation_scale)
        dequantized = pl.mul(activation_scaled, weight_scale)
        squares = pl.mul(dequantized, dequantized)
        row_sum = pl.row_sum(squares, tmp_tile=pl.tile.create([QUERY_HEAD_TILE, HEAD_DIM], dtype=pl.FP32))
        variance = pl.add(pl.mul(pl.reshape(row_sum, [1, QUERY_HEAD_TILE]), 1.0 / HEAD_DIM), EPS)
        # Preserve the original choice by the query's original eight-token tile.
        if query // Q_ROPE_T_TILE * Q_ROPE_T_TILE + Q_ROPE_T_TILE <= tokens:
            inverse = pl.tile.rsqrt(variance, pl.tile.create([1, QUERY_HEAD_TILE], dtype=pl.FP32))
        else:
            inverse = pl.recip(pl.sqrt(variance))
        inverse_col = pl.reshape(inverse, [QUERY_HEAD_TILE, 1])
        nope = pl.cast(pl.row_expand_mul(dequantized[:, 0:NOPE_DIM], inverse_col), pl.BF16, mode="rint")
        rope = pl.row_expand_mul(dequantized[:, NOPE_DIM:HEAD_DIM], inverse_col)
        swapped = pl.reshape(
            pl.tile.gather(
                pl.reshape(rope, [1, QUERY_HEAD_TILE * ROPE_DIM]),
                absolute_swap,
                pl.tile.create([1, QUERY_HEAD_TILE * ROPE_DIM], dtype=pl.INT32),
            ),
            [QUERY_HEAD_TILE, ROPE_DIM],
        )
        rotated = pl.add(pl.col_expand_mul(rope, cos_row), pl.col_expand_mul(swapped, sin_row))
        rope_bf16 = pl.cast(rotated, pl.BF16, mode="rint")
        output_row = query * H + lane * (H // 2) + chunk * QUERY_HEAD_TILE
        pl.store(nope, [output_row, 0], query_heads)
        pl.store(rope_bf16, [output_row, NOPE_DIM], query_heads)
    return query_heads
