# SPDX-License-Identifier: Apache-2.0
"""HC分行源码模板：物理Tile仍8行，valid_shape仅处理本worker拥有的行。

由prepare_hc_row_parallel复制到私有算子包；不单独作为测试模块导入。
保留32个K512平方和、四路pairwise混合、BF16舍入、16个K256平方和顺序。
所有小列向量保留8个FP32的物理跨度，满足A3的32B对齐。
"""

import pypto.language as pl

from .hc_stages import (
    D_TILE,
    HC_DIM,
    HC_DIM_INV,
    HC_EPS,
    HC_MULT,
    HC_PAD,
    MIX_HC,
    MIX_PAD,
    MIX_ROWS_DYN,
    NORM_EPS,
    PAD_ROWS_DYN,
    PARTIAL_ROWS_DYN,
    RMS_K_TILE,
    T_DYN,
    T_TILE,
    D,
    hc_projection_piece,
)


@pl.jit.inline(auto_scope=False)
def row_hc_widen_rms_range(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.BF16],
    x_fp32: pl.Out[pl.Tensor[[T_DYN, HC_DIM], pl.FP32]],
    inv_rms: pl.Out[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[T_TILE, HC_DIM], pl.FP32]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
    HC_ROW_TILE: pl.constexpr,
):
    tokens = pl.tensor.dim(x, 0)
    reduce_tmp = pl.create_tile([T_TILE, RMS_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    rsqrt_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    for block in pl.range(vector_worker, (tokens + HC_ROW_TILE - 1) // HC_ROW_TILE, vector_workers):
        first = block * HC_ROW_TILE
        rows = pl.min(HC_ROW_TILE, tokens - first)
        sums = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
            col = kb * RMS_K_TILE
            src = pl.load(
                x, [first, col], [T_TILE, RMS_K_TILE], valid_shape=[rows, RMS_K_TILE], target_memory=pl.MemorySpace.Vec
            )
            value = pl.set_validshape(pl.cast(src, pl.FP32), rows, RMS_K_TILE)
            pl.store(value, [first, col], x_fp32)
            # 归约只沿K，不跨行；未拥有的行不参与有效结果，也不写回它们。
            squared = pl.mul(value, value)
            row_sums = pl.reshape(pl.row_sum(squared, reduce_tmp), [1, T_TILE])
            sums = pl.add(sums, row_sums)
        mean = pl.add(pl.mul(sums, HC_DIM_INV), NORM_EPS)
        inverse = pl.reshape(pl.tile.rsqrt(mean, tmp=rsqrt_tmp), [T_TILE, 1])
        pl.store(pl.set_validshape(inverse, rows, 1), [first, 0], inv_rms)
    return x_fp32, inv_rms


@pl.jit.inline(auto_scope=False)
def row_hc_gates_range(
    inv_rms: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    mixes_raw: pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Out[pl.Tensor[[PAD_ROWS_DYN, HC_PAD], pl.FP32]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_tail_store: pl.InOut[pl.Tensor[[T_TILE, HC_PAD], pl.FP32]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
    partials: pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32],
    CONSUMER_REDUCE: pl.constexpr,
    HC_ROW_TILE: pl.constexpr,
):
    tokens = pl.tensor.dim(post, 0)
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    for block in pl.range(vector_worker, (tokens + HC_ROW_TILE - 1) // HC_ROW_TILE, vector_workers):
        first = block * HC_ROW_TILE
        rows = pl.min(HC_ROW_TILE, tokens - first)
        inverse = pl.load(inv_rms, [first, 0], [T_TILE, 1], valid_shape=[rows, 1], target_memory=pl.MemorySpace.Vec)
        pre_base = pl.reshape(pl.load(hc_base, [0], [HC_PAD]), [1, HC_PAD])
        pre_raw = hc_projection_piece(partials, mixes_raw, first, rows, 0, HC_PAD, CONSUMER_REDUCE)
        pre_scaled = pl.mul(pl.row_expand_mul(pre_raw, inverse), scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_value = pl.add(pre_sig, HC_EPS)
        pl.store(pl.set_validshape(pre_value, rows, HC_PAD), [first, 0], pre_val_store)
        post_base = pl.reshape(pl.load(hc_base, [HC_MULT], [HC_PAD]), [1, HC_PAD])
        post_raw = hc_projection_piece(partials, mixes_raw, first, rows, HC_MULT, HC_PAD, CONSUMER_REDUCE)
        post_scaled = pl.mul(pl.row_expand_mul(post_raw, inverse), scale1)
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
        post_value = pl.mul(post_sig, 2.0)
        pl.store(pl.set_validshape(post_value, rows, HC_MULT), [first, 0], post)
    return pre_val_store, post


@pl.jit.inline(auto_scope=False)
def row_hc_mix_norm_range(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    pre_val_store: pl.Tensor[[PAD_ROWS_DYN, HC_PAD], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_mixed: pl.InOut[pl.Tensor[[MIX_ROWS_DYN, D], pl.BF16]],
    x_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    x_normed_tail_store: pl.InOut[pl.Tensor[[T_TILE, D], pl.BF16]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
    HC_ROW_TILE: pl.constexpr,
):
    tokens = pl.tensor.dim(x, 0)
    reduce_tmp = pl.create_tile([T_TILE, D_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    rsqrt_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    for block in pl.range(vector_worker, (tokens + HC_ROW_TILE - 1) // HC_ROW_TILE, vector_workers):
        first = block * HC_ROW_TILE
        rows = pl.min(HC_ROW_TILE, tokens - first)
        pre = pl.load(
            pre_val_store, [first, 0], [T_TILE, HC_PAD], valid_shape=[rows, HC_PAD], target_memory=pl.MemorySpace.Vec
        )
        transposed = pl.transpose(pre, axis1=0, axis2=1)
        pre0 = pl.reshape(transposed[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(transposed[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(transposed[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(transposed[3:4, 0:T_TILE], [T_TILE, 1])
        sums = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(D // D_TILE, stage=2):
            col = kb * D_TILE
            x0 = pl.load(x, [first, col], [T_TILE, D_TILE], valid_shape=[rows, D_TILE])
            x1 = pl.load(x, [first, D + col], [T_TILE, D_TILE], valid_shape=[rows, D_TILE])
            x2 = pl.load(x, [first, 2 * D + col], [T_TILE, D_TILE], valid_shape=[rows, D_TILE])
            x3 = pl.load(x, [first, 3 * D + col], [T_TILE, D_TILE], valid_shape=[rows, D_TILE])
            y01 = pl.add(pl.row_expand_mul(x0, pre0), pl.row_expand_mul(x1, pre1))
            y23 = pl.add(pl.row_expand_mul(x2, pre2), pl.row_expand_mul(x3, pre3))
            rounded = pl.set_validshape(pl.cast(pl.add(y01, y23), pl.BF16, mode="rint"), rows, D_TILE)
            pl.store(rounded, [first, col], x_mixed)
            value = pl.set_validshape(pl.cast(rounded, pl.FP32), rows, D_TILE)
            sums = pl.add(sums, pl.reshape(pl.row_sum(pl.mul(value, value), reduce_tmp), [1, T_TILE]))
        inverse = pl.tile.rsqrt(pl.add(pl.mul(sums, 1.0 / D), NORM_EPS), tmp=rsqrt_tmp)
        inverse_col = pl.reshape(inverse, [T_TILE, 1])
        for kb in pl.pipeline(D // D_TILE, stage=2):
            col = kb * D_TILE
            rounded = pl.load(x_mixed, [first, col], [T_TILE, D_TILE], valid_shape=[rows, D_TILE])
            weight = pl.reshape(pl.load(norm_w, [col], [D_TILE]), [1, D_TILE])
            scaled = pl.row_expand_mul(pl.cast(rounded, pl.FP32), inverse_col)
            value = pl.col_expand_mul(scaled, pl.cast(weight, pl.FP32))
            normed = pl.cast(value, pl.BF16, mode="rint")
            pl.store(pl.set_validshape(normed, rows, D_TILE), [first, col], x_normed)
    return x_mixed, x_normed
