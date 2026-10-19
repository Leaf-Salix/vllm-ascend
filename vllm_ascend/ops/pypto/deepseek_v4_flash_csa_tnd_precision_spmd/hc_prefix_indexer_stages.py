# SPDX-License-Identifier: Apache-2.0
"""Native-order AIV FP32 arena and Indexer projection for the SPMD prefix."""

import pypto.language as pl

from .hc_prefix_stages import (
    COMB_T_TILE,
    D_SPMD,
    D_TILE,
    HC_DIM,
    HC_DIM_INV,
    HC_EPS,
    HC_MULT,
    HC_PAD,
    HC_SINKHORN_ITER,
    LINEAR_OK,
    LINEAR_T_TILE,
    MIX_HC,
    MIX_PAD,
    NORM_EPS,
    P16,
    PARTIAL_ROWS,
    T_TILE,
    D,
)
from .norm_wkv_stages import FP32_INFINITY_BITS, FRACTION_UNIT, SQRT_COMPARE_SHIFT, SQRT_CORRECTION_STEPS, T

# Rows [0,P16): raw0:32/pre32:40/inverse40:48/rstd48:56.
# Each inverse octet is contiguous at row8b. Last8rows hold post/comb tails.
VEC_ROWS = pl.dynamic("PRECISION_SPMD_VEC_ROWS")


@pl.jit.inline(auto_scope=False)
def packed_hc_rms(
    sq_partials: pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32],
    vec_ws: pl.InOut[pl.Tensor[[VEC_ROWS, 64], pl.FP32]],
    x_flat: pl.Tensor[[T, HC_DIM], pl.FP32],
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(x_flat, 0)
    t_linear = pl.tensor.dim(vec_ws, 0) - 8
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    for rms_block in pl.range(worker, token_tiles, 48):
        rms_t0 = rms_block * T_TILE
        rms_lanes = pl.arange(0, [1, T_TILE], dtype=pl.INT32)
        rms_rows = pl.add(rms_lanes, pl.cast(rms_t0, pl.INT32))
        rms_local = pl.add(rms_lanes, pl.cast(rms_t0 % 16, pl.INT32))
        rms_index = pl.add(pl.mul(rms_rows, 16), rms_local)
        rms_total = pl.gather(sq_partials, index=rms_index)
        for rms_part in pl.range(1, LINEAR_OK):
            rms_part_index = pl.add(rms_index, pl.cast(rms_part * t_linear * 16, pl.INT32))
            rms_value = pl.gather(sq_partials, index=rms_part_index)
            rms_total = pl.add(rms_total, rms_value)
        rms_variance = pl.add(pl.mul(rms_total, HC_DIM_INV), NORM_EPS)
        rms_root = pl.sqrt(rms_variance)
        native_inverse = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=1.0), rms_root)
        vec_ws[rms_t0 : rms_t0 + 1, 40:48] = native_inverse
    return vec_ws


@pl.jit.inline(auto_scope=False)
def packed_hc_linear_reduce(
    mixes_partials: pl.Tensor[[PARTIAL_ROWS, MIX_PAD], pl.FP32],
    vec_ws: pl.InOut[pl.Tensor[[VEC_ROWS, 64], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    t_linear = pl.tensor.dim(vec_ws, 0) - 8
    for linear_block in pl.range(worker, t_linear // LINEAR_T_TILE, 48):
        linear_t0 = linear_block * LINEAR_T_TILE
        mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + linear_t0
            partial_tile = mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD]
            mixes_total = pl.add(mixes_total, partial_tile)
        vec_ws[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mixes_total
    return vec_ws


@pl.jit.inline(auto_scope=False)
def packed_hc_pre_post(
    vec_ws: pl.InOut[pl.Tensor[[VEC_ROWS, 64], pl.FP32]],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    post: pl.InOut[pl.Tensor[[T, HC_MULT], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(post, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    for ob in pl.range(worker, token_tiles, 48):
        t0 = ob * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        inv_col = pl.reshape(pl.load(vec_ws, [t0, 40], [1, T_TILE], target_memory=pl.MemorySpace.Vec), [T_TILE, 1])
        pre_base = pl.reshape(pl.load(hc_base, [0], [HC_PAD], target_memory=pl.MemorySpace.Vec), [1, HC_PAD])
        pre_scaled = pl.mul(
            pl.row_expand_mul(pl.load(vec_ws, [t0, 0], [T_TILE, HC_PAD], target_memory=pl.MemorySpace.Vec), inv_col),
            scale0,
        )
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.div(
            pl.tile.full([T_TILE, HC_PAD], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(pre_logits)), 1.0)
        )
        pre_val = pl.add(pre_sig, HC_EPS)
        pl.store(pre_val, [t0, 0 + 32], vec_ws)
        post_base = pl.reshape(pl.load(hc_base, [HC_MULT], [HC_PAD], target_memory=pl.MemorySpace.Vec), [1, HC_PAD])
        post_scaled = pl.mul(
            pl.row_expand_mul(
                pl.load(vec_ws, [t0, HC_MULT], [T_TILE, HC_PAD], target_memory=pl.MemorySpace.Vec), inv_col
            ),
            scale1,
        )
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.div(
            pl.tile.full([T_TILE, HC_PAD], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(post_logits)), 1.0)
        )
        post_pad = pl.mul(post_sig, 2.0)
        if valid_rows == T_TILE:
            pl.store(pl.slice(post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[T_TILE, HC_MULT]), [t0, 0], post)
        else:
            pl.store(post_pad, [0 + (pl.tensor.dim(vec_ws, 0) - 8), 0], vec_ws)
            post_tile = pl.load(
                vec_ws,
                [0 + (pl.tensor.dim(vec_ws, 0) - 8), 0],
                [T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(post_tile, [t0, 0], post)
    return (vec_ws, post)


@pl.jit.inline(auto_scope=False)
def packed_hc_comb(
    vec_ws: pl.InOut[pl.Tensor[[VEC_ROWS, 64], pl.FP32]],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base_2d: pl.Tensor[[1, MIX_HC], pl.FP32],
    comb: pl.InOut[pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32]],
    row_recip: pl.constexpr,
    worker: pl.Scalar[pl.INDEX],
):
    # Preserve the Native Sinkhorn divisions and row/column reduction order.
    t_dim = pl.tensor.dim(comb, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    for ob in pl.range(worker, token_tiles, 48):
        scale2 = pl.read(hc_scale, [2])
        t0 = ob * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        inv_col_t = pl.set_validshape(
            pl.reshape(pl.load(vec_ws, [t0, 40], [1, COMB_T_TILE], target_memory=pl.MemorySpace.Vec), [COMB_T_TILE, 1]),
            valid_rows,
            1,
        )
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(
            vec_ws,
            [t0, comb_off + 0 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g1 = pl.load(
            vec_ws,
            [t0, comb_off + 1 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g2 = pl.load(
            vec_ws,
            [t0, comb_off + 2 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g3 = pl.load(
            vec_ws,
            [t0, comb_off + 3 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb0 = pl.load(
            hc_base_2d,
            [0, comb_off + 0 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb1 = pl.load(
            hc_base_2d,
            [0, comb_off + 1 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb2 = pl.load(
            hc_base_2d,
            [0, comb_off + 2 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb3 = pl.load(
            hc_base_2d,
            [0, comb_off + 3 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        row0 = pl.add(pl.mul(pl.row_expand_mul(mix_g0, inv_col_t), scale2), pl.col_expand(mix_g0, cb0))
        row1 = pl.add(pl.mul(pl.row_expand_mul(mix_g1, inv_col_t), scale2), pl.col_expand(mix_g1, cb1))
        row2 = pl.add(pl.mul(pl.row_expand_mul(mix_g2, inv_col_t), scale2), pl.col_expand(mix_g2, cb2))
        row3 = pl.add(pl.mul(pl.row_expand_mul(mix_g3, inv_col_t), scale2), pl.col_expand(mix_g3, cb3))
        row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
        row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
        row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
        row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)
        row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row0_max = pl.row_max(row0_p, row_max_tmp)
        row1_max = pl.row_max(row1_p, row_max_tmp)
        row2_max = pl.row_max(row2_p, row_max_tmp)
        row3_max = pl.row_max(row3_p, row_max_tmp)
        row0_exp = pl.exp(pl.row_expand_sub(row0_p, row0_max))
        row1_exp = pl.exp(pl.row_expand_sub(row1_p, row1_max))
        row2_exp = pl.exp(pl.row_expand_sub(row2_p, row2_max))
        row3_exp = pl.exp(pl.row_expand_sub(row3_p, row3_max))
        row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
        row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
        row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
        row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
        if row_recip:
            row0_inv = pl.recip(row0_sum)
            row0_prob = pl.row_expand_mul(row0_exp, row0_inv)
            row0_soft = pl.add(row0_prob, HC_EPS)
            row1_inv = pl.recip(row1_sum)
            row1_prob = pl.row_expand_mul(row1_exp, row1_inv)
            row1_soft = pl.add(row1_prob, HC_EPS)
            row2_inv = pl.recip(row2_sum)
            row2_prob = pl.row_expand_mul(row2_exp, row2_inv)
            row2_soft = pl.add(row2_prob, HC_EPS)
            row3_inv = pl.recip(row3_sum)
            row3_prob = pl.row_expand_mul(row3_exp, row3_inv)
            row3_soft = pl.add(row3_prob, HC_EPS)
        else:
            row0_soft = pl.add(pl.row_expand_div(row0_exp, row0_sum), HC_EPS)
            row1_soft = pl.add(pl.row_expand_div(row1_exp, row1_sum), HC_EPS)
            row2_soft = pl.add(pl.row_expand_div(row2_exp, row2_sum), HC_EPS)
            row3_soft = pl.add(pl.row_expand_div(row3_exp, row3_sum), HC_EPS)
        row0_valid = pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT)
        row1_valid = pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT)
        row2_valid = pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT)
        row3_valid = pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT)
        row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
        row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
        row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
        row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)
        row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        col_sum = pl.add(pl.add(pl.add(row0_eff, row1_eff), row2_eff), row3_eff)
        col_sum = pl.add(col_sum, HC_EPS)
        row0_cur = pl.div(row0_eff, col_sum)
        row1_cur = pl.div(row1_eff, col_sum)
        row2_cur = pl.div(row2_eff, col_sum)
        row3_cur = pl.div(row3_eff, col_sum)
        for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0_rowsum = pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS)
            row1_rowsum = pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS)
            row2_rowsum = pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS)
            row3_rowsum = pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS)
            if row_recip:
                row0_inv = pl.recip(row0_rowsum)
                row0_norm = pl.row_expand_mul(row0_cur, row0_inv)
                row1_inv = pl.recip(row1_rowsum)
                row1_norm = pl.row_expand_mul(row1_cur, row1_inv)
                row2_inv = pl.recip(row2_rowsum)
                row2_norm = pl.row_expand_mul(row2_cur, row2_inv)
                row3_inv = pl.recip(row3_rowsum)
                row3_norm = pl.row_expand_mul(row3_cur, row3_inv)
            else:
                row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
                row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
                row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
                row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
            col_sum = pl.add(pl.add(pl.add(row0_norm, row1_norm), row2_norm), row3_norm)
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_norm, col_sum)
            row1_cur = pl.div(row1_norm, col_sum)
            row2_cur = pl.div(row2_norm, col_sum)
            row3_cur = pl.div(row3_norm, col_sum)
        if valid_rows == COMB_T_TILE:
            row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
            row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
            row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
            row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
            pl.store(row0_out, [t0, 0 * HC_MULT], comb)
            pl.store(row1_out, [t0, 1 * HC_MULT], comb)
            pl.store(row2_out, [t0, 2 * HC_MULT], comb)
            pl.store(row3_out, [t0, 3 * HC_MULT], comb)
        else:
            pl.store(row0_cur, [0 + (pl.tensor.dim(vec_ws, 0) - 8), 0 * HC_PAD + 8], vec_ws)
            pl.store(row1_cur, [0 + (pl.tensor.dim(vec_ws, 0) - 8), 1 * HC_PAD + 8], vec_ws)
            pl.store(row2_cur, [0 + (pl.tensor.dim(vec_ws, 0) - 8), 2 * HC_PAD + 8], vec_ws)
            pl.store(row3_cur, [0 + (pl.tensor.dim(vec_ws, 0) - 8), 3 * HC_PAD + 8], vec_ws)
            row0_tail = pl.load(
                vec_ws,
                [0 + (pl.tensor.dim(vec_ws, 0) - 8), 0 * HC_PAD + 8],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row1_tail = pl.load(
                vec_ws,
                [0 + (pl.tensor.dim(vec_ws, 0) - 8), 1 * HC_PAD + 8],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row2_tail = pl.load(
                vec_ws,
                [0 + (pl.tensor.dim(vec_ws, 0) - 8), 2 * HC_PAD + 8],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row3_tail = pl.load(
                vec_ws,
                [0 + (pl.tensor.dim(vec_ws, 0) - 8), 3 * HC_PAD + 8],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(row0_tail, [t0, 0 * HC_MULT], comb)
            pl.store(row1_tail, [t0, 1 * HC_MULT], comb)
            pl.store(row2_tail, [t0, 2 * HC_MULT], comb)
            pl.store(row3_tail, [t0, 3 * HC_MULT], comb)
    return (vec_ws, comb)


@pl.jit.inline(auto_scope=False)
def packed_hc_mix(
    x_flat: pl.Tensor[[T, HC_DIM], pl.FP32],
    vec_ws: pl.Tensor[[VEC_ROWS, 64], pl.FP32],
    x_mixed: pl.InOut[pl.Tensor[[T, D], pl.BF16]],
    x_mixed_tail_store: pl.InOut[pl.Tensor[[T_TILE, D], pl.BF16]],
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(x_flat, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    for blk in pl.range(worker, token_tiles * (D // D_SPMD), 48):
        t0 = blk // (D // D_SPMD) * T_TILE
        d_base = blk % (D // D_SPMD) * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile_t = pl.transpose(vec_ws[t0 : t0 + T_TILE, 0 + 32 : HC_PAD + 32], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[valid_rows, D_TILE])
            x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[valid_rows, D_TILE])
            x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, D_TILE])
            x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, D_TILE])
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(pl.add(y0, y1), y2), y3)
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            if valid_rows == T_TILE:
                x_mixed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = y_bf16
            else:
                x_mixed_tail_store[0:T_TILE, d0 : d0 + D_TILE] = y_bf16
                y_out = pl.load(
                    x_mixed_tail_store,
                    [0, d0],
                    [T_TILE, D_TILE],
                    valid_shape=[valid_rows, D_TILE],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(y_out, [t0, d0], x_mixed)
    return (x_mixed, x_mixed_tail_store)


@pl.jit.inline(auto_scope=False)
def packed_norm_rows(
    x: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[4096], pl.BF16],
    output: pl.Out[pl.Tensor[[T, 4096], pl.BF16]],
    vec_ws: pl.InOut[pl.Tensor[[VEC_ROWS, 64], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    # Keep the exact INT64 square-root correction before the Native scalar division.
    rows = pl.tensor.dim(x, 0)
    for row in pl.range(worker, rows, 48):
        raw = pl.load(x, [row, 0], [1, 4096], target_memory=pl.MemorySpace.Vec)
        wide = pl.cast(raw, pl.FP32)
        square = pl.mul(wide, wide)
        fold2048 = pl.add(pl.slice(square, [1, 2048], [0, 0]), pl.slice(square, [1, 2048], [0, 2048]))
        fold1024 = pl.add(pl.slice(fold2048, [1, 1024], [0, 0]), pl.slice(fold2048, [1, 1024], [0, 1024]))
        fold512 = pl.add(pl.slice(fold1024, [1, 512], [0, 0]), pl.slice(fold1024, [1, 512], [0, 512]))
        fold256 = pl.add(pl.slice(fold512, [1, 256], [0, 0]), pl.slice(fold512, [1, 256], [0, 256]))
        fold128 = pl.add(pl.slice(fold256, [1, 128], [0, 0]), pl.slice(fold256, [1, 128], [0, 128]))
        fold64 = pl.add(pl.slice(fold128, [1, 64], [0, 0]), pl.slice(fold128, [1, 64], [0, 64]))
        repeated = pl.col_expand(pl.create_tile([8, 64], dtype=pl.FP32), fold64)
        total = pl.row_sum(repeated, pl.create_tile([8, 64], dtype=pl.FP32))
        total_scalar = pl.read(total, [0, 0])
        total_vector = pl.mul(pl.full([1, 8], dtype=pl.FP32, value=1.0), total_scalar)
        variance = pl.add(pl.mul(total_vector, 1.0 / 4096.0), 1e-06)
        root_value = pl.sqrt(variance)
        variance_bits = pl.reinterpret_view(variance, pl.INT32)
        approx_bits = pl.reinterpret_view(root_value, pl.INT32)
        x_bits = pl.cast(pl.read(variance_bits, [0, 0]), pl.INT64)
        y_bits = pl.cast(pl.read(approx_bits, [0, 0]), pl.INT64)
        fraction_unit = FRACTION_UNIT
        if x_bits >= fraction_unit and x_bits < FP32_INFINITY_BITS:
            x_exp = x_bits // fraction_unit
            x_sig = x_bits % fraction_unit + fraction_unit
            for correction_step in pl.range(SQRT_CORRECTION_STEPS):
                y_exp = y_bits // fraction_unit
                y_sig = y_bits % fraction_unit + fraction_unit
                upper_mid = 2 * y_sig + 1
                upper_square = upper_mid * upper_mid
                x_upper = x_sig << x_exp - 2 * y_exp + SQRT_COMPARE_SHIFT
                previous_bits = y_bits - 1
                previous_exp = previous_bits // fraction_unit
                previous_sig = previous_bits % fraction_unit + fraction_unit
                lower_mid = 2 * previous_sig + 1
                lower_square = lower_mid * lower_mid
                x_lower = x_sig << x_exp - 2 * previous_exp + SQRT_COMPARE_SHIFT
                if x_upper > upper_square or (x_upper == upper_square and y_bits % 2 == 1):
                    y_bits = y_bits + 1
                elif x_lower < lower_square or (x_lower == lower_square and y_bits % 2 == 1):
                    y_bits = y_bits - 1
        rounded_bits = pl.full([1, 8], dtype=pl.INT32, value=0)
        for root_lane in pl.range(8):
            pl.write(rounded_bits, [0, root_lane], pl.cast(y_bits, pl.INT32))
        rounded_root = pl.reinterpret_view(rounded_bits, pl.FP32)
        scalar_root_value = pl.read(rounded_root, [0, 0])
        scalar_inv = 1.0 / scalar_root_value
        inverse = pl.mul(pl.full([1, 8], dtype=pl.FP32, value=1.0), scalar_inv)
        vec_ws[row : row + 1, 0 + 48 : 8 + 48] = inverse
        apply_value = wide
        w_raw = pl.reshape(pl.load(weight, [0], [4096], target_memory=pl.MemorySpace.Vec), [1, 4096])
        w = pl.cast(w_raw, pl.FP32)
        inv = pl.read(inverse, [0, 0])
        scaled = pl.mul(apply_value, inv)
        result = pl.col_expand_mul(scaled, w)
        pl.store(pl.cast(result, pl.BF16, mode="rint"), [row, 0], output)
    return (output, vec_ws)


@pl.jit.inline(auto_scope=False)
def indexer_projection_matmul(
    x: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[64, 4096], pl.BF16],
    order: pl.Tensor[[16, 4, 16], pl.INT32],
    accum: pl.InOut[pl.Tensor[[P16, 64], pl.FP32]],
    physical_worker: pl.Scalar[pl.INDEX],
):
    rows = pl.tensor.dim(x, 0)
    row_tiles = (rows + 15) // 16
    for worker in pl.range(physical_worker, row_tiles * 4, 24):
        rb = worker // 4
        group = worker % 4
        row = rb * 16
        col = group * 16
        valid = pl.min(16, rows - row)
        tile = pl.create_tensor([16, 16], dtype=pl.FP32)
        for step in pl.range(16):
            k_block = pl.read(order, [rb, group, step])
            offset = pl.max(k_block, 0) * 256
            a = pl.slice(x, [16, 256], [row, offset], valid_shape=[valid, 256])
            b = weight[col : col + 16, offset : offset + 256]
            tile = pl.matmul_acc(tile, a, b, b_trans=True, init_cond=step == 0)
        accum[row : row + 16, col : col + 16] = tile
    return accum


@pl.jit.inline(auto_scope=False)
def indexer_projection_round(
    accum: pl.Tensor[[P16, 64], pl.FP32],
    raw: pl.InOut[pl.Tensor[[T, 64], pl.BF16]],
    scaled: pl.InOut[pl.Tensor[[T, 64], pl.BF16]],
    output: pl.InOut[pl.Tensor[[T, 64], pl.FP16]],
    physical_worker: pl.Scalar[pl.INDEX],
):
    rows = pl.tensor.dim(output, 0)
    row_tiles = (rows + 15) // 16
    for block in pl.range(physical_worker, row_tiles, 48):
        row = block * 16
        valid = pl.min(16, rows - row)
        value = pl.load(accum, [row, 0], [16, 64], valid_shape=[valid, 64])
        projected = pl.cast(value, pl.BF16, mode="rint")
        pl.store(projected, [row, 0], raw)
        value = pl.cast(projected, pl.FP32)
        value = pl.mul(value, 0.011048543456039806)
        weighted = pl.cast(value, pl.BF16, mode="rint")
        pl.store(weighted, [row, 0], scaled)
        result = pl.cast(weighted, pl.FP16, mode="rint")
        pl.store(result, [row, 0], output)
    return (raw, scaled, output)
