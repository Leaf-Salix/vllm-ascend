# SPDX-License-Identifier: Apache-2.0
"""HC-pre Native-order candidate, T<=256/A3 only.

The HC projection uses an explicit PTO C++ HF32 kernel via pl.jit.extern.
All remaining arithmetic is DSL; input RMSNorm is outside this module.
Original Native weights are used unchanged. No compiler interception.
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa_tnd_precision.config import FLASH as M

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # actual token rows, independent request lengths
HC_PAD_ROWS_DYN = pl.dynamic("HC_PAD_ROWS_DYN")
LINEAR_ROWS_DYN = pl.dynamic("LINEAR_ROWS_DYN")

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

# tiling
MIX_PAD = 32  # mix_hc (24) padded to a 32-wide cube N / vector row
HC_PAD = 8  # hc (4) padded for 32B-aligned vector ops
T_TILE = 8  # other values miscompare
LINEAR_T_TILE = 16  # cube matmul rows must be a 16-row boxed tile
COMB_T_TILE = 8
PRE_POST_WORKERS = 16
RMS_K_TILE = 512
LINEAR_K_TILE = 32
D_TILE = 256
D_SPMD = 4096
LINEAR_OK = 22
LINEAR_K_PER_SPLIT = 768
LINEAR_WORKERS = 24  # hc_pre_linear workers, one per AIC

# pre0..pre3 / row0..row3 are hand-unrolled over the hc lanes.
assert HC_MULT == 4, f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}"


@pl.jit.extern(source="hc_pre_linear_hf32.cpp", core_type="aic")
def hc_pre_linear_hf32(
    partials: pl.Out[pl.Tensor[[LINEAR_ROWS_DYN, MIX_PAD], pl.FP32]],
    flat: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    weight: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    units: pl.Scalar[pl.INDEX],
    workers: pl.Scalar[pl.INDEX],
    tokens: pl.Scalar[pl.INDEX],
    padded_tokens: pl.Scalar[pl.INDEX],
) -> pl.Tensor[[LINEAR_ROWS_DYN, MIX_PAD], pl.FP32]: ...


@pl.jit.inline
def hc_pre_gates_from_rms(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Tensor[[HC_PAD_ROWS_DYN, HC_PAD], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    row_recip: pl.Scalar[pl.BOOL],
    inv_rms: pl.Tensor[[HC_PAD_ROWS_DYN, 1], pl.FP32],
):
    """共享门控与Sinkhorn算术，复用调用方按原规则生成的RMS统计。"""
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE  # pad t_dim up to whole 16-row cube tiles
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])  # for per-group comb base loads in comb_sinkhorn

    # linear: split-K matmul -> per-split partials. The t_dim..t_linear pad rows are
    # zero-filled by valid_shape, never materialized.
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    linear_workers = pl.min(linear_units, LINEAR_WORKERS)
    with pl.spmd(linear_workers):
        mixes_partials = hc_pre_linear_hf32(
            mixes_partials, x_flat, hc_fn, linear_units, linear_workers, t_dim, t_linear
        )

    # Partials are reduced in ascending K order.
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for linear_block in pl.spmd(t_linear // LINEAR_T_TILE, name_hint="hc_pre_linear_reduce", allow_early_resolve=True):
        linear_t0 = linear_block * LINEAR_T_TILE
        mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + linear_t0
            partial_tile = mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD]
            mixes_total = pl.add(mixes_total, partial_tile)
        mixes_raw[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mixes_total

    # split_pre_post: inv_rms-scaled pre gate -> pre_val_store (for mix_x), post gate -> post.
    # Both compute at HC_PAD width; post narrows to HC_MULT via a valid-shape slice (an 8-wide
    # 32B tile, 4 cols valid -- a bare 4-wide slice allocs a 16B tile ptoas rejects). comb gate
    # lives in comb_sinkhorn.
    # Only the final partial token tile uses these fixed-size staging buffers.
    post_tail_store = pl.create_tensor([T_TILE, HC_PAD], dtype=pl.FP32)
    for ob_worker in pl.spmd(
        pl.min(token_tiles, PRE_POST_WORKERS), name_hint="split_pre_post", allow_early_resolve=True
    ):
        scale0 = pl.read(hc_scale, [0])
        scale1 = pl.read(hc_scale, [1])
        for ob in pl.range(ob_worker, token_tiles, pl.min(token_tiles, PRE_POST_WORKERS)):
            t0 = ob * T_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            inv_col = inv_rms[t0 : t0 + T_TILE, 0:1]

            pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
            pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
            pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
            pre_sig = pl.div(
                pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(pre_logits)), 1.0)
            )
            pre_val = pl.add(pre_sig, HC_EPS)
            pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD] = pre_val

            post_base = pl.reshape(hc_base[HC_MULT : HC_MULT + HC_PAD], [1, HC_PAD])
            post_scaled = pl.mul(
                pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, HC_MULT : HC_MULT + HC_PAD], inv_col), scale1
            )
            post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
            post_sig = pl.div(
                pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(post_logits)), 1.0)
            )
            post_pad = pl.mul(post_sig, 2.0)
            if valid_rows == T_TILE:
                post[t0 : t0 + T_TILE, 0:HC_MULT] = pl.slice(
                    post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[T_TILE, HC_MULT]
                )
            else:
                post_tail_store[0:T_TILE, 0:HC_PAD] = post_pad
                post_tile = pl.load(
                    post_tail_store,
                    [0, 0],
                    [T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(post_tile, [t0, 0], post)

    # comb_sinkhorn: comb gate from mixes_raw cols 8/12/16/20, softmax, then a
    # column-first 20-iteration Sinkhorn -> comb.
    comb_tail_store = pl.create_tensor([COMB_T_TILE, HC_PAD * HC_MULT], dtype=pl.FP32)
    for ob in pl.spmd(token_tiles, name_hint="comb_sinkhorn", allow_early_resolve=True):
        scale2 = pl.read(hc_scale, [2])
        t0 = ob * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        inv_col_t = pl.load(
            inv_rms, [t0, 0], [COMB_T_TILE, 1], valid_shape=[valid_rows, 1], target_memory=pl.MemorySpace.Vec
        )
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(
            mixes_raw,
            [t0, comb_off + 0 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g1 = pl.load(
            mixes_raw,
            [t0, comb_off + 1 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g2 = pl.load(
            mixes_raw,
            [t0, comb_off + 2 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g3 = pl.load(
            mixes_raw,
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
            pl.store(row0_cur, [0, 0 * HC_PAD], comb_tail_store)
            pl.store(row1_cur, [0, 1 * HC_PAD], comb_tail_store)
            pl.store(row2_cur, [0, 2 * HC_PAD], comb_tail_store)
            pl.store(row3_cur, [0, 3 * HC_PAD], comb_tail_store)
            row0_tail = pl.load(
                comb_tail_store,
                [0, 0 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row1_tail = pl.load(
                comb_tail_store,
                [0, 1 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row2_tail = pl.load(
                comb_tail_store,
                [0, 2 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row3_tail = pl.load(
                comb_tail_store,
                [0, 3 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(row0_tail, [t0, 0 * HC_MULT], comb)
            pl.store(row1_tail, [t0, 1 * HC_MULT], comb)
            pl.store(row2_tail, [t0, 2 * HC_MULT], comb)
            pl.store(row3_tail, [t0, 3 * HC_MULT], comb)

    return pre_val_store


@pl.jit.inline
def hc_pre_gates(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Tensor[[T_DYN, HC_PAD], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    row_recip: pl.Scalar[pl.BOOL],
):
    """保留原独立RMS任务及其归约次序。"""
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)

    # Diagnostic scope T<=256/AIC24; matched Native Gram partials, then exact K order.
    sq_partials = pl.create_tensor([LINEAR_OK * t_linear, 16], dtype=pl.FP32)
    sq_units = (t_linear // 16) * LINEAR_OK
    for sq_unit in pl.spmd(sq_units, name_hint="hc_pre_square_partial", allow_early_resolve=True):
        sq_t0 = (sq_unit // LINEAR_OK) * 16
        sq_part = sq_unit % LINEAR_OK
        sq_rows = pl.min(16, t_dim - sq_t0)
        sq_count = pl.min(768, HC_DIM - sq_part * 768) // 32
        native_sq_acc = pl.create_tensor([16, 16], dtype=pl.FP32)
        for sq_kb in pl.pipeline(0, sq_count, stage=2):
            sq_k = sq_part * 768 + sq_kb * 32
            sq_lhs = pl.slice(x_flat, [16, 32], [sq_t0, sq_k], valid_shape=[sq_rows, 32])
            native_sq_acc = pl.matmul_acc(native_sq_acc, sq_lhs, sq_lhs, b_trans=True, init_cond=(sq_kb == 0))
        sq_out_row = sq_part * t_linear + sq_t0
        sq_partials[sq_out_row : sq_out_row + 16, 0:16] = native_sq_acc
    for rms_block in pl.spmd(token_tiles, name_hint="hc_pre_native_rms", allow_early_resolve=True):
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
        inv_rms[rms_t0 : rms_t0 + T_TILE, 0:1] = pl.reshape(native_inverse, [T_TILE, 1])

    return hc_pre_gates_from_rms(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        pre_val_store,
        post,
        comb,
        row_recip,
        inv_rms,
    )


def _hc_pre(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    """Compute HC gates and BF16 pre-mixed activations."""
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    hc_pre_gates(x, hc_fn, hc_scale, hc_base, pre_val_store, post, comb, False)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])

    # mix_x: x_mixed = sum_h pre[:,h]*x[:,h,:], fanned over D/D_SPMD blocks per token tile.
    x_mixed_tail_store = pl.create_tensor([T_TILE, D], dtype=pl.BF16)
    for blk in pl.spmd(token_tiles * (D // D_SPMD), name_hint="mix_x", allow_early_resolve=True):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile_t = pl.transpose(pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
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
    return x_mixed


hc_pre = pl.jit.inline(_hc_pre)
hc_pre_fp32_test = pl.jit(_hc_pre)


@pl.jit
def hc_pre_test(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    tokens = pl.tensor.dim(x, 0)
    wide = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    flat = pl.reshape(x, [tokens, HC_DIM])
    wide_flat = pl.reshape(wide, [tokens, HC_DIM])
    # A single valid row per task avoids losing a tail-row valid_shape on cast.
    # This is an exact widening, not a change to the Native BF16 math boundary.
    for row in pl.spmd(tokens, name_hint="hc_pre_widen", allow_early_resolve=True):
        for column in pl.pipeline(HC_DIM // RMS_K_TILE, stage=2):
            start = column * RMS_K_TILE
            value = pl.cast(flat[row : row + 1, start : start + RMS_K_TILE], pl.FP32)
            wide_flat[row : row + 1, start : start + RMS_K_TILE] = value
    hc_pre(wide, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
    return x_mixed, post, comb
