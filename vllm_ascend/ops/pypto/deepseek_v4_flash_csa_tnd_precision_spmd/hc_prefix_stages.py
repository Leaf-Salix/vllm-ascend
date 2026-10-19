# SPDX-License-Identifier: Apache-2.0
"""Native-order HC prefix stages for the precision SPMD candidate."""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_full_precision.hc_pre import (
    COMB_T_TILE,
    D_SPMD,
    D_TILE,
    HC_DIM,
    HC_DIM_INV,
    HC_EPS,
    HC_MULT,
    HC_PAD,
    HC_SINKHORN_ITER,
    LINEAR_K_PER_SPLIT,
    LINEAR_OK,
    LINEAR_T_TILE,
    MIX_HC,
    MIX_PAD,
    NORM_EPS,
    RMS_K_TILE,
    T_TILE,
    D,
)

from .norm_wkv_stages import T

P16 = pl.dynamic("PRECISION_SPMD_HC_P16")
PARTIAL_ROWS = pl.dynamic("PRECISION_SPMD_HC_PARTIAL_ROWS")


@pl.jit.inline(auto_scope=False)
def hc_widen(
    flat: pl.Tensor[[T, HC_DIM], pl.BF16],
    wide_flat: pl.InOut[pl.Tensor[[T, HC_DIM], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    tokens = pl.tensor.dim(flat, 0)
    for row in pl.range(worker, tokens, 48):
        for column in pl.pipeline(HC_DIM // RMS_K_TILE, stage=2):
            start = column * RMS_K_TILE
            value = pl.cast(flat[row : row + 1, start : start + RMS_K_TILE], pl.FP32)
            wide_flat[row : row + 1, start : start + RMS_K_TILE] = value
    return wide_flat


@pl.jit.inline(auto_scope=False)
def hc_square(
    x_flat: pl.Tensor[[T, HC_DIM], pl.FP32],
    sq_partials: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(x_flat, 0)
    t_linear = pl.tensor.dim(sq_partials, 0) // LINEAR_OK
    sq_group = 1
    if t_dim >= 64:
        sq_group = 4
    sq_groups = (LINEAR_OK + sq_group - 1) // sq_group
    sq_units = t_linear // 16 * sq_groups
    for sq_unit in pl.range(worker, sq_units, 24):
        sq_t0 = sq_unit // sq_groups * 16
        sq_part_base = sq_unit % sq_groups * sq_group
        sq_rows = pl.min(16, t_dim - sq_t0)
        for sq_group_item in pl.range(pl.min(sq_group, LINEAR_OK - sq_part_base)):
            sq_part = sq_part_base + sq_group_item
            sq_count = pl.min(768, HC_DIM - sq_part * 768) // 32
            native_sq_acc = pl.create_tensor([16, 16], dtype=pl.FP32)
            for sq_kb in pl.pipeline(0, sq_count, stage=2):
                sq_k = sq_part * 768 + sq_kb * 32
                sq_lhs = pl.slice(x_flat, [16, 32], [sq_t0, sq_k], valid_shape=[sq_rows, 32])
                native_sq_acc = pl.matmul_acc(native_sq_acc, sq_lhs, sq_lhs, b_trans=True, init_cond=sq_kb == 0)
            sq_out_row = sq_part * t_linear + sq_t0
            sq_partials[sq_out_row : sq_out_row + 16, 0:16] = native_sq_acc
    return sq_partials


@pl.jit.inline(auto_scope=False)
def hc_rms(
    sq_partials: pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32],
    inv_rms: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    x_flat: pl.Tensor[[T, HC_DIM], pl.FP32],
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(x_flat, 0)
    t_linear = pl.tensor.dim(inv_rms, 0)
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
        inv_rms[rms_t0 : rms_t0 + T_TILE, 0:1] = pl.reshape(native_inverse, [T_TILE, 1])
    return inv_rms


@pl.jit.inline(auto_scope=False)
def hc_pack_weight(
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    panel64_packed_weight: pl.InOut[pl.Tensor[[HC_DIM * 2, MIX_PAD], pl.BF16]],
    worker: pl.Scalar[pl.INDEX],
):
    for panel64_block in pl.range(worker, HC_DIM // 256, 48):
        panel64_raw: pl.Tile[[32, 256], pl.FP32] = pl.load(
            hc_fn, [0, panel64_block * 256], [32, 256], valid_shape=[MIX_HC, 256]
        )
        panel64_transposed = pl.tile.transpose(panel64_raw, axis1=0, axis2=1)
        for panel64_pad_column in pl.range(MIX_HC, MIX_PAD):
            for panel64_pad_k in pl.range(256):
                pl.write(panel64_transposed, [panel64_pad_k, panel64_pad_column], 0.0)
        panel64_transposed = pl.set_validshape(panel64_transposed, 256, 32)
        panel64_bits = pl.tile.reinterpret_view(panel64_transposed, pl.INT32)
        panel64_converted = pl.tile.reinterpret_view(
            pl.tile.and_(pl.tile.add(panel64_bits, 2048), pl.tile.full([256, 32], dtype=pl.INT32, value=-4096)), pl.FP32
        )
        panel64_hi_bits = pl.tile.and_(
            pl.tile.reinterpret_view(panel64_converted, pl.INT32), pl.tile.full([256, 32], dtype=pl.INT32, value=-65536)
        )
        panel64_hi_fp32 = pl.tile.reinterpret_view(panel64_hi_bits, pl.FP32)
        panel64_lo_fp32 = pl.tile.sub(panel64_converted, panel64_hi_fp32)
        panel64_hi_bf16 = pl.tile.cast(panel64_hi_fp32, pl.BF16)
        panel64_lo_bf16 = pl.tile.cast(panel64_lo_fp32, pl.BF16)
        panel64_packed_block = pl.tile.full([512, 32], dtype=pl.BF16, value=0.0)
        for panel64_chunk in pl.range(32):
            panel64_high8 = pl.tile.slice(panel64_hi_bf16, [8, 32], [panel64_chunk * 8, 0])
            panel64_low8 = pl.tile.slice(panel64_lo_bf16, [8, 32], [panel64_chunk * 8, 0])
            panel64_packed_block = pl.tile.assemble(panel64_packed_block, panel64_high8, [panel64_chunk * 16, 0])
            panel64_packed_block = pl.tile.assemble(panel64_packed_block, panel64_low8, [panel64_chunk * 16 + 8, 0])
        pl.store(panel64_packed_block, [panel64_block * 512, 0], panel64_packed_weight)
    return panel64_packed_weight


@pl.jit.inline(auto_scope=False)
def hc_pack_input(
    x_flat: pl.Tensor[[T, HC_DIM], pl.FP32],
    panel64_packed_flat: pl.InOut[pl.Tensor[[P16, HC_DIM * 2], pl.BF16]],
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(x_flat, 0)
    t_linear = pl.tensor.dim(panel64_packed_flat, 0)
    panel64_inputs_per_task = 1
    if t_dim >= 64:
        panel64_inputs_per_task = 4
    for panel64_input_group in pl.range(worker, t_linear // 16 * (HC_DIM // 256) // panel64_inputs_per_task, 48):
        for panel64_input_local in pl.range(panel64_inputs_per_task):
            panel64_transpose_unit = panel64_input_group * panel64_inputs_per_task + panel64_input_local
            panel64_t0 = panel64_transpose_unit // (HC_DIM // 256) * 16
            panel64_k0 = panel64_transpose_unit % (HC_DIM // 256) * 256
            panel64_values = pl.load(
                x_flat, [panel64_t0, panel64_k0], [16, 256], valid_shape=[pl.min(16, t_dim - panel64_t0), 256]
            )
            for panel64_invalid_row in pl.range(pl.min(16, t_dim - panel64_t0), 16):
                for panel64_invalid_col in pl.range(256):
                    pl.write(panel64_values, [panel64_invalid_row, panel64_invalid_col], 0.0)
            panel64_values = pl.set_validshape(panel64_values, 16, 256)
            panel64_transposed_values = pl.tile.transpose(panel64_values, axis1=0, axis2=1)
            panel64_packed_transposed = pl.tile.full([512, 16], dtype=pl.FP32, value=0.0)
            for panel64_chunk in pl.range(32):
                panel64_input8 = pl.tile.slice(panel64_transposed_values, [8, 16], [panel64_chunk * 8, 0])
                panel64_packed_transposed = pl.tile.assemble(
                    panel64_packed_transposed, panel64_input8, [panel64_chunk * 16, 0]
                )
                panel64_packed_transposed = pl.tile.assemble(
                    panel64_packed_transposed, panel64_input8, [panel64_chunk * 16 + 8, 0]
                )
            panel64_packed_values = pl.tile.transpose(panel64_packed_transposed, axis1=0, axis2=1)
            pl.store(pl.tile.cast(panel64_packed_values, pl.BF16), [panel64_t0, panel64_k0 * 2], panel64_packed_flat)
    return panel64_packed_flat


@pl.jit.inline(auto_scope=False)
def hc_linear(
    x_flat: pl.Tensor[[T, HC_DIM], pl.FP32],
    panel64_packed_flat: pl.Tensor[[P16, HC_DIM * 2], pl.BF16],
    panel64_packed_weight: pl.Tensor[[HC_DIM * 2, MIX_PAD], pl.BF16],
    mixes_partials: pl.InOut[pl.Tensor[[PARTIAL_ROWS, MIX_PAD], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(x_flat, 0)
    t_linear = pl.tensor.dim(panel64_packed_flat, 0)
    if t_dim >= 64:
        for panel64_wide_unit in pl.range(worker, LINEAR_OK * ((t_linear + 63) // 64), 24):
            panel64_wide_split = panel64_wide_unit // ((t_linear + 63) // 64)
            panel64_wide_row = panel64_wide_unit % ((t_linear + 63) // 64) * 64
            panel64_wide_valid = pl.min(64, t_linear - panel64_wide_row)
            panel64_wide_count = pl.min(LINEAR_K_PER_SPLIT, HC_DIM - panel64_wide_split * LINEAR_K_PER_SPLIT) // 64
            panel64_wide_first_k = panel64_wide_split * LINEAR_K_PER_SPLIT * 2
            panel64_wide_first_x = pl.slice(
                panel64_packed_flat,
                [64, 128],
                [panel64_wide_row, panel64_wide_first_k],
                valid_shape=[panel64_wide_valid, 128],
            )
            panel64_wide_first_w = pl.slice(panel64_packed_weight, [128, MIX_PAD], [panel64_wide_first_k, 0])
            panel64_wide_accumulator = pl.matmul(panel64_wide_first_x, panel64_wide_first_w, out_dtype=pl.FP32)
            for panel64_wide_kb in pl.range(1, panel64_wide_count):
                panel64_wide_packed_k = panel64_wide_split * LINEAR_K_PER_SPLIT * 2 + panel64_wide_kb * 128
                panel64_wide_x = pl.slice(
                    panel64_packed_flat,
                    [64, 128],
                    [panel64_wide_row, panel64_wide_packed_k],
                    valid_shape=[panel64_wide_valid, 128],
                )
                panel64_wide_w = pl.slice(panel64_packed_weight, [128, MIX_PAD], [panel64_wide_packed_k, 0])
                panel64_wide_accumulator = pl.matmul_acc(
                    panel64_wide_accumulator, panel64_wide_x, panel64_wide_w, init_cond=False
                )
            panel64_wide_store = pl.tensor.set_validshape(panel64_wide_accumulator, panel64_wide_valid, MIX_PAD)
            mixes_partials = pl.assemble(
                mixes_partials, panel64_wide_store, [panel64_wide_split * t_linear + panel64_wide_row, 0]
            )
    else:
        for panel64_unit in pl.range(worker, LINEAR_OK * (t_linear // 16), 24):
            panel64_split = panel64_unit // (t_linear // 16)
            panel64_row = panel64_unit % (t_linear // 16) * 16
            panel64_count = pl.min(LINEAR_K_PER_SPLIT, HC_DIM - panel64_split * LINEAR_K_PER_SPLIT) // 64
            panel64_accumulator = pl.create_tensor([16, MIX_PAD], dtype=pl.FP32)
            for panel64_kb in pl.range(panel64_count):
                panel64_packed_k = panel64_split * LINEAR_K_PER_SPLIT * 2 + panel64_kb * 128
                panel64_x16 = pl.slice(panel64_packed_flat, [16, 128], [panel64_row, panel64_packed_k])
                panel64_w16 = pl.slice(panel64_packed_weight, [128, MIX_PAD], [panel64_packed_k, 0])
                panel64_accumulator = pl.matmul_acc(
                    panel64_accumulator, panel64_x16, panel64_w16, init_cond=panel64_kb == 0
                )
            mixes_partials = pl.assemble(
                mixes_partials, panel64_accumulator, [panel64_split * t_linear + panel64_row, 0]
            )
    return mixes_partials


@pl.jit.inline(auto_scope=False)
def hc_linear_reduce(
    mixes_partials: pl.Tensor[[PARTIAL_ROWS, MIX_PAD], pl.FP32],
    mixes_raw: pl.InOut[pl.Tensor[[P16, MIX_PAD], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    t_linear = pl.tensor.dim(mixes_raw, 0)
    for linear_block in pl.range(worker, t_linear // LINEAR_T_TILE, 48):
        linear_t0 = linear_block * LINEAR_T_TILE
        mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + linear_t0
            partial_tile = mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD]
            mixes_total = pl.add(mixes_total, partial_tile)
        mixes_raw[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mixes_total
    return mixes_raw


@pl.jit.inline(auto_scope=False)
def hc_pre_post(
    mixes_raw: pl.Tensor[[P16, MIX_PAD], pl.FP32],
    inv_rms: pl.Tensor[[P16, 1], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.InOut[pl.Tensor[[P16, HC_PAD], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T, HC_MULT], pl.FP32]],
    post_tail_store: pl.InOut[pl.Tensor[[T_TILE, HC_PAD], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(post, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    for ob in pl.range(worker, token_tiles, 48):
        t0 = ob * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        inv_col = inv_rms[t0 : t0 + T_TILE, 0:1]
        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.div(pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val = pl.add(pre_sig, HC_EPS)
        pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD] = pre_val
        post_base = pl.reshape(hc_base[HC_MULT : HC_MULT + HC_PAD], [1, HC_PAD])
        post_scaled = pl.mul(
            pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, HC_MULT : HC_MULT + HC_PAD], inv_col), scale1
        )
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.div(pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(post_logits)), 1.0))
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
    return (pre_val_store, post, post_tail_store)


@pl.jit.inline(auto_scope=False)
def hc_comb(
    mixes_raw: pl.Tensor[[P16, MIX_PAD], pl.FP32],
    inv_rms: pl.Tensor[[P16, 1], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base_2d: pl.Tensor[[1, MIX_HC], pl.FP32],
    comb: pl.InOut[pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32]],
    comb_tail_store: pl.InOut[pl.Tensor[[COMB_T_TILE, HC_PAD * HC_MULT], pl.FP32]],
    row_recip: pl.constexpr,
    worker: pl.Scalar[pl.INDEX],
):
    t_dim = pl.tensor.dim(comb, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    for ob in pl.range(worker, token_tiles, 48):
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
    return (comb, comb_tail_store)


@pl.jit.inline(auto_scope=False)
def hc_mix(
    x_flat: pl.Tensor[[T, HC_DIM], pl.FP32],
    pre_val_store: pl.Tensor[[P16, HC_PAD], pl.FP32],
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
    return (x_mixed, x_mixed_tail_store)
