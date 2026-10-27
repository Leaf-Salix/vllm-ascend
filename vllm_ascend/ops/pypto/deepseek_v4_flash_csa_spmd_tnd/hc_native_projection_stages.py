# SPDX-License-Identifier: Apache-2.0
"""对齐指定A3 Native融合HC的HF32转换与22段K768投影。

从旧精度基准的hc_prefix_stages移入已验证数值策略；不改变权重来源。
显式(bit+2048)&-4096模拟HF32转换，BF16高/低位布局保持8乘积累加边界。
这些helper没有任务或信号；调用方负责生产者完成与消费等待。
"""

import pypto.language as pl

from .hc_stages import HC_DIM, LINEAR_T_TILE, MIX_HC, MIX_PAD, PAD_ROWS_DYN, PARTIAL_ROWS_DYN, T_DYN

LINEAR_OK = 22
LINEAR_K_PER_SPLIT = 768


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
    x_flat: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    panel64_packed_flat: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, HC_DIM * 2], pl.BF16]],
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
    x_flat: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    panel64_packed_flat: pl.Tensor[[PAD_ROWS_DYN, HC_DIM * 2], pl.BF16],
    panel64_packed_weight: pl.Tensor[[HC_DIM * 2, MIX_PAD], pl.BF16],
    mixes_partials: pl.InOut[pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32]],
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
    mixes_partials: pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32],
    mixes_raw: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32]],
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
def hc_pack_input_from_bf16(
    x_flat: pl.Tensor[[T_DYN, HC_DIM], pl.BF16],
    panel64_packed_flat: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, HC_DIM * 2], pl.BF16]],
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
            # 无损局部加宽，保留已验证的尾行清零及打包；不读别核的wide GM。
            panel64_values = pl.cast(panel64_values, pl.FP32)
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
