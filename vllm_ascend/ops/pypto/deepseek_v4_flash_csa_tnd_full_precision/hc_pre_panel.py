# SPDX-License-Identifier: Apache-2.0
"""A3 Native HF32 K8边界的BF16单次K16实现；保持真实TND。

HF32权重精确拆为BF16高8/低4位，单次K16消费8个高位与8个低位。
这不是两次独立matmul相加。宽指数硬件行为按Native HF32实测验收；
不声称整数无限精确求和。只使用现有DSL，没有Native算术或C++回调。
"""

import pypto.language as pl

T_DYN = pl.dynamic("T_DYN")
LINEAR_ROWS_DYN = pl.dynamic("LINEAR_ROWS_DYN")
HC_DIM = 16384
MIX_HC = 24
MIX_PAD = 32
LINEAR_OK = 22
LINEAR_K_PER_SPLIT = 768


def _hcpre_linear_panel(
    flat: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    weight: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    partials: pl.Out[pl.Tensor[[LINEAR_ROWS_DYN, MIX_PAD], pl.FP32]],
):
    flat.bind_dynamic(0, T_DYN)
    partials.bind_dynamic(0, LINEAR_ROWS_DYN)
    tokens = pl.tensor.dim(flat, 0)
    padded_tokens = pl.tensor.dim(partials, 0) // LINEAR_OK
    # HF32 operand conversion is explicit bit arithmetic on original FP32 weight.
    # Transpose is DSL layout work, not a Native arithmetic callback.
    packed_weight = pl.create_tensor([HC_DIM * 2, MIX_PAD], dtype=pl.BF16)
    for block in pl.spmd(HC_DIM // 256, name_hint="hcpre_hf32_weight", allow_early_resolve=True):
        raw: pl.Tile[[32, 256], pl.FP32] = pl.load(weight, [0, block * 256], [32, 256], valid_shape=[MIX_HC, 256])
        transposed = pl.tile.transpose(raw, axis1=0, axis2=1)
        for pad_column in pl.range(MIX_HC, MIX_PAD):
            for pad_k in pl.range(256):
                pl.write(transposed, [pad_k, pad_column], 0.0)
        transposed = pl.set_validshape(transposed, 256, 32)
        bits = pl.tile.reinterpret_view(transposed, pl.INT32)
        converted = pl.tile.reinterpret_view(
            pl.tile.and_(pl.tile.add(bits, 2048), pl.tile.full([256, 32], dtype=pl.INT32, value=-4096)), pl.FP32
        )
        # Truncate the HF32 value into exact BF16 high8 and low4 pieces.
        hi_bits = pl.tile.and_(
            pl.tile.reinterpret_view(converted, pl.INT32), pl.tile.full([256, 32], dtype=pl.INT32, value=-65536)
        )
        hi_fp32 = pl.tile.reinterpret_view(hi_bits, pl.FP32)
        lo_fp32 = pl.tile.sub(converted, hi_fp32)
        hi_bf16 = pl.tile.cast(hi_fp32, pl.BF16)
        lo_bf16 = pl.tile.cast(lo_fp32, pl.BF16)
        packed_block = pl.tile.full([512, 32], dtype=pl.BF16, value=0.0)
        for chunk in pl.range(32):
            high8 = pl.tile.slice(hi_bf16, [8, 32], [chunk * 8, 0])
            low8 = pl.tile.slice(lo_bf16, [8, 32], [chunk * 8, 0])
            packed_block = pl.tile.assemble(packed_block, high8, [chunk * 16, 0])
            packed_block = pl.tile.assemble(packed_block, low8, [chunk * 16 + 8, 0])
        pl.store(packed_block, [block * 512, 0], packed_weight)
    packed_flat = pl.create_tensor([padded_tokens, HC_DIM * 2], dtype=pl.BF16)
    inputs_per_task = 1
    if tokens >= 64:
        inputs_per_task = 4
    for input_group in pl.spmd(
        ((padded_tokens // 16) * (HC_DIM // 256)) // inputs_per_task,
        name_hint="hcpre_input_panel",
        allow_early_resolve=True,
    ):
        for input_local in pl.range(inputs_per_task):
            transpose_unit = input_group * inputs_per_task + input_local
            t0 = (transpose_unit // (HC_DIM // 256)) * 16
            k0 = (transpose_unit % (HC_DIM // 256)) * 256
            values = pl.load(flat, [t0, k0], [16, 256], valid_shape=[pl.min(16, tokens - t0), 256])
            for invalid_row in pl.range(pl.min(16, tokens - t0), 16):
                for invalid_col in pl.range(256):
                    pl.write(values, [invalid_row, invalid_col], 0.0)
            values = pl.set_validshape(values, 16, 256)
            # Keep source/destination column widths equal for existing TMOV lowering.
            transposed_values = pl.tile.transpose(values, axis1=0, axis2=1)
            packed_transposed = pl.tile.full([512, 16], dtype=pl.FP32, value=0.0)
            for chunk in pl.range(32):
                input8 = pl.tile.slice(transposed_values, [8, 16], [chunk * 8, 0])
                packed_transposed = pl.tile.assemble(packed_transposed, input8, [chunk * 16, 0])
                packed_transposed = pl.tile.assemble(packed_transposed, input8, [chunk * 16 + 8, 0])
            packed_values = pl.tile.transpose(packed_transposed, axis1=0, axis2=1)
            pl.store(pl.tile.cast(packed_values, pl.BF16), [t0, k0 * 2], packed_flat)
    for unit in pl.spmd(
        LINEAR_OK * (padded_tokens // 16), name_hint="hcpre_bf16_half_exact8", allow_early_resolve=True
    ):
        split = unit // (padded_tokens // 16)
        row = (unit % (padded_tokens // 16)) * 16
        count = pl.min(LINEAR_K_PER_SPLIT, HC_DIM - split * LINEAR_K_PER_SPLIT) // 64
        accumulator = pl.create_tensor([16, MIX_PAD], dtype=pl.FP32)
        for kb in pl.range(count):
            packed_k = split * LINEAR_K_PER_SPLIT * 2 + kb * 128
            x16 = pl.slice(packed_flat, [16, 128], [row, packed_k])
            w16 = pl.slice(packed_weight, [128, MIX_PAD], [packed_k, 0])
            accumulator = pl.matmul_acc(accumulator, x16, w16, init_cond=(kb == 0))
        partials = pl.assemble(partials, accumulator, [split * padded_tokens + row, 0])
    return partials


hcpre_linear_panel = pl.jit.inline(auto_scope=False)(_hcpre_linear_panel)
hcpre_linear_test = pl.jit(_hcpre_linear_panel)
