# SPDX-License-Identifier: Apache-2.0
"""Native WO-A BF16 → 整行量化 → INT32 WO-B → 通道/行 scale → BF16。"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision.nz_mode import (
    QUANT_WEIGHT_LAYOUT,
    WO_A_WEIGHT_LAYOUT,
)

T = pl.dynamic("FULL_PRECISION_OPROJ_T")


@pl.jit
def o_proj_test(
    heads: pl.Tensor[[T, 32768], pl.BF16],
    wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[8192, 4096], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[4096], pl.FP32],
    start: pl.Tensor[[16, 8, 8], pl.INT32],
    projected: pl.Out[pl.Tensor[[T, 8192], pl.BF16]],
    quantized: pl.Out[pl.Tensor[[T, 8192], pl.INT8]],
    token_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
    output: pl.Out[pl.Tensor[[T, 4096], pl.BF16]],
):
    heads.bind_dynamic(0, T)
    projected.bind_dynamic(0, T)
    quantized.bind_dynamic(0, T)
    token_scale.bind_dynamic(0, T)
    output.bind_dynamic(0, T)
    rows = pl.tensor.dim(heads, 0)
    row_tiles = (rows + 15) // 16
    accum = pl.create_tensor([row_tiles * 16, 8192], dtype=pl.FP32)
    for group in pl.parallel(8):
        for block in pl.spmd(8, name_hint="oproj_a"):
            col = block * 128
            for rb in pl.range(row_tiles):
                row = rb * 16
                valid = pl.min(16, rows - row)
                first_block = pl.max(pl.read(start, [rb, group, block]), 0)
                first_k = first_block * 16
                x = pl.slice(heads, [16, 16], [row, group * 4096 + first_k], valid_shape=[valid, 16])
                w = wo_a[group : group + 1, first_k : first_k + 16, col : col + 128]
                acc = pl.matmul(x, w, out_dtype=pl.FP32)
                for step in pl.range(1, 256):
                    k = ((first_block + step) % 256) * 16
                    x = pl.slice(heads, [16, 16], [row, group * 4096 + k], valid_shape=[valid, 16])
                    w = wo_a[group : group + 1, k : k + 16, col : col + 128]
                    acc = pl.matmul_acc(acc, x, w)
                accum = pl.assemble(accum, acc, [row, group * 1024 + col])
    for block in pl.spmd(row_tiles * 8, name_hint="oproj_a_bf16"):
        row = block // 8 * 16
        col = block % 8 * 1024
        valid = pl.min(16, rows - row)
        oa_value = pl.load(accum, [row, col], [16, 1024], valid_shape=[valid, 1024])
        pl.store(pl.cast(oa_value, pl.BF16, mode="rint"), [row, col], projected)

    quant_rows = (rows + 7) // 8
    for block in pl.spmd(quant_rows, name_hint="oproj_quant"):
        row = block * 8
        valid = pl.min(8, rows - row)
        first_quant = pl.load(projected, [row, 0], [8, 1024], valid_shape=[valid, 1024])
        first_abs = pl.abs(pl.cast(first_quant, pl.FP32))
        amax = pl.row_max(first_abs, pl.create_tile([8, 1024], dtype=pl.FP32))
        for group in pl.range(1, 8):
            q_bf16 = pl.load(projected, [row, group * 1024], [8, 1024], valid_shape=[valid, 1024])
            q_fp32 = pl.cast(q_bf16, pl.FP32)
            amax = pl.maximum(amax, pl.row_max(pl.abs(q_fp32), pl.create_tile([8, 1024], dtype=pl.FP32)))
        dequant = pl.mul(amax, 1.0 / 127.0)
        denominator = pl.add(amax, 0.0)
        for lane in pl.range(valid):
            magnitude = pl.read(denominator, [lane, 0])
            if magnitude == 0.0:
                pl.write(denominator, [lane, 0], 1.0)
        multiplier = pl.div(pl.add(pl.mul(amax, 0.0), 127.0), denominator)
        pl.store(pl.set_validshape(dequant, valid, 1), [row, 0], token_scale)
        for group in pl.range(8):
            col = group * 1024
            quant_bf16 = pl.load(projected, [row, col], [8, 1024], valid_shape=[valid, 1024])
            quant_scaled = pl.row_expand_mul(pl.cast(quant_bf16, pl.FP32), multiplier)
            integer = pl.cast(quant_scaled, pl.INT32, mode="rint")
            half = pl.cast(integer, pl.FP16, mode="round")
            result = pl.cast(half, pl.INT8, mode="trunc")
            pl.store(pl.set_validshape(result, valid, 1024), [row, col], quantized)

    integer_accum = pl.create_tensor([row_tiles * 16, 4096], dtype=pl.INT32)
    for block in pl.spmd(row_tiles * 32, name_hint="oproj_b"):
        row = block // 32 * 16
        col = pl.max(block % 32, 0) * 128
        valid = pl.min(16, rows - row)
        b_acc = pl.create_tensor([16, 128], dtype=pl.INT32)
        for step in pl.range(16):
            k = step * 512
            b_input = pl.slice(quantized, [16, 512], [row, k], valid_shape=[valid, 512])
            b_weight = wo_b[k : k + 512, col : col + 128]
            b_acc = pl.matmul_acc(b_acc, b_input, b_weight, init_cond=(step == 0))
        integer_accum[row : row + 16, col : col + 128] = b_acc
    for block in pl.spmd(row_tiles * 8, name_hint="oproj_b_dequant"):
        row = block // 8 * 16
        col = block % 8 * 512
        valid = pl.min(16, rows - row)
        ob_integer = pl.load(integer_accum, [row, col], [16, 512], valid_shape=[valid, 512])
        channel = pl.reshape(pl.load(wo_b_scale, [col], [512]), [1, 512])
        scale = pl.load(token_scale, [row, 0], [16, 1], valid_shape=[valid, 1])
        coefficient_one = pl.tile.full([16, 512], dtype=pl.FP32, value=1.0)
        coefficient_channel = pl.col_expand_mul(coefficient_one, channel)
        coefficient = pl.row_expand_mul(coefficient_channel, scale)
        ob_scaled = pl.mul(pl.cast(ob_integer, pl.FP32), coefficient)
        pl.store(pl.cast(ob_scaled, pl.BF16, mode="rint"), [row, col], output)
    return output
