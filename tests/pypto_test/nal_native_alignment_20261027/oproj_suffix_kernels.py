# SPDX-License-Identifier: Apache-2.0
"""仅T24：Native整行量化和WO-B整数归并的独立纯DSL诊断。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.output_projection_a_stages import MM_ROWS_DYN, SCALE_ROWS_DYN
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.output_projection_b_stages import o_b_stage


@pl.jit
def output_quant_arm(
    value: pl.Tensor[[24, 8192], pl.BF16],
    quantized: pl.InOut[pl.Tensor[[MM_ROWS_DYN, 8192], pl.INT8]],
    scales: pl.InOut[pl.Tensor[[8, SCALE_ROWS_DYN], pl.FP32]],
):
    for worker in pl.spmd(24, name_hint="native_output_quant", allow_early_resolve=True):
        row = worker * 8
        if row < 24:
            first = pl.cast(pl.load(value, [row, 0], [8, 1024]), pl.FP32)
            amax = pl.row_max(pl.abs(first), pl.create_tile([8, 1024], dtype=pl.FP32))
            for group in pl.range(1, 8):
                chunk = pl.cast(pl.load(value, [row, group * 1024], [8, 1024]), pl.FP32)
                amax = pl.maximum(amax, pl.row_max(pl.abs(chunk), pl.create_tile([8, 1024], dtype=pl.FP32)))
            dequant = pl.mul(amax, 1.0 / 127.0)
            denominator = pl.add(amax, 0.0)
            for lane in pl.range(8):
                if pl.read(denominator, [lane, 0]) == 0.0:
                    pl.write(denominator, [lane, 0], 1.0)
            multiplier = pl.div(pl.add(pl.mul(amax, 0.0), 127.0), denominator)
            for group in pl.range(8):
                source = pl.cast(pl.load(value, [row, group * 1024], [8, 1024]), pl.FP32)
                scaled = pl.row_expand_mul(source, multiplier)
                integer = pl.cast(scaled, pl.INT32, mode="rint")
                half = pl.cast(integer, pl.FP16, mode="round")
                result = pl.cast(half, pl.INT8, mode="trunc")
                pl.store(result, [row, group * 1024], quantized)
                pl.store(pl.reshape(dequant, [1, 8]), [group, row], scales)
        padded = pl.tensor.dim(quantized, 0)
        for tail in pl.range(worker, (padded - 24 + 7) // 8, 24):
            tail_row = 24 + tail * 8
            zero = pl.cast(pl.tile.full([8, 1024], dtype=pl.FP16, value=0.0), pl.INT8, mode="trunc")
            for group in pl.range(8):
                pl.store(
                    pl.set_validshape(zero, pl.min(8, padded - tail_row), 1024), [tail_row, group * 1024], quantized
                )
    return quantized, scales


@pl.jit
def output_b_arm(
    quantized: pl.Tensor[[MM_ROWS_DYN, 8192], pl.INT8],
    scales: pl.Tensor[[8, SCALE_ROWS_DYN], pl.FP32],
    weight: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ],
    weight_scale: pl.Tensor[[4096], pl.FP32],
    partials: pl.InOut[pl.Tensor[[MM_ROWS_DYN, 32768], pl.INT32]],
    output: pl.InOut[pl.Tensor[[24, 4096], pl.BF16]],
):
    for worker in pl.spmd(24, name_hint="native_output_b", allow_early_resolve=True):
        partials = o_b_stage(quantized, weight, partials, 24, worker, 24, 128)
    for row in pl.spmd(24, name_hint="native_output_dequant", allow_early_resolve=True):
        integer = pl.load(partials, [row, 0], [1, 4096])
        for group in pl.range(1, 8):
            integer = pl.add(integer, pl.load(partials, [row, group * 4096], [1, 4096]))
        token_scale = pl.read(scales, [0, row])
        channel_scale = pl.reshape(pl.load(weight_scale, [0], [4096]), [1, 4096])
        joint_scale = pl.mul(channel_scale, token_scale)
        result = pl.mul(pl.cast(integer, pl.FP32), joint_scale)
        pl.store(pl.cast(result, pl.BF16, mode="rint"), [row, 0], output)
    return partials, output
