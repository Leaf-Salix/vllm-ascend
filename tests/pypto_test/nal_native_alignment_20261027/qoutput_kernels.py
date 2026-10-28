# SPDX-License-Identifier: Apache-2.0
"""同一完整SPMD accumulator，隔离Q反量化及Norm/RoPE数值边界。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.q_native_output_stages import q_native_dequant_rope_stage
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.query_projection_stages import (
    PAD_ROWS_DYN,
    QPROJ_ROWS_DYN,
    T_DYN,
    q_dequant_rope_stage,
)


@pl.jit
def qoutput_arm(
    accumulator: pl.Tensor[[QPROJ_ROWS_DYN, 32768], pl.INT32],
    weight_scale: pl.Tensor[[32768], pl.FP32],
    row_scale: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    cos: pl.Tensor[[T_DYN, 64], pl.FP32],
    signed_sin: pl.Tensor[[T_DYN, 64], pl.FP32],
    swap: pl.Tensor[[T_DYN, 64], pl.INT32],
    output: pl.InOut[pl.Tensor[[T_DYN, 32768], pl.BF16]],
    MODE: pl.constexpr,
):
    for worker in pl.spmd(48, name_hint="q_output_boundary", allow_early_resolve=True):
        if MODE == 0:
            output = q_dequant_rope_stage(
                weight_scale, cos, signed_sin, swap, output, row_scale, accumulator, worker, 48
            )
        else:
            output = q_native_dequant_rope_stage(
                weight_scale, cos, signed_sin, swap, output, row_scale, accumulator, worker, 48, MODE
            )
    return output


@pl.jit
def qdequant_arm(
    accumulator: pl.Tensor[[QPROJ_ROWS_DYN, 32768], pl.INT32],
    weight_scale: pl.Tensor[[32768], pl.FP32],
    row_scale: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    output: pl.InOut[pl.Tensor[[T_DYN, 32768], pl.BF16]],
    COMBINED: pl.constexpr,
):
    tokens = pl.tensor.dim(output, 0)
    for worker in pl.spmd(48, name_hint="q_dequant_boundary", allow_early_resolve=True):
        for job in pl.range(worker, (tokens + 7) // 8 * 64, 48):
            row = job // 64 * 8
            column = job % 64 * 512
            values = pl.cast(pl.load(accumulator, [row, column], [8, 512]), pl.FP32)
            act = pl.load(row_scale, [row, 0], [8, 1])
            weight = pl.reshape(pl.load(weight_scale, [column], [512]), [1, 512])
            if COMBINED:
                scales = pl.col_expand_mul(
                    pl.row_expand_mul(pl.tile.full([8, 512], dtype=pl.FP32, value=1.0), act), weight
                )
                dequantized = pl.mul(values, scales)
            else:
                dequantized = pl.col_expand_mul(pl.row_expand_mul(values, act), weight)
            rounded = pl.cast(dequantized, pl.BF16, mode="rint")
            pl.store(pl.set_validshape(rounded, pl.min(8, tokens - row), 512), [row, column], output)
    return output
