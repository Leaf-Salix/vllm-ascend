# SPDX-License-Identifier: Apache-2.0
"""保存的同一INT32 accumulator，仅对拍Indexer反量化/RoPE边界。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.indexer_native_query_stages import (
    native_indexer_query_dequant_rope_stage,
)
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.indexer_query_stages import (
    PROJ_ROWS_DYN,
    ROPE_ROWS_DYN,
    T_DYN,
    indexer_query_dequant_rope_stage,
)


@pl.jit
def idxquery_rope_arm(
    accumulator: pl.Tensor[[PROJ_ROWS_DYN, 8192], pl.INT32],
    row_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    weight_scale: pl.Tensor[[8192], pl.FP32],
    cos: pl.Tensor[[T_DYN, 64], pl.FP32],
    signed_sin: pl.Tensor[[T_DYN, 64], pl.FP32],
    output: pl.InOut[pl.Tensor[[ROPE_ROWS_DYN, 8192], pl.BF16]],
    MODE: pl.constexpr,
):
    for worker in pl.spmd(48, name_hint="indexer_query_boundary", allow_early_resolve=True):
        if MODE == 0:
            output = indexer_query_dequant_rope_stage(
                row_scale, weight_scale, cos, signed_sin, accumulator, output, worker, 48
            )
        else:
            output = native_indexer_query_dequant_rope_stage(
                row_scale, weight_scale, cos, signed_sin, accumulator, output, worker, 48, MODE
            )
    return output


@pl.jit
def idxquery_dequant_arm(
    accumulator: pl.Tensor[[PROJ_ROWS_DYN, 8192], pl.INT32],
    row_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    weight_scale: pl.Tensor[[8192], pl.FP32],
    output: pl.InOut[pl.Tensor[[T_DYN, 8192], pl.BF16]],
    COMBINED: pl.constexpr,
):
    tokens = pl.tensor.dim(output, 0)
    for worker in pl.spmd(48, name_hint="indexer_dequant_order", allow_early_resolve=True):
        for job in pl.range(worker, (tokens + 7) // 8 * 64, 48):
            row = job // 64 * 8
            column = job % 64 * 128
            acc = pl.cast(pl.load(accumulator, [row, column], [8, 128]), pl.FP32)
            act = pl.load(row_scale, [row, 0], [8, 1])
            weight = pl.reshape(pl.load(weight_scale, [column], [128]), [1, 128])
            if COMBINED:
                scales = pl.col_expand_mul(
                    pl.row_expand_mul(pl.tile.full([8, 128], dtype=pl.FP32, value=1.0), act), weight
                )
                value = pl.mul(acc, scales)
            else:
                value = pl.col_expand_mul(pl.row_expand_mul(acc, act), weight)
            rounded = pl.cast(value, pl.BF16, mode="rint")
            pl.store(pl.set_validshape(rounded, pl.min(8, tokens - row), 128), [row, column], output)
    return output
