# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""只隔离Native Hadamard BF16和scale BF16边界；不改变原量化表达式。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.indexer_query_stages import (
    HADAMARD_ROWS_DYN,
    HADAMARD_SCALE,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    QH_HEAD_DIM_TILE,
    QH_QUANT_TILE,
    QUANT_ROWS_DYN,
    SCALE_ROWS_DYN,
    indexer_query_quant_stage,
)


@pl.jit.inline(auto_scope=False)
def quant_boundary_stage(
    qh_acc_gm: pl.Tensor[[HADAMARD_ROWS_DYN, IDX_HEAD_DIM], pl.FP32],
    qr_hadamard_i8: pl.Tensor[[QUANT_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    qr_hadamard_scale_dq: pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    MODE: pl.constexpr,
):
    """AIV：先等Hadamard FP32，逐64head行归一比例并量化；输出INT8与独立FP32 scale。"""
    bs_heads = tokens * IDX_N_HEADS
    qh_quant_worker = worker_id
    for idx in pl.range(qh_quant_worker, bs_heads // QH_QUANT_TILE, WORKERS):
        o0 = idx * QH_QUANT_TILE
        raw = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, 0:IDX_HEAD_DIM]
        if MODE >= 1:
            linear = pl.cast(pl.cast(raw, pl.BF16, mode="rint"), pl.FP32)
        else:
            linear = raw
        scaled = pl.mul(linear, HADAMARD_SCALE)
        if MODE >= 2:
            qh_full_f32 = pl.cast(pl.cast(scaled, pl.BF16, mode="rint"), pl.FP32)
        else:
            qh_full_f32 = scaled
        qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_full_f32[:, h0 : h0 + QH_HEAD_DIM_TILE]
            qh_a_neg = pl.neg(qh_a_f32)
            qh_a_abs = pl.maximum(qh_a_f32, qh_a_neg)
            qh_a_max_col = pl.row_max(qh_a_abs)
            qh_a_max = pl.reshape(qh_a_max_col, [1, QH_QUANT_TILE])
            qh_amax = pl.maximum(qh_amax, qh_a_max)
        qh_scale_numerator = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        qh_scale_quant_row = pl.div(qh_scale_numerator, qh_amax)
        qh_scale_recip = pl.recip(qh_scale_quant_row)
        qh_scale_dq = pl.reshape(qh_scale_recip, [QH_QUANT_TILE, 1])
        qr_hadamard_scale_dq[o0 : o0 + QH_QUANT_TILE, :] = qh_scale_dq
        qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
        qh_q_scaled = pl.row_expand_mul(qh_full_f32, qh_scale_quant)
        qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
        qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
        qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
        qr_hadamard_i8[o0 : o0 + QH_QUANT_TILE, 0:IDX_HEAD_DIM] = qh_i8
    return qr_hadamard_i8, qr_hadamard_scale_dq


@pl.jit
def idxquant_arm(
    accumulator: pl.Tensor[[HADAMARD_ROWS_DYN, 128], pl.FP32],
    query: pl.InOut[pl.Tensor[[QUANT_ROWS_DYN, 128], pl.INT8]],
    scale: pl.InOut[pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32]],
    MODE: pl.constexpr,
):
    tokens = pl.tensor.dim(accumulator, 0) // 64
    for worker in pl.spmd(48, name_hint="query_quant_boundaries", allow_early_resolve=True):
        if MODE == 0:
            query, scale = indexer_query_quant_stage(accumulator, query, scale, tokens, worker, 48)
        else:
            query, scale = quant_boundary_stage(accumulator, query, scale, tokens, worker, 48, MODE)
    return query, scale
