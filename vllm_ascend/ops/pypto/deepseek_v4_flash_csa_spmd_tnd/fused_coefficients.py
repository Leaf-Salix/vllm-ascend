# SPDX-License-Identifier: Apache-2.0
"""META-owned zero initialization and token-owned diagonal coefficient writes.

The offset allocation has T additional INT32 words at its end. META initializes
them for this invocation before publishing; quantization only reads this tail.
All coefficient rows are zeroed before any token owner writes its 64-column
diagonal segment. Quantization publication covers that segment as well as its
original INT8/FP32 outputs. No token can clear another token's segment.
"""

import pypto.language as pl

from .indexer_group_stages import (
    COEFF_COLS_DYN,
    COEFF_ROWS_DYN,
    COEFFICIENT_ROWS,
    HEADS,
    OFFSET_ROWS_DYN,
    TOKENS_DYN,
    WEIGHT_ROWS_DYN,
    indexer_decode_group,
    indexer_query_visibility,
)
from .indexer_query_stages import (
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
)
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN

assert HEADS == IDX_N_HEADS == QH_QUANT_TILE == 64


@pl.jit.inline(auto_scope=False)
def prepare_fused_query_coefficients(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    offsets: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS_DYN, COEFF_COLS_DYN], pl.FP16]],
    tokens: pl.Scalar[pl.INDEX],
    GROUP_SIZE: pl.constexpr,
):
    """Only the existing META owner calls this, after the original group plan."""
    map_base = pl.tensor.dim(offsets, 0) - tokens
    for query in pl.range(tokens):
        pl.write(offsets, [map_base + query], pl.cast(-1, pl.INT32))
    requests = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
    groups = pl.cast(pl.read(offsets, [requests]), pl.INDEX)
    zero_rows = pl.tile.full([COEFFICIENT_ROWS, GROUP_SIZE * HEADS], dtype=pl.FP16, value=0.0)
    for group in pl.range(groups):
        (_request, first, live, _visible) = indexer_decode_group(bounds, lengths, positions, offsets, group, GROUP_SIZE)
        coefficients = pl.store(zero_rows, [group * COEFFICIENT_ROWS, 0], coefficients)
        for lane in pl.range(live):
            query = first + lane
            if query >= 0 and query < tokens:
                if pl.read(offsets, [map_base + query]) == -1:
                    pl.write(offsets, [map_base + query], pl.cast(group * GROUP_SIZE + lane, pl.INT32))
                else:
                    pl.write(offsets, [map_base + query], pl.cast(-2, pl.INT32))
    return (offsets, coefficients)


@pl.jit.inline(auto_scope=False)
def indexer_query_quant_coeff_stage(
    qh_acc_gm: pl.Tensor[[HADAMARD_ROWS_DYN, IDX_HEAD_DIM], pl.FP32],
    qr_hadamard_i8: pl.Tensor[[QUANT_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    qr_hadamard_scale_dq: pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    weights: pl.Tensor[[WEIGHT_ROWS_DYN, HEADS], pl.FP32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS_DYN, COEFF_COLS_DYN], pl.FP16]],
    GROUP_SIZE: pl.constexpr,
):
    """AIV：先等Hadamard FP32，逐64head行归一比例并量化；输出INT8与独立FP32 scale。"""
    bs_heads = tokens * IDX_N_HEADS
    qh_quant_worker = worker_id
    for idx in pl.range(qh_quant_worker, bs_heads // QH_QUANT_TILE, WORKERS):
        o0 = idx * QH_QUANT_TILE
        qh_full_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, 0:IDX_HEAD_DIM]
        qh_full_f32 = pl.mul(qh_full_f32, HADAMARD_SCALE)
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
        map_base = pl.tensor.dim(offsets, 0) - tokens
        encoded = pl.cast(pl.read(offsets, [map_base + idx]), pl.INDEX)
        if encoded >= 0:
            group = encoded // GROUP_SIZE
            lane = encoded % GROUP_SIZE
            (request, first, live, _visible) = indexer_decode_group(
                bounds, lengths, positions, offsets, group, GROUP_SIZE
            )
            if lane < live and idx == first + lane:
                (_visible, causal) = indexer_query_visibility(bounds, lengths, positions, request, idx)
                if causal != 0:
                    query_weights = weights[idx : idx + 1, 0:HEADS]
                    query_scale_half = pl.cast(qh_scale_recip, pl.FP16, mode="rint")
                    query_weight_half = pl.cast(query_weights, pl.FP16, mode="rint")
                    head_coefficients = pl.mul(query_scale_half, query_weight_half)
                    coefficients = pl.assemble(
                        coefficients, head_coefficients, [group * COEFFICIENT_ROWS + lane, lane * HEADS]
                    )
        if encoded == -2:
            requests = pl.tensor.dim(lengths, 0)
            groups = pl.cast(pl.read(offsets, [requests]), pl.INDEX)
            for overlap_group in pl.range(groups):
                (overlap_request, overlap_first, overlap_live, _overlap_visible) = indexer_decode_group(
                    bounds, lengths, positions, offsets, overlap_group, GROUP_SIZE
                )
                if idx >= overlap_first and idx < overlap_first + overlap_live:
                    overlap_lane = idx - overlap_first
                    (_overlap_visible, overlap_causal) = indexer_query_visibility(
                        bounds, lengths, positions, overlap_request, idx
                    )
                    if overlap_causal != 0:
                        overlap_weights = weights[idx : idx + 1, 0:HEADS]
                        overlap_scale_half = pl.cast(qh_scale_recip, pl.FP16, mode="rint")
                        overlap_weight_half = pl.cast(overlap_weights, pl.FP16, mode="rint")
                        overlap_coefficients = pl.mul(overlap_scale_half, overlap_weight_half)
                        coefficients = pl.assemble(
                            coefficients,
                            overlap_coefficients,
                            [overlap_group * COEFFICIENT_ROWS + overlap_lane, overlap_lane * HEADS],
                        )
    return (qr_hadamard_i8, qr_hadamard_scale_dq, coefficients)
