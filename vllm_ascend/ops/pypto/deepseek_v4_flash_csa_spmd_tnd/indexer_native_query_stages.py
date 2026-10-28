# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""对齐Native Indexer query的投影、RoPE及Hadamard量化舍入边界。

生产固定MODE2；MODE0原反量化、1仅合并scale保留用于单因素复现。
普通query量化恢复Hadamard线性及缩放后的BF16边界；融合系数路径尚未接入。
不改RoPE公式、工作划分、尾行有效域或外部发布/等待。
"""

import pypto.language as pl

from .indexer_query_stages import (
    DEQUANT_T_TILE,
    DQ_ROPE_H_TILE,
    HADAMARD_ROWS_DYN,
    HADAMARD_SCALE,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    IDX_NOPE_HEAD_DIM,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    MM_ROW_TILE,
    PROJ_ROWS_DYN,
    QH_HEAD_DIM_TILE,
    QH_QUANT_TILE,
    QUANT_ROWS_DYN,
    ROPE_HEAD_DIM,
    ROPE_ROWS_DYN,
    SCALE_ROWS_DYN,
    T_DYN,
    WEIGHTS_OK,
    WEIGHTS_PARTIAL_ROWS_DYN,
    WEIGHTS_ROWS_DYN,
    WEIGHTS_SCALE,
)


@pl.jit.inline(auto_scope=False)
def native_indexer_query_dequant_rope_stage(
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    qr_acc_pad: pl.Tensor[[PROJ_ROWS_DYN, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT32],
    qr_bf16_2d: pl.Tensor[[ROPE_ROWS_DYN, IDX_N_HEADS * IDX_HEAD_DIM], pl.BF16],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    MODE: pl.constexpr,
):
    """AIV：等query INT32投影与QR scale，原反量化/RoPE后写BF16，外层发布供Hadamard消费。"""
    bs = pl.tensor.dim(qr_scale, 0)
    dq_rope_units = ((bs + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE) * (IDX_N_HEADS // DQ_ROPE_H_TILE)
    dq_rope_workers = pl.min(dq_rope_units, WORKERS)
    if worker_id < dq_rope_workers:
        dq_rope_worker = worker_id
        sw_ones = pl.full([DEQUANT_T_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        sw_index = pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32)
        sw_col = pl.col_expand_mul(sw_ones, sw_index)
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        _rope_swap_idx = pl.cast(pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)), target_type=pl.INT32)
        # 每单元为8个token、4个head；核内多wave覆盖动态T，不按某个B/S冻结分配。
        for dq_unit in pl.range(dq_rope_worker, dq_rope_units, dq_rope_workers):
            hg = dq_unit % (IDX_N_HEADS // DQ_ROPE_H_TILE) * DQ_ROPE_H_TILE
            dq_t0 = dq_unit // (IDX_N_HEADS // DQ_ROPE_H_TILE) * DEQUANT_T_TILE
            if dq_t0 + DEQUANT_T_TILE <= bs:
                qr_scale_tile = pl.load(qr_scale, [dq_t0, 0], [DEQUANT_T_TILE, 1])
                cos_tile = pl.load(cos, [dq_t0, 0], [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                sin_tile = pl.load(sin, [dq_t0, 0], [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                flat_i = pl.tile.ci(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
                flat_tmp = pl.create_tile([1, ROPE_HEAD_DIM], dtype=pl.INT32)
                flat_lane = pl.tile.rems(flat_i, 2, flat_tmp)
                swap_row = pl.tile.adds(pl.tile.sub(flat_i, pl.tile.muls(flat_lane, 2)), IDX_NOPE_HEAD_DIM + 1)
                swap_base = pl.create_tile([DEQUANT_T_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
                swap_source = pl.col_expand(swap_base, swap_row)
                row_offsets = pl.tile.muls(pl.tile.ci(0, [1, DEQUANT_T_TILE], dtype=pl.INT32), IDX_HEAD_DIM)
                flat_swap = pl.reshape(
                    pl.row_expand_add(swap_source, pl.reshape(row_offsets, [DEQUANT_T_TILE, 1])),
                    [1, DEQUANT_T_TILE * ROPE_HEAD_DIM],
                )
                gather_tmp = pl.create_tile([1, DEQUANT_T_TILE * ROPE_HEAD_DIM], dtype=pl.INT32)
                for h_inner in pl.pipeline(DQ_ROPE_H_TILE, stage=2):
                    h0 = (hg + h_inner) * IDX_HEAD_DIM
                    wq_scale = pl.reshape(pl.load(wq_b_scale, [h0], [IDX_HEAD_DIM]), [1, IDX_HEAD_DIM])
                    acc_fp32 = pl.cast(
                        pl.load(qr_acc_pad, [dq_t0, h0], [DEQUANT_T_TILE, IDX_HEAD_DIM]),
                        target_type=pl.FP32,
                        mode="none",
                    )
                    if MODE >= 1:
                        scales = pl.col_expand_mul(
                            pl.row_expand_mul(
                                pl.tile.full([DEQUANT_T_TILE, IDX_HEAD_DIM], dtype=pl.FP32, value=1.0), qr_scale_tile
                            ),
                            wq_scale,
                        )
                        qr_dequant = pl.mul(acc_fp32, scales)
                    else:
                        qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, qr_scale_tile), wq_scale)
                    if MODE >= 2:
                        qr_rope_input = pl.cast(
                            pl.cast(qr_dequant, target_type=pl.BF16, mode="rint"), target_type=pl.FP32
                        )
                    else:
                        qr_rope_input = qr_dequant
                    qr_nope_bf16 = pl.cast(qr_rope_input[:, 0:IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode="rint")
                    qr_rope_slice = qr_rope_input[:, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
                    qr_swapped_flat = pl.tile.gather(
                        pl.reshape(qr_rope_input, [1, DEQUANT_T_TILE * IDX_HEAD_DIM]), flat_swap, gather_tmp
                    )
                    qr_swapped = pl.reshape(qr_swapped_flat, [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                    rope_rot = pl.add(pl.mul(qr_rope_slice, cos_tile), pl.mul(qr_swapped, sin_tile))
                    rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
                    pl.store(qr_nope_bf16, [dq_t0, h0], qr_bf16_2d)
                    pl.store(rope_bf16, [dq_t0, h0 + IDX_NOPE_HEAD_DIM], qr_bf16_2d)
            else:
                # GM只读取实际行；复用完整M8逐元素数学，最后仅写有效行。
                valid_tail_rows = bs - dq_t0
                tail_tile_qr_scale_tile = pl.load(
                    qr_scale, [dq_t0, 0], [DEQUANT_T_TILE, 1], valid_shape=[valid_tail_rows, 1]
                )
                tail_tile_cos_tile = pl.load(
                    cos, [dq_t0, 0], [DEQUANT_T_TILE, ROPE_HEAD_DIM], valid_shape=[valid_tail_rows, ROPE_HEAD_DIM]
                )
                tail_tile_sin_tile = pl.load(
                    sin, [dq_t0, 0], [DEQUANT_T_TILE, ROPE_HEAD_DIM], valid_shape=[valid_tail_rows, ROPE_HEAD_DIM]
                )
                tail_tile_qr_scale_tile = pl.set_validshape(tail_tile_qr_scale_tile, DEQUANT_T_TILE, 1)
                tail_tile_cos_tile = pl.set_validshape(tail_tile_cos_tile, DEQUANT_T_TILE, ROPE_HEAD_DIM)
                tail_tile_sin_tile = pl.set_validshape(tail_tile_sin_tile, DEQUANT_T_TILE, ROPE_HEAD_DIM)
                tail_tile_flat_i = pl.tile.ci(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
                tail_tile_flat_tmp = pl.create_tile([1, ROPE_HEAD_DIM], dtype=pl.INT32)
                tail_tile_flat_lane = pl.tile.rems(tail_tile_flat_i, 2, tail_tile_flat_tmp)
                tail_tile_swap_row = pl.tile.adds(
                    pl.tile.sub(tail_tile_flat_i, pl.tile.muls(tail_tile_flat_lane, 2)), IDX_NOPE_HEAD_DIM + 1
                )
                tail_tile_swap_base = pl.create_tile([DEQUANT_T_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
                tail_tile_swap_source = pl.col_expand(tail_tile_swap_base, tail_tile_swap_row)
                tail_tile_row_offsets = pl.tile.muls(pl.tile.ci(0, [1, DEQUANT_T_TILE], dtype=pl.INT32), IDX_HEAD_DIM)
                tail_tile_flat_swap = pl.reshape(
                    pl.row_expand_add(tail_tile_swap_source, pl.reshape(tail_tile_row_offsets, [DEQUANT_T_TILE, 1])),
                    [1, DEQUANT_T_TILE * ROPE_HEAD_DIM],
                )
                tail_tile_gather_tmp = pl.create_tile([1, DEQUANT_T_TILE * ROPE_HEAD_DIM], dtype=pl.INT32)
                for tail_tile_h_inner in pl.pipeline(DQ_ROPE_H_TILE, stage=2):
                    tail_tile_h0 = (hg + tail_tile_h_inner) * IDX_HEAD_DIM
                    tail_tile_wq_scale = pl.reshape(
                        pl.load(wq_b_scale, [tail_tile_h0], [IDX_HEAD_DIM]), [1, IDX_HEAD_DIM]
                    )
                    tail_tile_acc_fp32 = pl.cast(
                        pl.load(
                            qr_acc_pad,
                            [dq_t0, tail_tile_h0],
                            [DEQUANT_T_TILE, IDX_HEAD_DIM],
                            valid_shape=[valid_tail_rows, IDX_HEAD_DIM],
                        ),
                        target_type=pl.FP32,
                        mode="none",
                    )
                    tail_tile_acc_fp32 = pl.set_validshape(tail_tile_acc_fp32, DEQUANT_T_TILE, IDX_HEAD_DIM)
                    if MODE >= 1:
                        tail_scales = pl.col_expand_mul(
                            pl.row_expand_mul(
                                pl.tile.full([DEQUANT_T_TILE, IDX_HEAD_DIM], dtype=pl.FP32, value=1.0),
                                tail_tile_qr_scale_tile,
                            ),
                            tail_tile_wq_scale,
                        )
                        tail_tile_qr_dequant = pl.mul(tail_tile_acc_fp32, tail_scales)
                    else:
                        tail_tile_qr_dequant = pl.col_expand_mul(
                            pl.row_expand_mul(tail_tile_acc_fp32, tail_tile_qr_scale_tile), tail_tile_wq_scale
                        )
                    if MODE >= 2:
                        tail_rope_input = pl.cast(
                            pl.cast(tail_tile_qr_dequant, target_type=pl.BF16, mode="rint"), target_type=pl.FP32
                        )
                    else:
                        tail_rope_input = tail_tile_qr_dequant
                    tail_tile_qr_nope_bf16 = pl.cast(
                        tail_rope_input[:, 0:IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode="rint"
                    )
                    tail_tile_qr_rope_slice = tail_rope_input[:, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
                    tail_tile_qr_swapped_flat = pl.tile.gather(
                        pl.reshape(tail_rope_input, [1, DEQUANT_T_TILE * IDX_HEAD_DIM]),
                        tail_tile_flat_swap,
                        tail_tile_gather_tmp,
                    )
                    tail_tile_qr_swapped = pl.reshape(tail_tile_qr_swapped_flat, [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                    tail_tile_rope_rot = pl.add(
                        pl.mul(tail_tile_qr_rope_slice, tail_tile_cos_tile),
                        pl.mul(tail_tile_qr_swapped, tail_tile_sin_tile),
                    )
                    tail_tile_rope_bf16 = pl.cast(tail_tile_rope_rot, target_type=pl.BF16, mode="rint")
                    tail_tile_qr_nope_bf16 = pl.set_validshape(
                        tail_tile_qr_nope_bf16, valid_tail_rows, IDX_NOPE_HEAD_DIM
                    )
                    tail_tile_rope_bf16 = pl.set_validshape(tail_tile_rope_bf16, valid_tail_rows, ROPE_HEAD_DIM)
                    pl.store(tail_tile_qr_nope_bf16, [dq_t0, tail_tile_h0], qr_bf16_2d)
                    pl.store(tail_tile_rope_bf16, [dq_t0, tail_tile_h0 + IDX_NOPE_HEAD_DIM], qr_bf16_2d)
    return qr_bf16_2d


@pl.jit.inline(auto_scope=False)
def native_indexer_query_quant_stage(
    qh_acc_gm: pl.Tensor[[HADAMARD_ROWS_DYN, IDX_HEAD_DIM], pl.FP32],
    qr_hadamard_i8: pl.Tensor[[QUANT_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    qr_hadamard_scale_dq: pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV：先等Hadamard FP32，逐64head行归一比例并量化；输出INT8与独立FP32 scale。"""
    bs_heads = tokens * IDX_N_HEADS
    qh_quant_worker = worker_id
    for idx in pl.range(qh_quant_worker, bs_heads // QH_QUANT_TILE, WORKERS):
        o0 = idx * QH_QUANT_TILE
        qh_full_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, 0:IDX_HEAD_DIM]
        # Native先输出BF16 Hadamard线性结果，再单独做BF16缩放。
        qh_full_f32 = pl.cast(pl.cast(qh_full_f32, pl.BF16, mode="rint"), pl.FP32)
        qh_full_f32 = pl.cast(pl.cast(pl.mul(qh_full_f32, HADAMARD_SCALE), pl.BF16, mode="rint"), pl.FP32)
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


@pl.jit.inline(auto_scope=False)
def native_indexer_weights_reduce_stage(
    weights_partial: pl.Tensor[[WEIGHTS_PARTIAL_ROWS_DYN, IDX_N_HEADS], pl.FP32],
    weights: pl.Tensor[[WEIGHTS_ROWS_DYN, IDX_N_HEADS], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV：等当前M16的四个投影split，原左折叠后乘模型scale，保持FP32给score消费。"""
    row_blocks = (tokens + MM_ROW_TILE - 1) // MM_ROW_TILE
    t_pad = row_blocks * MM_ROW_TILE
    for w_rb in pl.range(worker_id, row_blocks, WORKERS):
        w_r0 = w_rb * MM_ROW_TILE
        w_sum = weights_partial[w_r0 : w_r0 + MM_ROW_TILE, :]
        # 与原核相同的((p0+p1)+p2)+p3；不把四项改成树形归约。
        for kb in pl.unroll(1, WEIGHTS_OK):
            partial_r0 = kb * t_pad + w_r0
            w_sum = pl.add(w_sum, weights_partial[partial_r0 : partial_r0 + MM_ROW_TILE, :])
        # Native线性层与随后乘系数各自输出BF16；保留原四split左折叠。
        linear = pl.cast(pl.cast(w_sum, pl.BF16, mode="rint"), pl.FP32)
        scaled = pl.cast(pl.mul(linear, WEIGHTS_SCALE), pl.BF16, mode="rint")
        weights[w_r0 : w_r0 + MM_ROW_TILE, :] = pl.cast(scaled, pl.FP32)
    return weights
