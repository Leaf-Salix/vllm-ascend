# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""同一Indexer M8/四head条带，隔离Native scale结合顺序及RoPE前BF16边界。

生产固定MODE2；MODE0原反量化、1仅合并scale保留用于单因素复现。
不改RoPE公式、工作划分、尾行有效域或外部发布/等待。
"""

import pypto.language as pl

from .indexer_query_stages import (
    DEQUANT_T_TILE,
    DQ_ROPE_H_TILE,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    IDX_NOPE_HEAD_DIM,
    PROJ_ROWS_DYN,
    ROPE_HEAD_DIM,
    ROPE_ROWS_DYN,
    T_DYN,
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
