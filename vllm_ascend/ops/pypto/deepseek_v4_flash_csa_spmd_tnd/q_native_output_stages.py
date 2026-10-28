# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Q反量化/Norm/RoPE独立数值候选，保留nalinaly原M8/head所有权。

MODE逐项增加scale合乘、投影BF16、Norm BF16、归约树和尾rsqrt，保留单因素对比。
"""

import pypto.language as pl

from .query_projection_stages import (
    EPS,
    HEAD_DIM,
    NOPE_DIM,
    PAD_ROWS_DYN,
    Q_DEQUANT_WORKERS,
    Q_ROPE_H_TILE,
    Q_ROPE_T_TILE,
    QPROJ_ROWS_DYN,
    ROPE_DIM,
    ROPE_DIM_SCALE,
    T_DYN,
    H,
)


@pl.jit.inline(auto_scope=False)
def q_native_dequant_rope_stage(
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q_flat: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    qr_scale_pad_store: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    q_proj_i32: pl.Tensor[[QPROJ_ROWS_DYN, H * HEAD_DIM], pl.INT32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    MODE: pl.constexpr,
):
    """显式 AIV 反量化、RMS、NOPE/RoPE；q_flat 必须是外层建立的 GM 二维别名。

    每次工作只写八行中的一个head组；MODE5采用Native归约及尾rsqrt。"""
    tokens = pl.tensor.dim(q_flat, 0)
    dq_token_tiles = (tokens + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE
    dq_head_tile = Q_ROPE_H_TILE
    # 只合并完整八行块的独立 head：选择最大的 64/32/16/8，使
    # dq_token_tiles*(H/head_tile) 恰为正的 48-AIV 整波；不满足则保留4。
    # 扩组后每个AIV的head迭代数不增，只减少scale/cos/sin/swap的重复准备。
    # 此条件取实际设备T，不用B/S或测试点；WORKERS仍只决定任务条带。
    # 有尾行必须留4，保持下面固定4-head尾循环及原recip(sqrt)数学。
    if tokens % Q_ROPE_T_TILE == 0:
        if dq_token_tiles >= Q_DEQUANT_WORKERS and dq_token_tiles % Q_DEQUANT_WORKERS == 0:
            dq_head_tile = H
        elif dq_token_tiles >= Q_DEQUANT_WORKERS * 32 // H and dq_token_tiles % (Q_DEQUANT_WORKERS * 32 // H) == 0:
            dq_head_tile = 32
        elif dq_token_tiles >= Q_DEQUANT_WORKERS * 16 // H and dq_token_tiles % (Q_DEQUANT_WORKERS * 16 // H) == 0:
            dq_head_tile = 16
        elif dq_token_tiles >= Q_DEQUANT_WORKERS * 8 // H and dq_token_tiles % (Q_DEQUANT_WORKERS * 8 // H) == 0:
            dq_head_tile = 8
    # 将“八行块、head组”展平成任务号，worker只处理自己的条带，避免重复写。
    for dq_work in pl.range(worker_id, (tokens + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE * (H // dq_head_tile), WORKERS):
        hg = dq_work % (H // dq_head_tile) * dq_head_tile
        tg = dq_work // (H // dq_head_tile) * Q_ROPE_T_TILE
        if tg + Q_ROPE_T_TILE <= tokens:
            qr_scale_dq_t = pl.load(qr_scale_pad_store, [tg, 0], [Q_ROPE_T_TILE, 1], target_memory=pl.MemorySpace.Vec)
            q_cos_il = pl.load(rope_cos_il, [tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec)
            q_sin_signed = pl.load(
                rope_sin_signed, [tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec
            )
            q_swap_idx = pl.load(rope_swap_idx, [tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec)
            q_flat_i = pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32), pl.FP32)
            q_row_i = pl.cast(pl.mul(q_flat_i, 1.0 / ROPE_DIM), pl.INT32, mode="trunc")
            q_row_starts = pl.mul(pl.cast(q_row_i, pl.FP32), ROPE_DIM_SCALE)
            q_abs_swap = pl.cast(
                pl.add(pl.cast(pl.reshape(q_swap_idx, [1, Q_ROPE_T_TILE * ROPE_DIM]), pl.FP32), q_row_starts), pl.INT32
            )
            q_flat_gather_tmp = pl.tile.create([1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32)
            # head之间数学独立；pipeline复用八行临时空间，逐head保持512列规约。
            for h_inner in pl.pipeline(dq_head_tile, stage=2):
                h = hg + h_inner
                h0 = h * HEAD_DIM
                q_head_acc = pl.load(q_proj_i32, [tg, h0], [Q_ROPE_T_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
                q_head_scale = pl.reshape(
                    pl.load(wq_b_scale, [h0], [HEAD_DIM], target_memory=pl.MemorySpace.Vec), [1, HEAD_DIM]
                )
                q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                # 原路径分两次乘法；MODE1起按Native先合乘scale，单独核验BF16投影边界。
                if MODE >= 1:
                    combined = pl.col_expand_mul(
                        pl.row_expand_mul(
                            pl.tile.full([Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32, value=1.0), qr_scale_dq_t
                        ),
                        q_head_scale,
                    )
                    dequantized = pl.mul(q_head_acc_fp32, combined)
                else:
                    q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                    dequantized = pl.col_expand_mul(q_head_row_scaled, q_head_scale)
                if MODE >= 2:
                    q_head_dq = pl.cast(pl.cast(dequantized, pl.BF16, mode="rint"), pl.FP32)
                else:
                    q_head_dq = dequantized
                q_head_sq = pl.mul(q_head_dq, q_head_dq)
                if MODE >= 4:
                    half = pl.add(q_head_sq[:, :256], q_head_sq[:, 256:512])
                    quarter = pl.add(half[:, :128], half[:, 128:256])
                    lanes = pl.add(quarter[:, :64], quarter[:, 64:128])
                    q_head_sq_row = pl.row_sum(lanes, tmp_tile=pl.tile.create([Q_ROPE_T_TILE, 64], dtype=pl.FP32))
                else:
                    q_head_sq_row = pl.row_sum(
                        q_head_sq, tmp_tile=pl.tile.create([Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32)
                    )
                q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
                q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                q_head_var = pl.add(q_head_sq_mean, EPS)
                q_head_inv_rms = pl.tile.rsqrt(q_head_var, pl.tile.create([1, Q_ROPE_T_TILE], dtype=pl.FP32))
                q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])
                q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
                q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                pl.store(q_nope_bf16, [tg, h0], q_flat)
                q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                rope_normalized = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                if MODE >= 3:
                    q_rope_chunk = pl.cast(pl.cast(rope_normalized, pl.BF16, mode="rint"), pl.FP32)
                else:
                    q_rope_chunk = rope_normalized
                q_rope_flat = pl.reshape(q_rope_chunk, [1, Q_ROPE_T_TILE * ROPE_DIM])
                q_swapped_flat = pl.tile.gather(q_rope_flat, q_abs_swap, q_flat_gather_tmp)
                q_rope_swapped = pl.reshape(q_swapped_flat, [Q_ROPE_T_TILE, ROPE_DIM])
                q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                pl.store(q_rope_bf16, [tg, h0 + NOPE_DIM], q_flat)
        else:
            valid_tail_rows = tokens - tg
            # 原尾块使用八行暂存计算、仅写有效行。这个分支保留 recip(sqrt)，
            # 不因完整块采用 rsqrt 就把两条数值路径强行统一。
            qr_scale_dq_tail = pl.load(
                qr_scale_pad_store, [tg, 0], [Q_ROPE_T_TILE, 1], target_memory=pl.MemorySpace.Vec
            )
            q_cos_il_tail = pl.load(
                rope_cos_il,
                [tg, 0],
                [Q_ROPE_T_TILE, ROPE_DIM],
                valid_shape=[valid_tail_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            q_sin_signed_tail = pl.load(
                rope_sin_signed,
                [tg, 0],
                [Q_ROPE_T_TILE, ROPE_DIM],
                valid_shape=[valid_tail_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            q_col = pl.col_expand_mul(
                pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
            )
            q_dup_f = pl.cast(pl.cast(pl.mul(q_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32)
            q_lane = pl.sub(q_col, pl.mul(q_dup_f, 2.0))
            q_swap_f = pl.sub(pl.add(q_col, 1.0), pl.mul(q_lane, 2.0))
            q_row_seed = pl.mul(
                pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32), target_type=pl.FP32), ROPE_DIM_SCALE
            )
            q_row_grid = pl.col_expand_mul(
                pl.tile.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.FP32, value=1.0), q_row_seed
            )
            q_row_offset = pl.transpose(q_row_grid, axis1=0, axis2=1)
            q_swap_idx_tail = pl.cast(pl.add(q_swap_f, q_row_offset), target_type=pl.INT32)
            q_head_reduce_tmp = pl.create_tile(
                [Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            q_gather_tmp = pl.create_tile([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
            # 扩组分支显式要求T整除8；有尾行必回退4。此处继续固定4，
            # 不改变原尾块的有效写回和recip(sqrt)路径。
            for h_inner_tail in pl.range(Q_ROPE_H_TILE):
                h_tail = hg + h_inner_tail
                h0_tail = h_tail * HEAD_DIM
                q_head_acc_tail = pl.load(
                    q_proj_i32, [tg, h0_tail], [Q_ROPE_T_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec
                )
                q_head_scale_input_tail = pl.load(wq_b_scale, [h0_tail], [HEAD_DIM], target_memory=pl.MemorySpace.Vec)
                q_head_scale_tail = pl.reshape(q_head_scale_input_tail, [1, HEAD_DIM])
                q_head_acc_fp32_tail = pl.cast(q_head_acc_tail, target_type=pl.FP32, mode="none")
                if MODE >= 1:
                    combined_tail = pl.col_expand_mul(
                        pl.row_expand_mul(
                            pl.tile.full([Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32, value=1.0), qr_scale_dq_tail
                        ),
                        q_head_scale_tail,
                    )
                    dequantized_tail = pl.mul(q_head_acc_fp32_tail, combined_tail)
                else:
                    q_head_row_scaled_tail = pl.row_expand_mul(q_head_acc_fp32_tail, qr_scale_dq_tail)
                    dequantized_tail = pl.col_expand_mul(q_head_row_scaled_tail, q_head_scale_tail)
                if MODE >= 2:
                    q_head_dq_tail = pl.cast(pl.cast(dequantized_tail, pl.BF16, mode="rint"), pl.FP32)
                else:
                    q_head_dq_tail = dequantized_tail
                q_head_sq_tail = pl.mul(q_head_dq_tail, q_head_dq_tail)
                if MODE >= 4:
                    half_tail = pl.add(q_head_sq_tail[:, :256], q_head_sq_tail[:, 256:512])
                    quarter_tail = pl.add(half_tail[:, :128], half_tail[:, 128:256])
                    lanes_tail = pl.add(quarter_tail[:, :64], quarter_tail[:, 64:128])
                    q_head_sq_sum_tail = pl.row_sum(
                        lanes_tail, tmp_tile=pl.tile.create([Q_ROPE_T_TILE, 64], dtype=pl.FP32)
                    )
                else:
                    q_head_sq_sum_tail = pl.row_sum(q_head_sq_tail, q_head_reduce_tmp)
                # MODE<5保留原尾行recip(sqrt)，MODE5使用与Native对齐的高精度rsqrt。
                variance_tail = pl.add(pl.mul(q_head_sq_sum_tail, 1.0 / HEAD_DIM), EPS)
                if MODE >= 5:
                    q_head_inv_rms_tail = pl.tile.rsqrt(
                        variance_tail, pl.tile.create([Q_ROPE_T_TILE, 1], dtype=pl.FP32)
                    )
                else:
                    q_head_inv_rms_tail = pl.recip(pl.sqrt(variance_tail))
                q_nope_normed_tail = pl.row_expand_mul(q_head_dq_tail[:, 0:NOPE_DIM], q_head_inv_rms_tail)
                q_nope_bf16_tail = pl.cast(q_nope_normed_tail, target_type=pl.BF16, mode="rint")
                q_nope_valid = pl.set_validshape(q_nope_bf16_tail, valid_tail_rows, NOPE_DIM)
                pl.store(q_nope_valid, [tg, h0_tail], q_flat)
                q_rope_chunk_raw_tail = q_head_dq_tail[:, NOPE_DIM:HEAD_DIM]
                rope_normalized_tail = pl.row_expand_mul(q_rope_chunk_raw_tail, q_head_inv_rms_tail)
                if MODE >= 3:
                    q_rope_chunk_tail = pl.cast(pl.cast(rope_normalized_tail, pl.BF16, mode="rint"), pl.FP32)
                else:
                    q_rope_chunk_tail = rope_normalized_tail
                q_rope_swapped_tail = pl.tile.gather(q_rope_chunk_tail, q_swap_idx_tail, q_gather_tmp)
                rope_valid = pl.set_validshape(q_rope_chunk_tail, valid_tail_rows, ROPE_DIM)
                swapped_valid = pl.set_validshape(q_rope_swapped_tail, valid_tail_rows, ROPE_DIM)
                q_rope_base_tail = pl.mul(rope_valid, q_cos_il_tail)
                q_rope_delta_tail = pl.mul(swapped_valid, q_sin_signed_tail)
                q_rope_rot_tail = pl.add(q_rope_base_tail, q_rope_delta_tail)
                q_rope_bf16_tail = pl.cast(q_rope_rot_tail, target_type=pl.BF16, mode="rint")
                q_rope_valid = pl.set_validshape(q_rope_bf16_tail, valid_tail_rows, ROPE_DIM)
                pl.store(q_rope_valid, [tg, h0_tail + NOPE_DIM], q_flat)
    return q_flat
