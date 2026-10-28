# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native K512 attention数值更新；保留单SPMD，内部采用保守两更新握手。

数学取自旧精度7472fb9a，并已在当前SDK、同一实际Native输入下独立复核。
外层phase/row owner不变；每次SWA或compressed更新均为KV→SCORE→PROB→PV。
没有沿用K128三槽的提前两拍等待，因此不会在四片score到齐前等待其概率。
"""

import pypto.language as pl

from .attention_stream_stages import (
    BLOCK_SIZE,
    COMPRESSED_COLUMNS_DYN,
    COMPRESSED_ROWS_DYN,
    FFTS_WORDS,
    HEAD_DIM,
    HEAD_RING_ROWS_DYN,
    KV_RING_ROWS_DYN,
    NOPE_DIM,
    ORIGINAL_COLUMNS_DYN,
    ORIGINAL_ROWS_DYN,
    PACK_ROWS_DYN,
    PLAN_CAP_DYN,
    Q_ROWS_DYN,
    QK_KV_READY_EVENT,
    QK_PROB_READY_EVENT,
    QK_PV_READY_EVENT,
    QK_SCORE_READY_EVENT,
    ROPE_CAP_DYN,
    ROPE_DIM,
    SOFTMAX_SCALE,
    T_DYN,
    WIN,
    H,
)
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, query_request

NATIVE_HEAD_ROWS = pl.dynamic("CSA_NATIVE_ATTN_HEAD_ROWS")
ATTN_K_TILE = 512
ATTN_CUBE_KV_TILE = 128
SPARSE_BLOCKS = 2
SOFTMAX_HEAD_TILE = 8
CMP_TOPK = 512
COMPRESS_RATIO = 4
SOFTMAX_ROWS = pl.dynamic("CSA_NATIVE_SOFTMAX_ROWS")
SWA_RUNS = 5
NEG_INF = -1.0e20
FINAL_HEAD_TILE = 16
O_GROUPS = 8
HEADS_PER_GROUP = 8
O_GROUP_IN = 4096
PUBLISH_GROUPS = 2


@pl.jit.inline(auto_scope=False)
def checked_compressed_index(
    candidate: pl.Scalar[pl.INT32],
    request: pl.Scalar[pl.INDEX],
    limit: pl.Scalar[pl.INDEX],
    table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    cache_pages: pl.Scalar[pl.INDEX],
):
    checked = pl.cast(-1, pl.INT32)
    if candidate >= 0 and candidate < limit:
        column = pl.cast(candidate, pl.INDEX) // BLOCK_SIZE
        if column < pl.tensor.dim(table, 1):
            page = pl.cast(pl.read(table, [request, column]), pl.INDEX)
            if page >= 0 and page < cache_pages:
                checked = candidate
    return checked


@pl.jit.inline(auto_scope=False)
def native_compressed_pair_swap(
    first: pl.Scalar[pl.INT32],
    second: pl.Scalar[pl.INT32],
    request: pl.Scalar[pl.INDEX],
    limit: pl.Scalar[pl.INDEX],
    table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
):
    swap = pl.cast(0, pl.INT32)
    if first >= 0 and second >= 0:
        if first + 1 < limit and second + 1 < limit:
            first_page = pl.cast(pl.read(table, [request, pl.cast(first, pl.INDEX) // BLOCK_SIZE]), pl.INDEX)
            second_page = pl.cast(pl.read(table, [request, pl.cast(second, pl.INDEX) // BLOCK_SIZE]), pl.INDEX)
            first_row = first_page * BLOCK_SIZE + pl.cast(first, pl.INDEX) % BLOCK_SIZE
            second_row = second_page * BLOCK_SIZE + pl.cast(second, pl.INDEX) % BLOCK_SIZE
            if first_row > second_row:
                if (first_row - second_row - 1) * HEAD_DIM * 2 < 2147483647:
                    swap = pl.cast(1, pl.INT32)
    return swap


@pl.jit.inline(auto_scope=False)
def native_attention_cube(
    q_flat: pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    kv_transfer: pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16],
    score_transfer: pl.InOut[pl.Tensor[[NATIVE_HEAD_ROWS, ATTN_K_TILE], pl.FP32]],
    probability_transfer: pl.Tensor[[NATIVE_HEAD_ROWS, ATTN_K_TILE], pl.BF16],
    pv_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    core_id: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 0,
):
    t_dim = pl.tensor.dim(q_flat, 0) // H
    qk_core = core_id
    qk_kv_base = qk_core * ATTN_K_TILE
    qk_head_base = qk_core * H
    pl.system.set_ffts(ffts_workspace)
    qk_l1 = pl.create_tile([3 * ATTN_CUBE_KV_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
    for qk_t in pl.range(qk_core, t_dim, CORE_GROUPS):
        qk_q = pl.load(q_flat, [qk_t * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
        for qk_sb in pl.range(SPARSE_BLOCKS):
            if qk_sb == 0:
                block_active = pl.yield_(pl.read(valid_block_mask, [qk_t, 0]))
            else:
                block_active = pl.yield_(
                    pl.max(
                        pl.max(pl.read(valid_block_mask, [qk_t, 1]), pl.read(valid_block_mask, [qk_t, 2])),
                        pl.max(pl.read(valid_block_mask, [qk_t, 3]), pl.read(valid_block_mask, [qk_t, 4])),
                    )
                )
            if block_active > 0:
                pl.system.sync_wait(EVENT_BASE + QK_KV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                for qk_part in pl.range(ATTN_K_TILE // ATTN_CUBE_KV_TILE):
                    qk_col = qk_part * ATTN_CUBE_KV_TILE
                    qk_slot_row = pl.max(qk_part - 1, 0) * ATTN_CUBE_KV_TILE
                    if qk_sb > 0 or qk_part == 0:
                        qk_l1 = pl.gather_row(
                            qk_l1,
                            kv_transfer,
                            [qk_slot_row, 0],
                            [qk_kv_base + qk_col, 0],
                            [ATTN_CUBE_KV_TILE, HEAD_DIM],
                        )
                        # Transpose the complete carrier before slicing: the current
                        # backend preserves the slice's column offset in matmul.
                        qk_l1_t = pl.tile.transpose_view(qk_l1)
                        qk_kv_t = pl.tile.slice(qk_l1_t, [HEAD_DIM, ATTN_CUBE_KV_TILE], [0, qk_slot_row])
                        qk_scores = pl.matmul(qk_q, qk_kv_t, out_dtype=pl.FP32)
                        pl.store(qk_scores, [qk_head_base, qk_col], score_transfer)
                pl.system.sync_set(
                    EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
                )
                pl.system.sync_wait(
                    EVENT_BASE + QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC
                )
                # Keep a single FP32 PV accumulator across the Native 512-candidate block.
                pv_acc = pl.create_tile([H, HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                for pv_part in pl.range(ATTN_K_TILE // ATTN_CUBE_KV_TILE):
                    if qk_sb > 0 or pv_part == 0:
                        pv_col = pv_part * ATTN_CUBE_KV_TILE
                        pv_probability = pl.load(
                            probability_transfer,
                            [qk_head_base, pv_col],
                            [H, ATTN_CUBE_KV_TILE],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv_slot_row = pl.max(pv_part - 1, 0) * ATTN_CUBE_KV_TILE
                        if pv_part < 2:
                            qk_l1 = pl.gather_row(
                                qk_l1,
                                kv_transfer,
                                [pv_slot_row, 0],
                                [qk_kv_base + pv_col, 0],
                                [ATTN_CUBE_KV_TILE, HEAD_DIM],
                            )
                        pv_kv = pl.tile.slice(qk_l1, [ATTN_CUBE_KV_TILE, HEAD_DIM], [pv_slot_row, 0])
                        pv_acc = pl.matmul_acc(pv_acc, pv_probability, pv_kv, init_cond=(pv_part == 0))
                pl.store(pv_acc, [qk_head_base, 0], pv_transfer)
                pl.system.sync_set(
                    EVENT_BASE + QK_PV_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
                )
    return score_transfer, pv_transfer


@pl.jit.inline(auto_scope=False)
def native_attention_vector(
    ori_kv_flat: pl.Tensor[[ORIGINAL_ROWS_DYN, HEAD_DIM], pl.BF16],
    cmp_kv_flat: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    cmp_sparse_indices: pl.Tensor[[PLAN_CAP_DYN, 512], pl.INT32],
    sparse_bias: pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    attn_sink_col: pl.Tensor[[H, 1], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_CAP_DYN, ROPE_DIM], pl.FP32],
    kv_transfer: pl.InOut[pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16]],
    score_transfer: pl.Tensor[[NATIVE_HEAD_ROWS, ATTN_K_TILE], pl.FP32],
    probability_transfer: pl.InOut[pl.Tensor[[NATIVE_HEAD_ROWS, ATTN_K_TILE], pl.BF16]],
    pv_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32],
    mi_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    li_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    o_packed_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    softmax_copy64: pl.InOut[pl.Tensor[[SOFTMAX_ROWS, 64], pl.FP32]],
    pre_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16]],
    core_id: pl.Scalar[pl.INDEX],
    lane_id: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 0,
    OBSERVE: pl.constexpr = False,
):
    t_dim = pl.tensor.dim(position_ids, 0)
    qk_core = core_id
    qk_aiv = lane_id
    qk_kv_base = qk_core * ATTN_K_TILE
    qk_head_base = qk_core * H
    for qk_t in pl.range(qk_core, t_dim, CORE_GROUPS):
        qk_b, query_begin, query_end, active = query_request(bounds, lengths, qk_t, t_dim)
        qk_b = pl.max(qk_b, 0)
        pl.system.set_ffts(ffts_workspace)
        qk_lane_head = qk_aiv * (H // 2)
        qk_lane_kv = qk_aiv * (ATTN_CUBE_KV_TILE // 2)
        qk_reduce_tmp = pl.create_tile(
            [SOFTMAX_HEAD_TILE, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        running_m = pl.load(attn_sink_col, [qk_lane_head, 0], [H // 2, 1])
        running_l = pl.tile.adds(pl.tile.muls(running_m, 0.0), 1.0)
        running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        running_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        # Publish row statistics so each head group reads its explicit GM row.
        pl.store(running_m, [qk_head_base + qk_lane_head, 0], mi_transfer)
        pl.store(running_l, [qk_head_base + qk_lane_head, 0], li_transfer)
        for qk_sb, (l_iter, left_iter, right_iter) in pl.range(
            SPARSE_BLOCKS, init_values=(running_l, running_left, running_right)
        ):
            if qk_sb == 0:
                block_active = pl.yield_(pl.read(valid_block_mask, [qk_t, 0]))
            else:
                block_active = pl.yield_(
                    pl.max(
                        pl.max(pl.read(valid_block_mask, [qk_t, 1]), pl.read(valid_block_mask, [qk_t, 2])),
                        pl.max(pl.read(valid_block_mask, [qk_t, 3]), pl.read(valid_block_mask, [qk_t, 4])),
                    )
                )
            if block_active > 0:
                # Gather in 64-row pieces per AIV; a whole 512x512 KV tile would exceed UB.
                for qk_part in pl.range(ATTN_K_TILE // ATTN_CUBE_KV_TILE):
                    if qk_sb > 0 or qk_part == 0:
                        qk_part_row = qk_part * ATTN_CUBE_KV_TILE + qk_lane_kv
                        qk_kv_half = pl.tile.full([ATTN_CUBE_KV_TILE // 2, HEAD_DIM], dtype=pl.BF16, value=0.0)
                        if qk_sb == 0:
                            qk_pos = pl.cast(pl.read(position_ids, [qk_t]), pl.INDEX)
                            qk_win_len = pl.min(qk_pos + 1, WIN)
                            qk_win_start = qk_pos - qk_win_len + 1
                            qk_head = qk_win_start % BLOCK_SIZE
                            qk_rows = pl.min(pl.max(qk_win_len - qk_part_row, 0), ATTN_CUBE_KV_TILE // 2)
                            for qk_run in pl.unroll(SWA_RUNS):
                                qk_lo = pl.max(qk_run * BLOCK_SIZE - qk_head - qk_part_row, 0)
                                qk_hi = pl.min((qk_run + 1) * BLOCK_SIZE - qk_head - qk_part_row, qk_rows)
                                if qk_hi > qk_lo:
                                    qk_absolute = qk_win_start + qk_part_row + qk_lo
                                    qk_raw_column = qk_absolute // BLOCK_SIZE
                                    if qk_raw_column >= 0 and qk_raw_column < pl.tensor.dim(ori_block_table, 1):
                                        qk_raw_page = pl.cast(pl.read(ori_block_table, [qk_b, qk_raw_column]), pl.INDEX)
                                        if (
                                            qk_raw_page >= 0
                                            and qk_raw_page < pl.tensor.dim(ori_kv_flat, 0) // BLOCK_SIZE
                                        ):
                                            qk_raw_row = qk_raw_page * BLOCK_SIZE + qk_absolute % BLOCK_SIZE
                                            qk_kv_half = pl.gather_row(
                                                qk_kv_half,
                                                ori_kv_flat,
                                                [qk_lo, 0],
                                                [qk_raw_row, 0],
                                                [ATTN_CUBE_KV_TILE // 2, HEAD_DIM],
                                                valid_shape=[qk_hi - qk_lo, HEAD_DIM],
                                            )
                        else:
                            for qk_row in pl.range(ATTN_CUBE_KV_TILE // 2):
                                qk_cmp_k = (qk_sb - 1) * ATTN_K_TILE + qk_part_row + qk_row
                                if qk_cmp_k < CMP_TOPK:
                                    qk_pair = qk_cmp_k // 2 * 2
                                    qk_limit = (pl.cast(pl.read(position_ids, [qk_t]), pl.INDEX) + 1) // 4
                                    qk_first = checked_compressed_index(
                                        pl.read(cmp_sparse_indices, [qk_t, qk_pair]),
                                        qk_b,
                                        qk_limit,
                                        cmp_block_table,
                                        pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE,
                                    )
                                    qk_second = checked_compressed_index(
                                        pl.read(cmp_sparse_indices, [qk_t, qk_pair + 1]),
                                        qk_b,
                                        qk_limit,
                                        cmp_block_table,
                                        pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE,
                                    )
                                    qk_swap = native_compressed_pair_swap(
                                        qk_first, qk_second, qk_b, qk_limit, cmp_block_table
                                    )
                                    if (qk_cmp_k % 2 == 0 and qk_swap == 0) or (qk_cmp_k % 2 == 1 and qk_swap != 0):
                                        qk_ridx = pl.yield_(qk_first)
                                    else:
                                        qk_ridx = pl.yield_(qk_second)
                                    if qk_ridx >= 0:
                                        qk_page = pl.cast(
                                            pl.read(cmp_block_table, [qk_b, qk_ridx // BLOCK_SIZE]), pl.INDEX
                                        )
                                        if qk_page >= 0:
                                            qk_src = qk_page * BLOCK_SIZE + qk_ridx % BLOCK_SIZE
                                            qk_kv_half = pl.gather_row(
                                                qk_kv_half, cmp_kv_flat, [qk_row, 0], [qk_src, 0], [1, HEAD_DIM]
                                            )
                        pl.store(qk_kv_half, [qk_kv_base + qk_part_row, 0], kv_transfer)
                pl.system.sync_set(
                    EVENT_BASE + QK_KV_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                )
                pl.system.sync_wait(
                    EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                )
                for sm_part in pl.range((H // 2) // SOFTMAX_HEAD_TILE):
                    sm_h = sm_part * SOFTMAX_HEAD_TILE
                    sm_row = qk_head_base + qk_lane_head + sm_h
                    if qk_sb == 0:
                        # Skipped Cube columns are explicit +0, never stale GM/NaN.
                        sm_live_scores = pl.load(score_transfer, [sm_row, 0], [SOFTMAX_HEAD_TILE, WIN])
                        sm_scores = pl.concat(
                            sm_live_scores,
                            pl.tile.full([SOFTMAX_HEAD_TILE, ATTN_K_TILE - WIN], dtype=pl.FP32, value=0.0),
                        )
                    else:
                        sm_scores = pl.load(score_transfer, [sm_row, 0], [SOFTMAX_HEAD_TILE, ATTN_K_TILE])
                    if qk_sb == 0:
                        sm_bias_live = pl.load(sparse_bias, [qk_t, 0], [1, WIN])
                        sm_bias = pl.yield_(
                            pl.concat(sm_bias_live, pl.tile.full([1, ATTN_K_TILE - WIN], dtype=pl.FP32, value=NEG_INF))
                        )
                    else:
                        sm_bias = pl.yield_(pl.load(sparse_bias, [qk_t, WIN], [1, ATTN_K_TILE]))
                    sm_masked = pl.col_expand_add(pl.mul(sm_scores, SOFTMAX_SCALE), sm_bias)
                    sm_block_max = pl.row_max(sm_masked, qk_reduce_tmp)
                    sm_old_m = pl.load(mi_transfer, [sm_row, 0], [SOFTMAX_HEAD_TILE, 1])
                    sm_old_l = pl.load(li_transfer, [sm_row, 0], [SOFTMAX_HEAD_TILE, 1])
                    sm_max = pl.maximum(sm_old_m, sm_block_max)
                    sm_alpha = pl.exp(pl.sub(sm_old_m, sm_max))
                    sm_exp = pl.exp(pl.row_expand_sub(sm_masked, sm_max))
                    # Native selects the reduction tree from the live actual K,
                    # not the padded 512-column probability storage. Empty
                    # blocks are skipped by valid_block_mask before this point.
                    sm_position = pl.cast(pl.read(position_ids, [qk_t]), pl.INDEX)
                    if qk_sb == 0:
                        sm_key_count = pl.min(sm_position + 1, WIN)
                    else:
                        sm_key_count = pl.min((sm_position + 1) // COMPRESS_RATIO, CMP_TOPK)
                    if sm_key_count < 64:
                        # WholeReduceSum uses actual K as its vector mask.
                        sm_short = pl.tile.slice(
                            sm_exp,
                            [SOFTMAX_HEAD_TILE, 64],
                            [0, 0],
                            valid_shape=[SOFTMAX_HEAD_TILE, sm_key_count],
                        )
                        sm_block_sum = pl.row_sum(sm_short, pl.create_tile([SOFTMAX_HEAD_TILE, 64], dtype=pl.FP32))
                    elif sm_key_count < 512:
                        # Copy into compact64 storage, then fold blocks in
                        # Native's sequential order. Masked-off tail channels
                        # retain their original bits, including subnormals.
                        sm_columns = pl.col_expand_add(
                            pl.tile.full([SOFTMAX_HEAD_TILE, 64], dtype=pl.FP32, value=0.0),
                            pl.cast(pl.tile.arange(0, [1, 64], dtype=pl.INT32), pl.FP32),
                        )
                        pl.store(sm_exp[0:SOFTMAX_HEAD_TILE, 0:64], [sm_row, 0], softmax_copy64)
                        sm_fold64 = pl.load(softmax_copy64, [sm_row, 0], [SOFTMAX_HEAD_TILE, 64])
                        for sm_chunk in pl.unroll(1, 8):
                            if sm_key_count > sm_chunk * 64:
                                sm_added = pl.add(
                                    sm_fold64,
                                    pl.tile.slice(sm_exp, [SOFTMAX_HEAD_TILE, 64], [0, sm_chunk * 64]),
                                )
                                if sm_key_count < (sm_chunk + 1) * 64:
                                    sm_tail = pl.cast(pl.cast(sm_key_count - sm_chunk * 64, pl.INT32), pl.FP32)
                                    sm_tail_mask = pl.tile.cmps(sm_columns, sm_tail, cmp_type=2)  # LT
                                    sm_fold64 = pl.tile.sel(
                                        sm_tail_mask,
                                        sm_added,
                                        sm_fold64,
                                        pl.create_tile([1, 16], dtype=pl.UINT32),
                                    )
                                else:
                                    sm_fold64 = sm_added
                        sm_groups64 = pl.reshape(sm_fold64, [SOFTMAX_HEAD_TILE * 8, 8])
                        sm_sum8 = pl.row_sum(sm_groups64, pl.create_tile([SOFTMAX_HEAD_TILE * 8, 8], dtype=pl.FP32))
                        sm_final8 = pl.reshape(sm_sum8, [SOFTMAX_HEAD_TILE, 8])
                        sm_block_sum = pl.row_sum(sm_final8, pl.create_tile([SOFTMAX_HEAD_TILE, 8], dtype=pl.FP32))
                    else:
                        # Native BigBlockReduceSum for the full K512 block.
                        sm_groups8 = pl.reshape(sm_exp, [SOFTMAX_HEAD_TILE * 64, 8])
                        sm_sum64 = pl.row_sum(sm_groups8, pl.create_tile([SOFTMAX_HEAD_TILE * 64, 8], dtype=pl.FP32))
                        sm_groups64 = pl.reshape(sm_sum64, [SOFTMAX_HEAD_TILE * 8, 8])
                        sm_sum8 = pl.row_sum(sm_groups64, pl.create_tile([SOFTMAX_HEAD_TILE * 8, 8], dtype=pl.FP32))
                        sm_final8 = pl.reshape(sm_sum8, [SOFTMAX_HEAD_TILE, 8])
                        sm_block_sum = pl.row_sum(sm_final8, pl.create_tile([SOFTMAX_HEAD_TILE, 8], dtype=pl.FP32))
                    sm_sum = pl.add(pl.mul(sm_old_l, sm_alpha), sm_block_sum)
                    # Native SAS uses CAST_ROUND for probabilities: ties
                    # round away from zero, unlike the final output cast.
                    sm_probability = pl.cast(sm_exp, target_type=pl.BF16, mode="round")
                    pl.store(sm_probability, [sm_row, 0], probability_transfer)
                    pl.store(sm_max, [sm_row, 0], mi_transfer)
                    pl.store(sm_sum, [sm_row, 0], li_transfer)
                    pl.store(sm_alpha, [sm_row + CORE_GROUPS * H, 0], li_transfer)
                pl.system.sync_set(
                    EVENT_BASE + QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                )
                pl.system.sync_wait(EVENT_BASE + QK_PV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                pv_row = qk_head_base + qk_lane_head
                next_l = pl.load(li_transfer, [pv_row, 0], [H // 2, 1])
                alpha = pl.load(li_transfer, [pv_row + CORE_GROUPS * H, 0], [H // 2, 1])
                pv_left = pl.load(pv_transfer, [pv_row, 0], [H // 2, HEAD_DIM // 2])
                next_left = pl.add(pl.row_expand_mul(left_iter, alpha), pv_left)
                pv_right = pl.load(pv_transfer, [pv_row, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2])
                next_right = pl.add(pl.row_expand_mul(right_iter, alpha), pv_right)
                l_after, left_after, right_after = pl.yield_(next_l, next_left, next_right)
            else:
                l_after, left_after, right_after = pl.yield_(l_iter, left_iter, right_iter)
            running_l, running_left, running_right = pl.yield_(l_after, left_after, right_after)
        packed_stride = pl.tensor.dim(o_packed_heads, 0) // 8
        # 复用性能版的整数换位索引：j + 1 - 2*(j % 2)，逐元素等于原表。
        # 只改变索引生成与依赖，保留累计 softmax、BF16 输出及逆 RoPE 的算术。
        m_idx = pl.tile.ci(0, [1, ROPE_DIM], dtype=pl.INT32)
        m_rem_tmp = pl.create_tile([1, ROPE_DIM], dtype=pl.INT32)
        m_lane = pl.tile.rems(m_idx, 2, m_rem_tmp)
        m_swap_row = pl.tile.adds(pl.tile.sub(m_idx, pl.tile.muls(m_lane, 2)), NOPE_DIM + 1)
        m_swap_base = pl.create_tile([FINAL_HEAD_TILE, ROPE_DIM], dtype=pl.INT32)
        m_swap_source = pl.col_expand(m_swap_base, m_swap_row)
        m_row_offsets = pl.tile.muls(pl.tile.ci(0, [1, FINAL_HEAD_TILE], dtype=pl.INT32), HEAD_DIM)
        m_swap_idx = pl.row_expand_add(m_swap_source, pl.reshape(m_row_offsets, [FINAL_HEAD_TILE, 1]))
        m_gather_tmp = pl.create_tile([FINAL_HEAD_TILE, ROPE_DIM], dtype=pl.INT32)
        for pub_h in pl.unroll(2):
            pub_local_head = pub_h * FINAL_HEAD_TILE
            m_h0 = qk_lane_head + pub_local_head
            m_t = qk_t
            if active != 0:
                m_li = pl.reshape(
                    pl.tile.extract(
                        pl.reshape(running_l, [1, H // 2]),
                        0,
                        pub_local_head,
                        [1, FINAL_HEAD_TILE],
                        target_memory=pl.MemorySpace.Vec,
                    ),
                    [FINAL_HEAD_TILE, 1],
                )
                m_oi = pl.concat(
                    pl.slice(running_left, [FINAL_HEAD_TILE, HEAD_DIM // 2], [pub_local_head, 0]),
                    pl.slice(running_right, [FINAL_HEAD_TILE, HEAD_DIM // 2], [pub_local_head, 0]),
                )

                # The online sum already includes the sink's unit initial weight.
                n_full = pl.row_expand_div(m_oi, m_li)
                n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
                # Native sparse attention publishes BF16 before inverse RoPE.
                n_rounded = pl.cast(n_bf16, target_type=pl.FP32)

                # Inverse-RoPE head tile.
                m_rope = n_rounded[0:FINAL_HEAD_TILE, NOPE_DIM:HEAD_DIM]
                m_cos_il = pl.load(freqs_cos, [m_t, 0], [1, ROPE_DIM])
                m_sin_signed = pl.load(rope_sin_signed, [m_t, 0], [1, ROPE_DIM])
                m_swapped = pl.tile.gather(n_rounded, m_swap_idx, m_gather_tmp)
                m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
                n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")
                n_full_bf16 = pl.concat(n_bf16[0:FINAL_HEAD_TILE, 0:NOPE_DIM], n_rope_bf16)
            else:
                n_full_bf16 = pl.tile.full([FINAL_HEAD_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                n_bf16 = n_full_bf16

            n_group_bf16 = pl.reshape(n_full_bf16, [PUBLISH_GROUPS, O_GROUP_IN])
            n_pack_first = n_group_bf16[0:1, 0:O_GROUP_IN]
            n_pack_second = n_group_bf16[1:2, 0:O_GROUP_IN]
            n_pack_row = (m_h0 // HEADS_PER_GROUP) * packed_stride + m_t
            n_pack_row_second = n_pack_row + packed_stride
            if OBSERVE:
                n_pre_group = pl.reshape(n_bf16, [PUBLISH_GROUPS, O_GROUP_IN])
                pl.store(n_pre_group[0:1, :], [n_pack_row, 0], pre_heads)
                pl.store(n_pre_group[1:2, :], [n_pack_row_second, 0], pre_heads)
            pl.store(n_pack_first, [n_pack_row, 0], o_packed_heads)
            pl.store(n_pack_second, [n_pack_row_second, 0], o_packed_heads)
    return kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads
