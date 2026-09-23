# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CSA integration dependency adapted from pypto-lib 205255b4/decode_sparse_attn_csa.py."""

import pypto.language as pl

from .config import (
    BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    KV_ORI_BLOCK_NUM,
    TP,
)
from .config import (
    FLASH as M,
)
from .layout import COMPRESSED_TABLE_COLUMNS_DYN, ORIGINAL_TABLE_COLUMNS_DYN

B_DYN = pl.dynamic("B_DYN")  # per-request axis (block tables)

T_DYN = pl.dynamic("T_DYN")  # T = B * S

ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")

CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")

B = DECODE_BATCH // TP

S = DECODE_SEQ

T = B * S

D = M.hidden_size

H = M.num_attention_heads

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim

NOPE_DIM = M.nope_head_dim

WIN = M.sliding_window

MAX_SEQ_LEN = M.max_position_embeddings

IDX_TOPK = M.index_topk

CMP_TOPK = IDX_TOPK

SOFTMAX_SCALE = M.softmax_scale

O_LORA = M.o_lora_rank

O_GROUPS = M.o_groups

HEADS_PER_GROUP = H // O_GROUPS

O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

COMPRESS_RATIO = 4

COMPRESS_RATIO_INV = 1.0 / COMPRESS_RATIO

CSA_CMP_GE_BIAS = 1.0  # raw + 1, folded for the ge clamp

NEG_INF = -1.0e20

ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE

ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM

CMP_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE

H_TILE = 16

MERGE_WORKERS = 48

QK_KV_READY_EVENT = 0

QK_SCORE_READY_EVENT = 1

QK_PROB_READY_EVENT = 2

QK_PV_READY_EVENT = 3

# Native A3 softmax rounds probabilities after each 512-candidate update.
# Cube transfers remain smaller so Q and KV fit in A3 L1 together.
ATTN_K_TILE = 512

ATTN_CUBE_KV_TILE = 128

SOFTMAX_HEAD_TILE = 8

NUM_QK_CORES = 24  # qk_pv dispatch lanes

CSA_PLAN_WORKERS = 16  # csa_slots_build_valid_qk_plan token-tile lanes

T_PAD = ((T + 16 - 1) // 16) * 16  # Cube M floor

ATTENTION_PUBLISH_WORKERS = 48

ATTENTION_PUBLISH_T_TILE = 4

LOCAL_O_GROUPS = O_GROUPS // TP

GROUP_T_PAD = TP * T_PAD

ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD

PUBLISH_GROUPS = H_TILE // HEADS_PER_GROUP

ROPE_CS_T_TILE = 8

TOPK = WIN + CMP_TOPK

SPARSE_BLOCKS = 1 + (CMP_TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE

MASK_LINE_ELEMS = 64 // 4

VALID_BLOCK_MASK_COLS = ((SPARSE_BLOCKS + MASK_LINE_ELEMS - 1) // MASK_LINE_ELEMS) * MASK_LINE_ELEMS

PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE

SWA_TILE_WIN_ROWS = min(ATTN_K_TILE, WIN)

SWA_RUNS = (min(ATTN_CUBE_KV_TILE, WIN) + 2 * (BLOCK_SIZE - 1)) // BLOCK_SIZE

BIAS_T_TILE = min(T, 8)

if T % BIAS_T_TILE != 0:
    raise ValueError("CSA token capacity must contain complete bias tiles")

if H_TILE % HEADS_PER_GROUP != 0:
    raise ValueError(f"CSA head tile {H_TILE} must contain complete output groups")

if PUBLISH_GROUPS != 2:
    raise ValueError("CSA TP1 merge requires exactly two output groups per head tile")

if O_GROUPS % TP != 0:
    raise ValueError(f"output groups {O_GROUPS} must be divisible by TP size {TP}")

if LOCAL_O_GROUPS % PUBLISH_GROUPS != 0:
    raise ValueError("local output groups must contain complete CSA publish tiles")

if T % ATTENTION_PUBLISH_T_TILE != 0:
    raise ValueError("local token capacity must contain complete attention publish tiles")


@pl.jit.inline(auto_scope=False)
def sparse_attn_csa(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32],
    idx_topk: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    position_ids: pl.Tensor[[T_DYN, 1], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
):
    """Plan and run CSA QK/PV over sparse blocks, and build inverse-RoPE metadata."""
    # Compressed index contract.
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    t_dim = pl.tensor.dim(q, 0)
    t_heads = t_dim * H
    rope_cs_blocks = (t_dim + ROPE_CS_T_TILE - 1) // ROPE_CS_T_TILE
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])

    # pypto-lib#481 original-cache WAR marker.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_touch", allow_early_resolve=True):
        ori_kv_flat[0:1, 0:HEAD_DIM] = ori_kv_flat[0:1, 0:HEAD_DIM]

    # Sparse slot indices, additive softmax bias, and per-block validity.
    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)
    cmp_sparse_indices = pl.create_tensor([t_dim, CMP_TOPK], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([t_dim, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    # Read Native positions/page tables at the consumer. No expanded SWA index
    # matrix or intermediate INT32 position buffer is needed.
    with pl.spmd(CSA_PLAN_WORKERS, name_hint="csa_slots_build_valid_qk_plan", allow_early_resolve=True) as qk_plan_tid:
        plan_worker = pl.tile.get_block_idx()
        for bias_t0 in pl.range(plan_worker * BIAS_T_TILE, t_dim, CSA_PLAN_WORKERS * BIAS_T_TILE):
            bias_rows = pl.min(BIAS_T_TILE, t_dim - bias_t0)
            for bias_dt in pl.range(bias_rows):
                bias_t = bias_t0 + bias_dt
                bias_request = bias_t // S
                c_position = pl.cast(pl.read(position_ids, [bias_t, 0]), pl.INDEX)
                c_raw = pl.cast(idx_topk[bias_t : bias_t + 1, 0:IDX_TOPK], target_type=pl.FP32)
                c_upper = pl.cast(pl.cast((c_position + 1) // COMPRESS_RATIO, pl.INT32), pl.FP32)
                c_ge = pl.minimum(pl.maximum(pl.add(c_raw, CSA_CMP_GE_BIAS), 0.0), 1.0)
                c_lt = pl.minimum(pl.maximum(pl.add(pl.neg(c_raw), c_upper), 0.0), 1.0)
                c_mask = pl.mul(c_ge, c_lt)
                c_out = pl.sub(pl.mul(c_mask, pl.add(c_raw, 1.0)), 1.0)
                cmp_sparse_indices[bias_t : bias_t + 1, 0:IDX_TOPK] = pl.cast(c_out, target_type=pl.INT32)

                # Match the Native page-valid window, including short histories
                # and negative page entries. Intervals from distinct pages do not overlap.
                v_length = pl.min(c_position + 1, WIN)
                v_start = c_position - v_length + 1
                v_head = v_start % BLOCK_SIZE
                v_columns = pl.cast(pl.arange(0, [1, WIN], dtype=pl.INT32), pl.FP32)
                v_valid = pl.full([1, WIN], dtype=pl.FP32, value=0.0)
                v_block_valid = pl.cast(0, pl.INT32)
                for v_run in pl.range(SWA_RUNS):
                    v_lo = pl.max(v_run * BLOCK_SIZE - v_head, 0)
                    v_hi = pl.min((v_run + 1) * BLOCK_SIZE - v_head, v_length)
                    if v_hi > v_lo:
                        v_page = pl.read(ori_block_table, [bias_request, (v_start + v_lo) // BLOCK_SIZE])
                        if v_page >= 0:
                            v_block_valid = pl.cast(1, pl.INT32)
                            v_lo_fp32 = pl.cast(pl.cast(v_lo, pl.INT32), pl.FP32)
                            v_hi_fp32 = pl.cast(pl.cast(v_hi, pl.INT32), pl.FP32)
                            v_ge = pl.minimum(pl.maximum(pl.add(pl.sub(v_columns, v_lo_fp32), 1.0), 0.0), 1.0)
                            v_lt = pl.minimum(pl.maximum(pl.add(pl.neg(v_columns), v_hi_fp32), 0.0), 1.0)
                            v_valid = pl.add(v_valid, pl.mul(v_ge, v_lt))
                pl.write(valid_block_mask, [bias_t, 0], v_block_valid)

                sparse_bias[bias_t : bias_t + 1, 0:WIN] = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)
                sparse_bias[bias_t : bias_t + 1, WIN:ATTN_K_TILE] = pl.full(
                    [1, ATTN_K_TILE - WIN], dtype=pl.FP32, value=NEG_INF
                )
                sparse_bias[bias_t : bias_t + 1, ATTN_K_TILE : ATTN_K_TILE + CMP_TOPK] = pl.mul(
                    pl.minimum(c_out, 0.0), -NEG_INF
                )
            # Eight rows keep the column-major reduction output 32-byte aligned.
            # Reuse the final sparse indices; no position/window adapter buffer.
            c_blk_tmp = pl.create_tile([BIAS_T_TILE, ATTN_K_TILE], dtype=pl.FP32)
            for c_sb in pl.range(1, SPARSE_BLOCKS):
                c_s0 = (c_sb - 1) * ATTN_K_TILE
                c_block_slots = pl.cast(
                    pl.load(
                        cmp_sparse_indices,
                        [bias_t0, c_s0],
                        [BIAS_T_TILE, ATTN_K_TILE],
                        valid_shape=[bias_rows, ATTN_K_TILE],
                    ),
                    target_type=pl.FP32,
                )
                c_blk_valid = pl.row_max(pl.minimum(pl.maximum(pl.add(c_block_slots, 1.0), 0.0), 1.0), c_blk_tmp)
                for c_dt in pl.range(bias_rows):
                    c_valid = pl.cast(pl.read(c_blk_valid, [c_dt, 0]), pl.INT32)
                    pl.write(valid_block_mask, [bias_t0 + c_dt, c_sb], c_valid)

    # QK/PV scratch tensors.
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [t_heads, HEAD_DIM])
    attn_sink_col = pl.reshape(attn_sink, [H, 1])
    attn_mi = pl.create_tensor([t_heads, 1], dtype=pl.FP32)
    attn_li = pl.create_tensor([t_heads, 1], dtype=pl.FP32)
    attn_oi = pl.create_tensor([t_heads, HEAD_DIM], dtype=pl.FP32)

    transfer_heads = NUM_QK_CORES * H
    transfer_kv_rows = NUM_QK_CORES * ATTN_K_TILE
    kv_transfer = pl.create_tensor([transfer_kv_rows, HEAD_DIM], dtype=pl.BF16)
    score_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.FP32)
    probability_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.BF16)
    pv_transfer = pl.create_tensor([transfer_heads, HEAD_DIM], dtype=pl.FP32)
    mi_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    li_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    alpha_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
    with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[qk_plan_tid], allow_early_resolve=True) as qk_tid:
        qk_core = pl.tile.get_block_idx()
        qk_kv_base = qk_core * ATTN_K_TILE
        qk_head_base = qk_core * H
        pl.system.set_ffts(ffts_workspace)
        for qk_t in pl.range(qk_core, t_dim, NUM_QK_CORES):
            qk_b = qk_t // S
            qk_q = pl.load(q_flat, [qk_t * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            for qk_sb in pl.range(SPARSE_BLOCKS):
                if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                    pl.system.sync_wait(QK_KV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    for qk_part in pl.range(ATTN_K_TILE // ATTN_CUBE_KV_TILE):
                        qk_col = qk_part * ATTN_CUBE_KV_TILE
                        qk_kv = pl.load(
                            kv_transfer,
                            [qk_kv_base + qk_col, 0],
                            [ATTN_CUBE_KV_TILE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        qk_scores = pl.matmul(qk_q, pl.tile.transpose_view(qk_kv), out_dtype=pl.FP32)
                        pl.store(qk_scores, [qk_head_base, qk_col], score_transfer)
                    pl.system.sync_set(
                        QK_SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
                    )
                    pl.system.sync_wait(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    # Keep a single FP32 PV accumulator across the Native 512-candidate block.
                    pv_acc = pl.create_tile([H, HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                    for pv_part in pl.range(ATTN_K_TILE // ATTN_CUBE_KV_TILE):
                        pv_col = pv_part * ATTN_CUBE_KV_TILE
                        pv_probability = pl.load(
                            probability_transfer,
                            [qk_head_base, pv_col],
                            [H, ATTN_CUBE_KV_TILE],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv_kv = pl.load(
                            kv_transfer,
                            [qk_kv_base + pv_col, 0],
                            [ATTN_CUBE_KV_TILE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv_acc = pl.matmul_acc(pv_acc, pv_probability, pv_kv, init_cond=(pv_part == 0))
                    pl.store(pv_acc, [qk_head_base, 0], pv_transfer)
                    pl.system.sync_set(
                        QK_PV_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
                    )

            for qk_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
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
                for qk_sb, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                    SPARSE_BLOCKS, init_values=(running_m, running_l, running_left, running_right)
                ):
                    if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                        # Gather in 64-row pieces per AIV; a whole 512x512 KV tile would exceed UB.
                        for qk_part in pl.range(ATTN_K_TILE // ATTN_CUBE_KV_TILE):
                            qk_part_row = qk_part * ATTN_CUBE_KV_TILE + qk_lane_kv
                            qk_kv_half = pl.tile.full([ATTN_CUBE_KV_TILE // 2, HEAD_DIM], dtype=pl.BF16, value=0.0)
                            if qk_sb == 0:
                                qk_pos = pl.cast(pl.read(position_ids, [qk_t, 0]), pl.INDEX)
                                qk_win_len = pl.min(qk_pos + 1, WIN)
                                qk_win_start = qk_pos - qk_win_len + 1
                                qk_head = qk_win_start % BLOCK_SIZE
                                qk_rows = pl.min(pl.max(qk_win_len - qk_part_row, 0), ATTN_CUBE_KV_TILE // 2)
                                for qk_run in pl.unroll(SWA_RUNS):
                                    qk_lo = pl.max(qk_run * BLOCK_SIZE - qk_head - qk_part_row, 0)
                                    qk_hi = pl.min((qk_run + 1) * BLOCK_SIZE - qk_head - qk_part_row, qk_rows)
                                    if qk_hi > qk_lo:
                                        qk_absolute = qk_win_start + qk_part_row + qk_lo
                                        qk_raw_page = pl.read(ori_block_table, [qk_b, qk_absolute // BLOCK_SIZE])
                                        if qk_raw_page >= 0:
                                            qk_raw_row = (
                                                pl.cast(qk_raw_page, pl.INDEX) * BLOCK_SIZE + qk_absolute % BLOCK_SIZE
                                            )
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
                                        qk_ridx = pl.read(cmp_sparse_indices, [qk_t, qk_cmp_k])
                                        if qk_ridx >= 0:
                                            qk_page = pl.cast(
                                                pl.read(cmp_block_table, [qk_b, qk_ridx // BLOCK_SIZE]), pl.INDEX
                                            )
                                            qk_src = qk_page * BLOCK_SIZE + qk_ridx % BLOCK_SIZE
                                            qk_kv_half = pl.gather_row(
                                                qk_kv_half, cmp_kv_flat, [qk_row, 0], [qk_src, 0], [1, HEAD_DIM]
                                            )
                            pl.store(qk_kv_half, [qk_kv_base + qk_part_row, 0], kv_transfer)
                        pl.system.sync_set(
                            QK_KV_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                        )
                        pl.system.sync_wait(QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        for sm_part in pl.range((H // 2) // SOFTMAX_HEAD_TILE):
                            sm_h = sm_part * SOFTMAX_HEAD_TILE
                            sm_row = qk_head_base + qk_lane_head + sm_h
                            sm_scores = pl.load(score_transfer, [sm_row, 0], [SOFTMAX_HEAD_TILE, ATTN_K_TILE])
                            sm_bias = pl.load(sparse_bias, [qk_t, qk_sb * ATTN_K_TILE], [1, ATTN_K_TILE])
                            sm_masked = pl.col_expand_add(pl.mul(sm_scores, SOFTMAX_SCALE), sm_bias)
                            sm_block_max = pl.row_max(sm_masked, qk_reduce_tmp)
                            sm_old_m = pl.load(mi_transfer, [sm_row, 0], [SOFTMAX_HEAD_TILE, 1])
                            sm_old_l = pl.load(li_transfer, [sm_row, 0], [SOFTMAX_HEAD_TILE, 1])
                            sm_max = pl.maximum(sm_old_m, sm_block_max)
                            sm_alpha = pl.exp(pl.sub(sm_old_m, sm_max))
                            sm_exp = pl.exp(pl.row_expand_sub(sm_masked, sm_max))
                            sm_sum = pl.add(pl.mul(sm_old_l, sm_alpha), pl.row_sum(sm_exp, qk_reduce_tmp))
                            # Native SAS uses CAST_ROUND for probabilities: ties
                            # round away from zero, unlike the final output cast.
                            sm_probability = pl.cast(sm_exp, target_type=pl.BF16, mode="round")
                            pl.store(sm_probability, [sm_row, 0], probability_transfer)
                            pl.store(sm_max, [sm_row, 0], mi_transfer)
                            pl.store(sm_sum, [sm_row, 0], li_transfer)
                            pl.store(sm_alpha, [sm_row, 0], alpha_transfer)
                        pl.system.sync_set(
                            QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                        )
                        pl.system.sync_wait(QK_PV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        pv_row = qk_head_base + qk_lane_head
                        next_m = pl.load(mi_transfer, [pv_row, 0], [H // 2, 1])
                        next_l = pl.load(li_transfer, [pv_row, 0], [H // 2, 1])
                        alpha = pl.load(alpha_transfer, [pv_row, 0], [H // 2, 1])
                        pv_left = pl.load(pv_transfer, [pv_row, 0], [H // 2, HEAD_DIM // 2])
                        next_left = pl.add(pl.row_expand_mul(left_iter, alpha), pv_left)
                        pv_right = pl.load(pv_transfer, [pv_row, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2])
                        next_right = pl.add(pl.row_expand_mul(right_iter, alpha), pv_right)
                        m_after, l_after, left_after, right_after = pl.yield_(next_m, next_l, next_left, next_right)
                    else:
                        m_after, l_after, left_after, right_after = pl.yield_(m_iter, l_iter, left_iter, right_iter)
                    running_m, running_l, running_left, running_right = pl.yield_(
                        m_after, l_after, left_after, right_after
                    )
                qk_output_row = qk_t * H + qk_lane_head
                pl.store(running_m, [qk_output_row, 0], attn_mi)
                pl.store(running_l, [qk_output_row, 0], attn_li)
                pl.store(running_left, [qk_output_row, 0], attn_oi)
                pl.store(running_right, [qk_output_row, HEAD_DIM // 2], attn_oi)

    # Native cosine rows already have the consumer's interleaved layout.
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    # Inverse-RoPE lane-swap index.
    rope_swap_idx = pl.create_tensor([H_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs", allow_early_resolve=True) as rope_tid:
        sw_ones = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        sw_idx_f = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        sw_col = pl.col_expand_mul(sw_ones, sw_idx_f)
        sw_dup_i32 = pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc")
        sw_dup_f = pl.cast(sw_dup_i32, target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        sw_swap_f = pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0))
        rope_swap_idx[0:H_TILE, 0:ROPE_DIM] = pl.cast(sw_swap_f, target_type=pl.INT32)

        cs_ones = pl.tile.full([ROPE_CS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        cs_idx_f = pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_idx_f)
        cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        for cs_rb in pl.range(rope_cs_blocks):
            cs_t0 = cs_rb * ROPE_CS_T_TILE
            cs_rows = pl.min(ROPE_CS_T_TILE, t_dim - cs_t0)
            cs_sin = pl.load(freqs_sin, [cs_t0, 0], [ROPE_CS_T_TILE, ROPE_DIM], valid_shape=[cs_rows, ROPE_DIM])
            cs_sign_rows = pl.set_validshape(cs_sign, cs_rows, ROPE_DIM)
            pl.store(pl.mul(cs_sin, cs_sign_rows), [cs_t0, 0], rope_sin_signed)

    return (
        attn_mi,
        attn_li,
        attn_oi,
        freqs_cos,
        rope_sin_signed,
        rope_swap_idx,
        qk_tid,
        rope_tid,
    )


@pl.jit.inline
def sparse_attn_csa_tp1(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32],
    idx_topk: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    position_ids: pl.Tensor[[T_DYN, 1], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Write CSA heads as ``[group, T_PAD, O_GROUP_IN]`` slabs.

    Only the first runtime ``t_dim`` rows in each group are valid. The
    returned task ID covers every write to the packed output tensor.
    """
    (
        attn_mi,
        attn_li,
        attn_oi,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
        qk_tid,
        rope_tid,
    ) = sparse_attn_csa(
        q,
        ori_kv,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        idx_topk,
        position_ids,
        attn_sink,
        freqs_cos,
        freqs_sin,
    )
    t_dim = pl.tensor.dim(q, 0)

    with pl.spmd(MERGE_WORKERS, name_hint="merge_norm", deps=[qk_tid, rope_tid]) as merge_tid:
        m_worker = pl.tile.get_block_idx()
        m_swap = pl.load(rope_swap_idx, [0, 0], [H_TILE, ROPE_DIM])
        m_swap_f = pl.cast(m_swap, target_type=pl.FP32)
        m_swap_source = pl.add(m_swap_f, NOPE_DIM)
        m_row_ids = pl.tile.arange(0, [1, H_TILE], dtype=pl.INT32)
        m_row_ids_f = pl.cast(m_row_ids, target_type=pl.FP32)
        m_row_offsets = pl.mul(m_row_ids_f, HEAD_DIM)
        m_row_offsets_col = pl.reshape(m_row_offsets, [H_TILE, 1])
        m_swap_flat = pl.row_expand_add(m_swap_source, m_row_offsets_col)
        m_swap_idx = pl.cast(m_swap_flat, target_type=pl.INT32)
        m_gather_tmp = pl.create_tile([H_TILE, ROPE_DIM], dtype=pl.INT32)
        for m_idx in pl.range(m_worker, t_dim * (H // H_TILE), MERGE_WORKERS):
            m_t = m_idx // (H // H_TILE)
            m_h_idx = m_idx - m_t * (H // H_TILE)
            m_h0 = m_h_idx * H_TILE
            m_row = m_idx * H_TILE
            m_li = pl.load(attn_li, [m_row, 0], [H_TILE, 1])
            m_oi = pl.load(attn_oi, [m_row, 0], [H_TILE, HEAD_DIM])

            # The online sum already includes the sink's unit initial weight.
            n_full = pl.row_expand_div(m_oi, m_li)
            n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
            # Native sparse attention publishes BF16 before inverse RoPE.
            n_rounded = pl.cast(n_bf16, target_type=pl.FP32)

            # Inverse-RoPE head tile.
            m_rope = n_rounded[0:H_TILE, NOPE_DIM:HEAD_DIM]
            m_cos_il = pl.load(rope_cos_il, [m_t, 0], [1, ROPE_DIM])
            m_sin_signed = pl.load(rope_sin_signed, [m_t, 0], [1, ROPE_DIM])
            m_swapped = pl.tile.gather(n_rounded, m_swap_idx, m_gather_tmp)
            m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
            n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")
            n_full_bf16 = pl.concat(n_bf16[0:H_TILE, 0:NOPE_DIM], n_rope_bf16)

            n_group_bf16 = pl.reshape(n_full_bf16, [PUBLISH_GROUPS, O_GROUP_IN])
            n_pack_first = n_group_bf16[0:1, 0:O_GROUP_IN]
            n_pack_second = n_group_bf16[1:2, 0:O_GROUP_IN]
            n_pack_row = (m_h0 // HEADS_PER_GROUP) * T_PAD + m_t
            n_pack_row_second = n_pack_row + T_PAD
            pl.store(n_pack_first, [n_pack_row, 0], o_packed_heads)
            pl.store(n_pack_second, [n_pack_row_second, 0], o_packed_heads)

    return o_packed_heads, merge_tid
