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

# QK/PV 软件流水：AIC 在第 k 拍算第 k 块的 QK、第 k-QK_PRE_LAUNCH 块的 PV，
# AIV 同拍采第 k 块 KV、做第 k-1 块 softmax、归并第 k-QK_PRE_LAUNCH-1 块。
# 槽数取 QK_PRE_LAUNCH+1，保证在飞的三块各有自己的 KV/score/prob 缓冲。
QK_PRE_LAUNCH = 2

QK_TRANSFER_SLOTS = QK_PRE_LAUNCH + 1

QK_KV_READY_EVENT = 0

QK_SCORE_READY_EVENT = 1

QK_PROB_READY_EVENT = 2

QK_PV_READY_EVENT = 3

# 性能版取 128，与上游一致。精度版取 512 是为了复刻 Native A3 每 512 个候选更新一次
# softmax 的舍入节奏，块内再用一个常驻 FP32 累加器把四个 128 子块串起来；本版本放弃
# 该性质。
# 派生量与上游对齐：SPARSE_BLOCKS = 1 + ceil(512/128) = 5、PADDED_TOPK = 640，
# 与上游 max(2, ceil((WIN+CMP_TOPK)/128)) = 5 得到的 640 完全一致。
ATTN_K_TILE = 128

NUM_QK_CORES = 24  # qk_pv dispatch lanes

# 提前发布KV在每核至少6个query的先导中减少了核内时间；较小工作量
# 尚无本体收益，保留原顺序。阈值来自单卡实测，不改变attention算术。
EARLY_KV_MIN_TOKENS = NUM_QK_CORES * 6

CSA_PLAN_WORKERS = 16  # csa_slots_build_valid_qk_plan token-tile lanes

T_PAD = ((T + 16 - 1) // 16) * 16  # Cube M floor

ATTENTION_PUBLISH_WORKERS = 48

ATTENTION_PUBLISH_T_TILE = 4

LOCAL_O_GROUPS = O_GROUPS // TP

GROUP_T_PAD = TP * T_PAD

ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD

PUBLISH_GROUPS = H_TILE // HEADS_PER_GROUP

# 每块正好一个请求：上游 e68e091 把 RoPE 行块绑定到 S，S=6 时 t_dim=batch*S 整除，
# 不再有尾块。原值 8 与 S 无关，batch*6 常不是 8 的倍数，每轮都要处理残块。
ROPE_CS_T_TILE = S
ROPE_CS_WORKERS = 16

TOPK = WIN + CMP_TOPK

SPARSE_BLOCKS = 1 + (CMP_TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE

MASK_LINE_ELEMS = 64 // 4

VALID_BLOCK_MASK_COLS = ((SPARSE_BLOCKS + MASK_LINE_ELEMS - 1) // MASK_LINE_ELEMS) * MASK_LINE_ELEMS

PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE

SWA_TILE_WIN_ROWS = min(ATTN_K_TILE, WIN)

SWA_RUNS = (SWA_TILE_WIN_ROWS + 2 * (BLOCK_SIZE - 1)) // BLOCK_SIZE

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
    seq_lens: pl.Tensor[[B_DYN], pl.INT32],
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
            # 压缩索引这段按整个 BIAS_T_TILE 一次算完（上游写法）。原先逐 token 做
            # [1, IDX_TOPK]，向量操作条数是现在的 BIAS_T_TILE 倍、每条只有 1/8 宽，
            # 实测该任务 2.53x 上游。SWA 那段仍需逐 token，因为它要按 token 读
            # position_ids 与页表；这里只把能向量化的部分提出来。
            # runtime token 数不保证整除 8（如 B1/S6）。切片不会自动截断，
            # 显式限制读写行数，避免尾块覆盖相邻 scratch。
            c_raw_tile = pl.cast(
                pl.slice(idx_topk, [BIAS_T_TILE, IDX_TOPK], [bias_t0, 0],
                         valid_shape=[bias_rows, IDX_TOPK], clamp=True), target_type=pl.FP32
            )
            c_pos = pl.cast(
                pl.slice(position_ids, [BIAS_T_TILE, 1], [bias_t0, 0],
                         valid_shape=[bias_rows, 1], clamp=True),
                target_type=pl.FP32,
            )
            c_pos_q = pl.cast(
                pl.cast(pl.mul(pl.add(c_pos, 1.0), COMPRESS_RATIO_INV), target_type=pl.INT32, mode="trunc"),
                target_type=pl.FP32,
            )
            c_upper_b = pl.row_expand_mul(
                pl.full([BIAS_T_TILE, IDX_TOPK], dtype=pl.FP32, value=1.0), c_pos_q
            )
            c_ge_tile = pl.minimum(pl.maximum(pl.add(c_raw_tile, CSA_CMP_GE_BIAS), 0.0), 1.0)
            c_lt_tile = pl.minimum(pl.maximum(pl.sub(c_upper_b, c_raw_tile), 0.0), 1.0)
            c_mask_tile = pl.mul(c_ge_tile, c_lt_tile)
            c_out_tile = pl.sub(pl.mul(c_mask_tile, pl.add(c_raw_tile, 1.0)), 1.0)
            cmp_sparse_indices = pl.assemble(
                cmp_sparse_indices,
                pl.set_validshape(pl.cast(c_out_tile, target_type=pl.INT32), bias_rows, IDX_TOPK),
                [bias_t0, 0],
            )
            sparse_bias = pl.assemble(
                sparse_bias,
                pl.set_validshape(pl.mul(pl.minimum(c_out_tile, 0.0), -NEG_INF), bias_rows, CMP_TOPK),
                [bias_t0, ATTN_K_TILE],
            )
            # 每块的有效位直接由 c_mask_tile 归约得到，不再回读 cmp_sparse_indices。
            for c_sb in pl.range(1, SPARSE_BLOCKS):
                c_s0 = (c_sb - 1) * ATTN_K_TILE
                c_blk_valid = pl.row_max(pl.slice(
                    c_mask_tile, [BIAS_T_TILE, ATTN_K_TILE], [0, c_s0],
                    valid_shape=[bias_rows, ATTN_K_TILE],
                ))
                for c_dt in pl.range(bias_rows):
                    c_valid = pl.cast(pl.read(c_blk_valid, [c_dt, 0]), pl.INT32)
                    pl.write(valid_block_mask, [bias_t0 + c_dt, c_sb], c_valid)
            # ---- 滑窗有效位：常见路径整块算 -------------------------------
            # 每 token 的窗口长度 v_length = min(position+1, WIN)，补位请求压到 0。
            # 有效位的主体就是 j < v_length[t]，与页表无关，可以一次算出
            # [BIAS_T_TILE, WIN]。原来逐 token 再逐 SWA_RUNS 做 [1, WIN]，
            # 一个 tile 要 BIAS_T_TILE * SWA_RUNS = 40 组窄向量运算。
            v_columns = pl.cast(pl.arange(0, [1, WIN], dtype=pl.INT32), pl.FP32)
            v_len_col = pl.create_tensor([BIAS_T_TILE, 1], dtype=pl.FP32)
            for bias_len_dt in pl.range(BIAS_T_TILE):
                v_len_t = bias_t0 + pl.min(bias_len_dt, bias_rows - 1)
                v_len_request = v_len_t // S
                v_len_position = pl.cast(pl.read(position_ids, [v_len_t, 0]), pl.INDEX)
                v_len_value = pl.min(v_len_position + 1, WIN)
                # 补位请求的 position 是上一步残留，用它推出的页表列号可能越界。
                # Native 把补位请求的 seq_lens 清零，真实 decode 请求恒 >= S。
                # 窗口长度压到 0 后该 token 的 SWA 偏置全是 NEG_INF，
                # attention 退化为只剩 sink，输出有限且与真实请求无关。
                if pl.read(seq_lens, [v_len_request]) <= 0:
                    v_len_value = pl.cast(0, pl.INDEX)
                pl.write(v_len_col, [bias_len_dt, 0], pl.cast(pl.cast(v_len_value, pl.INT32), pl.FP32))
            v_col_tile = pl.col_expand_mul(
                pl.full([BIAS_T_TILE, WIN], dtype=pl.FP32, value=1.0), v_columns
            )
            # clamp(v_length[t] - j, 0, 1)：j < v_length[t] 时为 1
            v_valid_tile = pl.minimum(
                pl.maximum(pl.neg(pl.row_expand_sub(v_col_tile, v_len_col)), 0.0), 1.0
            )
            sparse_bias = pl.assemble(
                sparse_bias,
                pl.set_validshape(pl.mul(pl.sub(v_valid_tile, 1.0), -NEG_INF), bias_rows, WIN),
                [bias_t0, 0],
            )
            if WIN < ATTN_K_TILE:
                sparse_bias = pl.assemble(
                    sparse_bias,
                    pl.set_validshape(
                        pl.full([BIAS_T_TILE, ATTN_K_TILE - WIN], dtype=pl.FP32, value=NEG_INF),
                        bias_rows, ATTN_K_TILE - WIN,
                    ),
                    [bias_t0, WIN],
                )

            for bias_dt in pl.range(bias_rows):
                bias_t = bias_t0 + bias_dt
                bias_request = bias_t // S
                c_position = pl.cast(pl.read(position_ids, [bias_t, 0]), pl.INDEX)
                v_length = pl.min(c_position + 1, WIN)
                if pl.read(seq_lens, [bias_request]) <= 0:
                    v_length = pl.cast(0, pl.INDEX)
                v_start = c_position - v_length + 1
                v_head = v_start % BLOCK_SIZE
                # 逐页只做两件事：本 token 有没有任何有效页，以及有没有空洞。
                # Match the Native page-valid window, including short histories
                # and negative page entries. Intervals from distinct pages do not overlap.
                v_block_valid = pl.cast(0, pl.INT32)
                v_has_hole = pl.cast(0, pl.INT32)
                for v_run in pl.range(SWA_RUNS):
                    v_lo = pl.max(v_run * BLOCK_SIZE - v_head, 0)
                    v_hi = pl.min((v_run + 1) * BLOCK_SIZE - v_head, v_length)
                    if v_hi > v_lo:
                        v_page = pl.read(ori_block_table, [bias_request, (v_start + v_lo) // BLOCK_SIZE])
                        if v_page >= 0:
                            v_block_valid = pl.cast(1, pl.INT32)
                        else:
                            v_has_hole = pl.cast(1, pl.INT32)
                pl.write(valid_block_mask, [bias_t, 0], v_block_valid)

                # 只有页表真有空洞时才逐 run 精算这一行，覆盖整块算出的结果。
                if v_has_hole == 1:
                    v_columns_row = pl.cast(pl.arange(0, [1, WIN], dtype=pl.INT32), pl.FP32)
                    v_valid = pl.full([1, WIN], dtype=pl.FP32, value=0.0)
                    for v_hole_run in pl.range(SWA_RUNS):
                        v_hole_lo = pl.max(v_hole_run * BLOCK_SIZE - v_head, 0)
                        v_hole_hi = pl.min((v_hole_run + 1) * BLOCK_SIZE - v_head, v_length)
                        if v_hole_hi > v_hole_lo:
                            v_hole_page = pl.read(
                                ori_block_table, [bias_request, (v_start + v_hole_lo) // BLOCK_SIZE]
                            )
                            if v_hole_page >= 0:
                                v_lo_fp32 = pl.cast(pl.cast(v_hole_lo, pl.INT32), pl.FP32)
                                v_hi_fp32 = pl.cast(pl.cast(v_hole_hi, pl.INT32), pl.FP32)
                                v_ge = pl.minimum(
                                    pl.maximum(pl.add(pl.sub(v_columns_row, v_lo_fp32), 1.0), 0.0), 1.0
                                )
                                v_lt = pl.minimum(
                                    pl.maximum(pl.add(pl.neg(v_columns_row), v_hi_fp32), 0.0), 1.0
                                )
                                v_valid = pl.add(v_valid, pl.mul(v_ge, v_lt))
                    sparse_bias[bias_t : bias_t + 1, 0:WIN] = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)

    # QK/PV scratch tensors.
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [t_heads, HEAD_DIM])
    attn_sink_col = pl.reshape(attn_sink, [H, 1])
    attn_mi = pl.create_tensor([t_heads, 1], dtype=pl.FP32)
    attn_li = pl.create_tensor([t_heads, 1], dtype=pl.FP32)
    attn_oi = pl.create_tensor([t_heads, HEAD_DIM], dtype=pl.FP32)

    # 每核 QK_TRANSFER_SLOTS 个轮转槽：AIV 采下一块 KV 的同时，AIC 还能算本块 QK
    # 与更早一块的 PV。单槽时三者只能首尾相接，泳道上就是 qk_pv 单次 246.8us
    # 对上游 140.2us 的来源。
    transfer_slots = NUM_QK_CORES * QK_TRANSFER_SLOTS
    transfer_heads = transfer_slots * H
    transfer_kv_rows = transfer_slots * ATTN_K_TILE
    kv_transfer = pl.create_tensor([transfer_kv_rows, HEAD_DIM], dtype=pl.BF16)
    score_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.FP32)
    probability_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.BF16)
    pv_transfer = pl.create_tensor([transfer_heads, HEAD_DIM], dtype=pl.FP32)
    mi_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    li_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
    with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[qk_plan_tid], allow_early_resolve=True) as qk_tid:
        qk_core = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace)
        for qk_t in pl.range(qk_core, t_dim, NUM_QK_CORES):
            qk_b = qk_t // S
            qk_q = pl.load(q_flat, [qk_t * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            # KV 只进一次 L1：QK 用它的转置视图，PV 用同一份的行切片。
            # 原先 QK 与 PV 各自从 GM 搬一遍，同一块 KV 过两次 GM->L1。
            qk_l1 = pl.create_tile(
                [QK_TRANSFER_SLOTS * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat
            )
            for qk_tick in pl.range(SPARSE_BLOCKS + QK_PRE_LAUNCH):
                if qk_tick < SPARSE_BLOCKS:
                    qk_sb = qk_tick
                    if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                        qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                        qk_kv_row = qk_slot * ATTN_K_TILE
                        qk_transfer_row = qk_slot * H
                        pl.system.sync_wait(QK_KV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        qk_l1_row = (qk_sb % QK_TRANSFER_SLOTS) * ATTN_K_TILE
                        qk_l1 = pl.gather_row(
                            qk_l1, kv_transfer, [qk_l1_row, 0], [qk_kv_row, 0], [ATTN_K_TILE, HEAD_DIM]
                        )
                        qk_l1_t = pl.tile.transpose_view(qk_l1)
                        qk_kv_t = pl.tile.slice(qk_l1_t, [HEAD_DIM, ATTN_K_TILE], [0, qk_l1_row])
                        qk_scores = pl.matmul(qk_q, qk_kv_t, out_dtype=pl.FP32)
                        pl.store(qk_scores, [qk_transfer_row, 0], score_transfer)
                        pl.system.sync_set(
                            QK_SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
                        )
                if qk_tick >= QK_PRE_LAUNCH:
                    pv_sb = qk_tick - QK_PRE_LAUNCH
                    if pl.read(valid_block_mask, [qk_t, pv_sb]) > 0:
                        pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                        pv_transfer_row = pv_slot * H
                        pl.system.sync_wait(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                        pv_probability = pl.load(
                            probability_transfer, [pv_transfer_row, 0], [H, ATTN_K_TILE],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv_l1_row = (pv_sb % QK_TRANSFER_SLOTS) * ATTN_K_TILE
                        pv_kv = pl.tile.slice(qk_l1, [ATTN_K_TILE, HEAD_DIM], [pv_l1_row, 0])
                        pv_output = pl.matmul(pv_probability, pv_kv, out_dtype=pl.FP32)
                        pl.store(pv_output, [pv_transfer_row, 0], pv_transfer)
                        pl.system.sync_set(
                            QK_PV_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
                        )

            for qk_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(ffts_workspace)
                qk_lane_head = qk_aiv * (H // 2)
                qk_lane_kv = qk_aiv * (ATTN_K_TILE // 2)
                qk_reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                running_m = pl.load(attn_sink_col, [qk_lane_head, 0], [H // 2, 1])
                # 上游口径：l 从 0 起算，sink 的那一项留到 merge_norm 的分母里补。
                running_l = pl.tile.muls(running_m, 0.0)
                running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                running_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                for qk_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                    SPARSE_BLOCKS + QK_PRE_LAUNCH + 1,
                    init_values=(running_m, running_l, running_left, running_right),
                ):
                    if qk_tick < SPARSE_BLOCKS:
                        qk_sb = qk_tick
                        if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                            qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                            qk_kv_row = qk_slot * ATTN_K_TILE
                            qk_kv_half = pl.tile.full([ATTN_K_TILE // 2, HEAD_DIM], dtype=pl.BF16, value=0.0)
                            if qk_sb == 0:
                                qk_pos = pl.cast(pl.read(position_ids, [qk_t, 0]), pl.INDEX)
                                qk_win_len = pl.min(qk_pos + 1, WIN)
                                qk_win_start = qk_pos - qk_win_len + 1
                                qk_head = qk_win_start % BLOCK_SIZE
                                qk_rows = pl.min(pl.max(qk_win_len - qk_lane_kv, 0), ATTN_K_TILE // 2)
                                for qk_run in pl.unroll(SWA_RUNS):
                                    qk_lo = pl.max(qk_run * BLOCK_SIZE - qk_head - qk_lane_kv, 0)
                                    qk_hi = pl.min((qk_run + 1) * BLOCK_SIZE - qk_head - qk_lane_kv, qk_rows)
                                    if qk_hi > qk_lo:
                                        qk_absolute = qk_win_start + qk_lane_kv + qk_lo
                                        qk_raw_page = pl.read(ori_block_table, [qk_b, qk_absolute // BLOCK_SIZE])
                                        if qk_raw_page >= 0:
                                            qk_raw_row = (
                                                pl.cast(qk_raw_page, pl.INDEX) * BLOCK_SIZE + qk_absolute % BLOCK_SIZE
                                            )
                                            qk_kv_half = pl.gather_row(
                                                qk_kv_half, ori_kv_flat, [qk_lo, 0], [qk_raw_row, 0],
                                                [ATTN_K_TILE // 2, HEAD_DIM],
                                                valid_shape=[qk_hi - qk_lo, HEAD_DIM],
                                            )
                            else:
                                for qk_row in pl.range(ATTN_K_TILE // 2):
                                    qk_cmp_k = (qk_sb - 1) * ATTN_K_TILE + qk_lane_kv + qk_row
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
                            pl.store(qk_kv_half, [qk_kv_row + qk_lane_kv, 0], kv_transfer)
                    # Consume the preceding Score notification before allowing
                    # the next QK to publish. KV is already written, so release
                    # the Cube before running the preceding block's softmax.
                    if t_dim >= EARLY_KV_MIN_TOKENS:
                        if qk_tick > 0 and qk_tick <= SPARSE_BLOCKS:
                            if pl.read(valid_block_mask, [qk_t, qk_tick - 1]) > 0:
                                pl.system.sync_wait(
                                    QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                                )
                        if qk_tick < SPARSE_BLOCKS:
                            if pl.read(valid_block_mask, [qk_t, qk_tick]) > 0:
                                pl.system.sync_set(
                                    QK_KV_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                                )
                    if qk_tick > 0 and qk_tick <= SPARSE_BLOCKS:
                        softmax_sb = qk_tick - 1
                        if pl.read(valid_block_mask, [qk_t, softmax_sb]) > 0:
                            qk_slot = qk_core * QK_TRANSFER_SLOTS + softmax_sb % QK_TRANSFER_SLOTS
                            qk_transfer_row = qk_slot * H
                            qk_s0 = softmax_sb * ATTN_K_TILE
                            if t_dim < EARLY_KV_MIN_TOKENS:
                                pl.system.sync_wait(
                                    QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                                )
                            # 整条 H//2 一次做完：原先按 SOFTMAX_HEAD_TILE=8 切成四段，
                            # 是 ATTN_K_TILE=512 时 UB 放不下留下的，现在 128 宽已无必要。
                            qk_scores_half = pl.load(
                                score_transfer, [qk_transfer_row + qk_lane_head, 0], [H // 2, ATTN_K_TILE]
                            )
                            qk_bias = pl.load(sparse_bias, [qk_t, qk_s0], [1, ATTN_K_TILE])
                            qk_masked = pl.col_expand_add(pl.mul(qk_scores_half, SOFTMAX_SCALE), qk_bias)
                            # 块内取局部最大值，跨块的重标定挪到下面的归并里；
                            # 这样每块 softmax 互不依赖，流水才排得开。
                            qk_mi = pl.row_max(qk_masked, qk_reduce_tmp)
                            qk_exp = pl.exp(pl.row_expand_sub(qk_masked, qk_mi))
                            qk_li = pl.row_sum(qk_exp, qk_reduce_tmp)
                            # 性能版用 rint（就近偶数），与上游一致。精度版用 round
                            # 是为了复刻 Native SAS 的 CAST_ROUND——半数远离零。
                            qk_probability = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                            pl.store(qk_probability, [qk_transfer_row + qk_lane_head, 0], probability_transfer)
                            pl.store(qk_mi, [qk_transfer_row + qk_lane_head, 0], mi_transfer)
                            pl.store(qk_li, [qk_transfer_row + qk_lane_head, 0], li_transfer)
                            pl.system.sync_set(
                                QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                            )
                    if t_dim < EARLY_KV_MIN_TOKENS:
                        if qk_tick < SPARSE_BLOCKS:
                            if pl.read(valid_block_mask, [qk_t, qk_tick]) > 0:
                                pl.system.sync_set(
                                    QK_KV_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                                )
                    if qk_tick >= QK_PRE_LAUNCH + 1:
                        pv_sb = qk_tick - QK_PRE_LAUNCH - 1
                        if pl.read(valid_block_mask, [qk_t, pv_sb]) > 0:
                            pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                            pv_transfer_row = pv_slot * H
                            pl.system.sync_wait(QK_PV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            pv_m = pl.load(mi_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1])
                            pv_l = pl.load(li_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1])
                            next_m = pl.maximum(m_iter, pv_m)
                            alpha = pl.exp(pl.sub(m_iter, next_m))
                            beta = pl.exp(pl.sub(pv_m, next_m))
                            next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, pv_l))
                            pv_left = pl.load(
                                pv_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, HEAD_DIM // 2]
                            )
                            next_left = pl.add(pl.row_expand_mul(left_iter, alpha), pl.row_expand_mul(pv_left, beta))
                            pv_right = pl.load(
                                pv_transfer, [pv_transfer_row + qk_lane_head, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2]
                            )
                            next_right = pl.add(pl.row_expand_mul(right_iter, alpha), pl.row_expand_mul(pv_right, beta))
                            m_valid, l_valid, left_valid, right_valid = pl.yield_(
                                next_m, next_l, next_left, next_right
                            )
                        else:
                            m_valid, l_valid, left_valid, right_valid = pl.yield_(
                                m_iter, l_iter, left_iter, right_iter
                            )
                        m_after, l_after, left_after, right_after = pl.yield_(
                            m_valid, l_valid, left_valid, right_valid
                        )
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
    # 不再依赖 rope_swap：本任务自己重算符号表（见下面的 cs_lane / cs_sign），
    # 从不读那张 GM 索引表，原来的 deps 是一条假依赖。
    with pl.spmd(pl.min(rope_cs_blocks, ROPE_CS_WORKERS), name_hint="rope_cs",
                 allow_early_resolve=True) as rope_tid:
        for cs_rb in pl.range(pl.tile.get_block_idx(), rope_cs_blocks,
                              pl.min(rope_cs_blocks, ROPE_CS_WORKERS)):
            cs_t0 = cs_rb * ROPE_CS_T_TILE
            cs_rows = pl.min(ROPE_CS_T_TILE, t_dim - cs_t0)
            # 符号表按块重算：SPMD 下每个 worker 要有自己的一份，不能在区外共享。
            cs_ones = pl.tile.full([ROPE_CS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
            cs_idx_f = pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
            cs_col = pl.col_expand_mul(cs_ones, cs_idx_f)
            cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
            cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
            cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
            cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
            cs_sin = pl.load(freqs_sin, [cs_t0, 0], [ROPE_CS_T_TILE, ROPE_DIM],
                             valid_shape=[cs_rows, ROPE_DIM])
            # tile 绑定到 S 后 cs_rows 恒等于 ROPE_CS_T_TILE，保留 valid_shape 只作兜底。
            cs_sign_rows = pl.set_validshape(cs_sign, cs_rows, ROPE_DIM)
            pl.store(pl.mul(cs_sin, cs_sign_rows), [cs_t0, 0], rope_sin_signed)

    return (
        attn_mi,
        attn_li,
        attn_oi,
        freqs_cos,
        rope_sin_signed,
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
    seq_lens: pl.Tensor[[B_DYN], pl.INT32],
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
        seq_lens,
        attn_sink,
        freqs_cos,
        freqs_sin,
    )
    t_dim = pl.tensor.dim(q, 0)
    merge_sink = pl.reshape(attn_sink, [H, 1])

    with pl.spmd(MERGE_WORKERS, name_hint="merge_norm", deps=[qk_tid, rope_tid]) as merge_tid:
        m_worker = pl.tile.get_block_idx()
        # 换算索引（j^1 的 lane swap）就地算：它与 t_dim 无关，是纯常量表。
        # 原先由一个 CORE_GROUP 的 rope_swap 任务算好写进 GM，本任务再读回来——
        # 关键路径实测那个任务 compute 只有 1.7us 却要 core-wait 72.1us（占 makespan
        # 8.95%），因为它独占一个核、要等核空出来。本任务是 SPMD，每个 worker 自己
        # 算这几条向量指令即可，与 rope_cs 里的做法一致。算式与原来逐字相同。
        # lane swap 的索引就是 `j ^ 1`（偶数 +1、奇数 -1），一条 xors 即可，不必绕 FP32。
        # 原先用 13 步 FP32（含 5 次 cast）算同一张表，SPMD 下**每个 block 都要重算
        # 一遍**，于是整段进了 `local_setup_us`：实测 8.15µs/块，而上游从 GM 读表只要
        # 0.54µs，48 块合计多 365 核·µs。in-core 也印证过这些指令本身很便宜
        # （读表版 3.31µs vs 就地算 3.30µs，纯计算几乎不变），贵在落进了启动路径。
        # 改成纯 INT32 之后 9 步、零 cast，且仍然不需要 rope_swap 那个独占一核的任务。
        # 用整数取模算 `j + 1 - 2*(j % 2)`，与 `j ^ 1` 等值（偶数 +1、奇数 -1），全程
        # INT32、零 cast。
        # 试过两条更短的路都被硬件挡了，记下来免得再走：① `pl.tile.xors` 一步到位，
        # 但 A2/A3 的 `pto.txors` 只接受 i8/i16 元素类型，i32 被 ptoas 拒；② 退成 i16
        # 做异或再 cast 回 i32，而 `LegalizeTileCast` 说 a2a3 上 int16→int32 没有原生
        # 路径（`pto.tcvt` 不支持）。取模这条虽然多两步，但仍是 INT32 域内、无 cast。
        m_idx = pl.tile.ci(0, [1, ROPE_DIM], dtype=pl.INT32)
        m_rem_tmp = pl.create_tile([1, ROPE_DIM], dtype=pl.INT32)
        m_lane = pl.tile.rems(m_idx, 2, m_rem_tmp)
        m_swap_row = pl.tile.adds(
            pl.tile.sub(m_idx, pl.tile.muls(m_lane, 2)), NOPE_DIM + 1
        )
        m_swap_base = pl.create_tile([H_TILE, ROPE_DIM], dtype=pl.INT32)
        m_swap_source = pl.col_expand(m_swap_base, m_swap_row)
        m_row_offsets = pl.tile.muls(pl.tile.ci(0, [1, H_TILE], dtype=pl.INT32), HEAD_DIM)
        m_swap_idx = pl.row_expand_add(m_swap_source, pl.reshape(m_row_offsets, [H_TILE, 1]))
        m_gather_tmp = pl.create_tile([H_TILE, ROPE_DIM], dtype=pl.INT32)
        for m_idx in pl.range(m_worker, t_dim * (H // H_TILE), MERGE_WORKERS):
            m_t = m_idx // (H // H_TILE)
            m_h_idx = m_idx - m_t * (H // H_TILE)
            m_h0 = m_h_idx * H_TILE
            m_row = m_idx * H_TILE
            m_mi = pl.load(attn_mi, [m_row, 0], [H_TILE, 1])
            m_li = pl.load(attn_li, [m_row, 0], [H_TILE, 1])
            m_oi = pl.load(attn_oi, [m_row, 0], [H_TILE, HEAD_DIM])

            # qk_pv 的 l 从 0 起算，sink 那一项在这里按当前最大值补进分母。
            n_sink_bias = pl.load(merge_sink, [m_h0, 0], [H_TILE, 1])
            n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
            n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
            n_full = pl.row_expand_div(m_oi, n_denom)
            n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

            # Inverse-RoPE head tile. 性能版直接用 FP32 结果做逆 RoPE；精度版先过一次
            # BF16 再回 FP32，是为了复刻 Native 在逆 RoPE 前先发布 BF16 的次序。
            m_rope = n_full[0:H_TILE, NOPE_DIM:HEAD_DIM]
            m_cos_il = pl.load(rope_cos_il, [m_t, 0], [1, ROPE_DIM])
            m_sin_signed = pl.load(rope_sin_signed, [m_t, 0], [1, ROPE_DIM])
            m_swapped = pl.tile.gather(n_full, m_swap_idx, m_gather_tmp)
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
