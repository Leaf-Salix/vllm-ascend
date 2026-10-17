# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HCA 注意力：参考 pypto-lib TP1 流水，直接读取 Native 页表、位置和 RoPE。"""


import pypto.language as pl

from ..deepseek_v4_flash_csa.config import (
    BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    KV_ORI_BLOCK_NUM,
    TP,
)
from ..deepseek_v4_flash_csa.config import (
    FLASH as M,
)

CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE
HCA_KV_CMP_BLOCK_NUM = 256
ORI_TABLE_COLUMNS_DYN = pl.dynamic("HCA_ORIGINAL_TABLE_COLUMNS")


B_DYN = pl.dynamic("B_DYN")
T_DYN = pl.dynamic("T_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
CMP_TABLE_BLOCKS_DYN = pl.dynamic("CMP_TABLE_BLOCKS_DYN")


B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
SOFTMAX_SCALE = M.softmax_scale
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

COMPRESS_RATIO = 128
NEG_INF = -1.0e20


ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
CMP_BLOCK_NUM = HCA_KV_CMP_BLOCK_NUM



HCA_MAX_COMPRESSED_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO
HCA_COMPRESSED_POOL_ROWS = CMP_BLOCK_NUM * CMP_STORAGE_BLOCK_SIZE


VALID_TOKEN_TILE = 8
GATHER_RUN_TILE = 16
REQUEST_KV_ROWS = WIN + S - 1
ATTN_K_TILE = 128
RAW_K_TILE = ATTN_K_TILE
PV_N_TILE = 128
RAW_WORKERS = 20
H_TILE = 16
NUM_QK_CORES = 24
QK_PRE_LAUNCH = 2
CMP_QUERY_LOOKAHEAD = 2
QK_TRANSFER_SLOTS = QK_PRE_LAUNCH + 1
QK_SCORE_READY_EVENT = 0
QK_PROB_READY_EVENT = 1
QK_PV_READY_EVENT = 2
CMP_ATTN_K_TILE = 128 if TP == 1 else 32
CMP_PAGES_PER_WORK = CMP_ATTN_K_TILE // CMP_STORAGE_BLOCK_SIZE
CMP_GATHER_WORK_TILE = max(1, 8 // CMP_PAGES_PER_WORK)
CMP_GATHER_AIV_WAVE = 48  # 一个 AIV 波的块数；长档 gather 的块数按它反推
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
ROPE_CS_T_TILE = S
T_PAD = ((T + 16 - 1) // 16) * 16
MERGE_WORKERS = 48
ATTENTION_PUBLISH_WORKERS = 48
ATTENTION_PUBLISH_T_TILE = 4
LOCAL_O_GROUPS = O_GROUPS // TP
GROUP_T_PAD = TP * T_PAD
ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD
PUBLISH_GROUPS = H_TILE // HEADS_PER_GROUP

if WIN != RAW_K_TILE:
    raise ValueError("HCA raw attention evaluates the window in one tile; WIN must equal RAW_K_TILE")
if HCA_MAX_COMPRESSED_ROWS > HCA_COMPRESSED_POOL_ROWS:
    raise ValueError("HCA compressed rows exceed the configured pool")
if CMP_ATTN_K_TILE % CMP_STORAGE_BLOCK_SIZE != 0:
    raise ValueError("HCA work must contain complete cache pages")
if BLOCK_SIZE % GATHER_RUN_TILE != 0:
    raise ValueError("a contiguous gather run must stay inside one cache block")
if S % ROPE_CS_T_TILE != 0:
    raise ValueError("each request must contain complete inverse-RoPE token tiles")
if H_TILE % HEADS_PER_GROUP != 0:
    raise ValueError(f"HCA head tile {H_TILE} must contain complete output groups")
if PUBLISH_GROUPS != 2:
    raise ValueError("HCA merge pack expects two output groups per head tile")
if O_GROUPS % TP != 0:
    raise ValueError(f"output groups {O_GROUPS} must be divisible by TP size {TP}")
if LOCAL_O_GROUPS % PUBLISH_GROUPS != 0:
    raise ValueError("local output groups must contain complete HCA publish tiles")
if T % ATTENTION_PUBLISH_T_TILE != 0:
    raise ValueError("local token capacity must contain complete attention publish tiles")


@pl.jit.inline(auto_scope=False)
def _cmp_query_kv(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    token: pl.Scalar[pl.INDEX],
):
    query_3d = pl.load(q, [token, 0, 0], [1, H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
    query = pl.reshape(query_3d, [H, HEAD_DIM])
    kv = pl.create_tile([QK_TRANSFER_SLOTS * CMP_ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
    return query, kv


@pl.jit.inline(auto_scope=False)
def sparse_attn_hca(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B_DYN, ORI_TABLE_COLUMNS_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    ori_cache_ready_dep: pl.Scalar[pl.TASK_ID],
    cmp_cache_ready_dep: pl.Scalar[pl.TASK_ID],
):
    """分别计算原始滑窗、C128 压缩历史的在线 softmax 状态。"""
    t_dim = pl.tensor.dim(q, 0)
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_table_blocks = pl.tensor.dim(cmp_block_table, 1)
    cmp_work_count = (cmp_table_blocks + CMP_PAGES_PER_WORK - 1) // CMP_PAGES_PER_WORK
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * CMP_STORAGE_BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [t_dim * H, HEAD_DIM])
    request_count = pl.tensor.dim(cmp_block_table, 0)
    raw_gather_count = request_count
    cmp_gather_count = request_count * cmp_work_count
    cmp_partial_rows = t_dim * H

    stream_state_m = pl.create_tensor([t_dim * H, 1], dtype=pl.FP32)
    stream_state_l = pl.create_tensor([t_dim * H, 1], dtype=pl.FP32)
    stream_heads = pl.create_tensor([t_dim * H, HEAD_DIM], dtype=pl.FP32)
    cmp_partial_m = pl.create_tensor([cmp_partial_rows, 1], dtype=pl.FP32)
    cmp_partial_l = pl.create_tensor([cmp_partial_rows, 1], dtype=pl.FP32)
    cmp_partial_o = pl.create_tensor([cmp_partial_rows, HEAD_DIM], dtype=pl.FP32)
    rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    raw_branch_tids = pl.array.create(1, pl.TASK_ID)
    cmp_branch_tids = pl.array.create(1, pl.TASK_ID)
    rope_cs_tids = pl.array.create(1, pl.TASK_ID)

    with pl.scope():
        raw_kv = pl.create_tensor([request_count * REQUEST_KV_ROWS, HEAD_DIM], dtype=pl.BF16)
        raw_valid = pl.create_tensor([t_dim, WIN], dtype=pl.FP32)
        # 按 Native 原始 KV 页表聚合一个请求的 128+5 行；不创建外部窗口索引。
        with pl.spmd(raw_gather_count, name_hint="hca_gather_kv", allow_early_resolve=True, deps=[ori_cache_ready_dep]) as raw_gather_tid:
            g_req = pl.tile.get_block_idx()
            g_t0 = g_req * S
            g_base = g_req * REQUEST_KV_ROWS
            g_length = pl.read(kv_seq_lens, [g_req])
            g_position = pl.read(position_ids, [g_t0])
            g_first_len = pl.min(WIN, pl.max(pl.min(g_position + 1, g_length), 0))
            g_start = pl.max(g_position + 1 - g_first_len, 0)
            raw_kv[g_base:g_base + REQUEST_KV_ROWS, :] = pl.full([REQUEST_KV_ROWS, HEAD_DIM], dtype=pl.BF16, value=0.0)
            if g_length > 0:
                # 一个 Native 页内的行物理连续；首尾页只复制有效行，不读取未初始化 KV。
                g_rows = pl.min(g_first_len + S - 1, pl.max(g_length - g_start, 0))
                g_pages = (g_start % BLOCK_SIZE + g_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
                for g_part in pl.range(g_pages):
                    g_absolute_page = g_start // BLOCK_SIZE + g_part
                    g_offset = pl.max(g_start - g_absolute_page * BLOCK_SIZE, 0)
                    g_row = pl.max(g_absolute_page * BLOCK_SIZE - g_start, 0)
                    g_count = pl.min(BLOCK_SIZE - g_offset, g_rows - g_row)
                    if g_count > 0:
                        g_page = pl.read(ori_block_table, [g_req, g_absolute_page])
                        if g_page >= 0 and g_page < ori_block_num:
                            g_source = pl.cast(g_page, pl.INDEX) * BLOCK_SIZE + g_offset
                            g_chunk = pl.load(
                                ori_kv_flat, [g_source, 0], [BLOCK_SIZE, HEAD_DIM],
                                valid_shape=[g_count, HEAD_DIM],
                            )
                            pl.store(g_chunk, [g_base + g_row, 0], raw_kv)

        # B*6 不一定是 8 的倍数，尾块逐实际 token 写入；seq_lens=0 排除 dummy。
        with pl.spmd((t_dim + VALID_TOKEN_TILE - 1) // VALID_TOKEN_TILE, name_hint="hca_raw_valid", allow_early_resolve=True) as raw_valid_tid:
            valid_t0 = pl.tile.get_block_idx() * VALID_TOKEN_TILE
            valid_col = pl.cast(pl.tile.arange(0, [1, WIN], dtype=pl.INT32), pl.FP32)
            for valid_dt in pl.range(pl.min(VALID_TOKEN_TILE, t_dim - valid_t0)):
                valid_token = valid_t0 + valid_dt
                valid_length = pl.min(WIN, pl.max(pl.min(
                    pl.read(position_ids, [valid_token]) + 1,
                    pl.read(kv_seq_lens, [valid_token // S]),
                ), 0))
                valid_mask = pl.minimum(pl.maximum(pl.neg(pl.sub(valid_col, pl.cast(valid_length, pl.FP32))), 0.0), 1.0)
                pl.store(valid_mask, [valid_token, 0], raw_valid)

        # Native 已提供交错的 FP32 频率；inverse RoPE 仅在消费者内折叠符号。
        with pl.spmd(rope_cs_blocks, name_hint="hca_inverse_rope_sign", allow_early_resolve=True) as rope_cs_tid:
            cs_t0 = pl.tile.get_block_idx() * ROPE_CS_T_TILE
            cs_index = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), pl.FP32)
            cs_pair = pl.cast(pl.cast(pl.mul(cs_index, 0.5), pl.INT32, mode="trunc"), pl.FP32)
            cs_odd = pl.sub(cs_index, pl.mul(cs_pair, 2.0))
            cs_sign = pl.neg(pl.sub(pl.mul(cs_odd, 2.0), 1.0))
            rope_cos_il[cs_t0:cs_t0 + ROPE_CS_T_TILE, :] = freqs_cos[cs_t0:cs_t0 + ROPE_CS_T_TILE, :]
            rope_sin_signed[cs_t0:cs_t0 + ROPE_CS_T_TILE, :] = pl.col_expand_mul(
                freqs_sin[cs_t0:cs_t0 + ROPE_CS_T_TILE, :], cs_sign,
            )


        raw_transfer_rows = RAW_WORKERS * QK_TRANSFER_SLOTS * H
        raw_score_transfer = pl.create_tensor([raw_transfer_rows, ATTN_K_TILE], dtype=pl.FP32)
        raw_probability_transfer = pl.create_tensor([raw_transfer_rows, ATTN_K_TILE], dtype=pl.BF16)
        raw_ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)

        with pl.spmd(RAW_WORKERS, name_hint="hca_raw_attn", deps=[raw_gather_tid, raw_valid_tid], allow_early_resolve=True) as raw_heads_tid:
            raw_qk_task = pl.tile.get_block_idx()
            pl.system.set_ffts(raw_ffts_workspace)
            raw_qk_count = pl.max((t_dim - raw_qk_task + RAW_WORKERS - 1) // RAW_WORKERS, 0)
            raw_kv_l1 = pl.create_tile([QK_TRANSFER_SLOTS * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
            for raw_qk_tick in pl.range(raw_qk_count + QK_PRE_LAUNCH):
                if raw_qk_tick < raw_qk_count:
                    raw_qk_t = raw_qk_task + raw_qk_tick * RAW_WORKERS
                    raw_qk_slot = raw_qk_task * QK_TRANSFER_SLOTS + raw_qk_tick % QK_TRANSFER_SLOTS
                    raw_qk_row = raw_qk_slot * H
                    raw_qk_request = raw_qk_t // S
                    raw_qk_token = raw_qk_t % S
                    raw_qk_first_len = pl.min(WIN, pl.max(pl.min(
                        pl.read(position_ids, [raw_qk_request * S]) + 1,
                        pl.read(kv_seq_lens, [raw_qk_request]),
                    ), 0))
                    raw_qk_drop = pl.max(raw_qk_first_len + raw_qk_token - WIN, 0)
                    raw_qk_base = raw_qk_request * REQUEST_KV_ROWS + raw_qk_drop
                    raw_qk_q = pl.load(q_flat, [raw_qk_t * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                    raw_qk_l1_row = (raw_qk_tick % QK_TRANSFER_SLOTS) * ATTN_K_TILE
                    raw_kv_l1 = pl.gather_row(raw_kv_l1, raw_kv, [raw_qk_l1_row, 0], [raw_qk_base, 0], [ATTN_K_TILE, HEAD_DIM])
                    raw_kv_l1_t = pl.tile.transpose_view(raw_kv_l1)
                    raw_qk_kv_t = pl.tile.slice(raw_kv_l1_t, [HEAD_DIM, ATTN_K_TILE], [0, raw_qk_l1_row])
                    raw_qk_scores = pl.matmul(raw_qk_q, raw_qk_kv_t, out_dtype=pl.FP32)
                    pl.store(raw_qk_scores, [raw_qk_row, 0], raw_score_transfer)
                    pl.system.sync_set(QK_SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                if raw_qk_tick >= QK_PRE_LAUNCH:
                    raw_pv_item = raw_qk_tick - QK_PRE_LAUNCH
                    raw_pv_t = raw_qk_task + raw_pv_item * RAW_WORKERS
                    raw_pv_slot = raw_qk_task * QK_TRANSFER_SLOTS + raw_pv_item % QK_TRANSFER_SLOTS
                    pl.system.sync_wait(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    raw_pv_probability = pl.load(
                        raw_probability_transfer, [raw_pv_slot * H, 0], [H, ATTN_K_TILE], target_memory=pl.MemorySpace.Mat,
                    )
                    raw_pv_l1_row = (raw_pv_item % QK_TRANSFER_SLOTS) * ATTN_K_TILE
                    # N128/K128 保留原累加顺序；两份 Acc 交替，使 FIX 写回与下一块 Cube 重叠。
                    raw_pv_left = pl.tile.move(raw_pv_probability, target_memory=pl.MemorySpace.Left)
                    raw_pv_right = pl.tile.extract(
                        raw_kv_l1, raw_pv_l1_row, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right,
                    )
                    raw_pv_previous = pl.tile.matmul(raw_pv_left, raw_pv_right)
                    for raw_pv_n in pl.unroll(1, HEAD_DIM // PV_N_TILE):
                        raw_pv_next = pl.tile.extract(
                            raw_kv_l1, raw_pv_l1_row, raw_pv_n * PV_N_TILE,
                            [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right,
                        )
                        raw_pv_current = pl.tile.matmul(raw_pv_left, raw_pv_next)
                        pl.store(raw_pv_previous, [raw_pv_t * H, (raw_pv_n - 1) * PV_N_TILE], stream_heads)
                        raw_pv_previous = raw_pv_current
                    pl.store(raw_pv_previous, [raw_pv_t * H, HEAD_DIM - PV_N_TILE], stream_heads)

            for raw_qk_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(raw_ffts_workspace)
                raw_qk_head = raw_qk_aiv * (H // 2)
                raw_qk_reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                for raw_qk_item in pl.range(raw_qk_count):
                    raw_qk_t = raw_qk_task + raw_qk_item * RAW_WORKERS
                    raw_qk_slot = raw_qk_task * QK_TRANSFER_SLOTS + raw_qk_item % QK_TRANSFER_SLOTS
                    raw_qk_row = raw_qk_slot * H + raw_qk_head
                    pl.system.sync_wait(QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    raw_qk_scores_half = pl.load(raw_score_transfer, [raw_qk_row, 0], [H // 2, ATTN_K_TILE], target_memory=pl.MemorySpace.Vec)
                    raw_qk_valid_row = pl.load(raw_valid, [raw_qk_t, 0], [1, ATTN_K_TILE], target_memory=pl.MemorySpace.Vec)
                    raw_qk_bias = pl.mul(pl.sub(raw_qk_valid_row, 1.0), -NEG_INF)
                    raw_qk_scores_half = pl.col_expand_add(pl.mul(raw_qk_scores_half, SOFTMAX_SCALE), raw_qk_bias)
                    raw_qk_mi = pl.row_max(raw_qk_scores_half, raw_qk_reduce_tmp)
                    raw_qk_exp = pl.exp(pl.row_expand_sub(raw_qk_scores_half, raw_qk_mi))
                    raw_qk_exp = pl.col_expand_mul(raw_qk_exp, raw_qk_valid_row)
                    raw_qk_li = pl.row_sum(raw_qk_exp, raw_qk_reduce_tmp)
                    raw_qk_probability = pl.cast(raw_qk_exp, target_type=pl.BF16, mode="rint")
                    pl.store(raw_qk_probability, [raw_qk_row, 0], raw_probability_transfer)
                    pl.store(raw_qk_mi, [raw_qk_t * H + raw_qk_head, 0], stream_state_m)
                    pl.store(raw_qk_li, [raw_qk_t * H + raw_qk_head, 0], stream_state_l)
                    pl.system.sync_set(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)

        raw_branch_tids[0] = raw_heads_tid
        rope_cs_tids[0] = rope_cs_tid

    with pl.scope():
        cmp_work_kv = pl.create_tensor([cmp_gather_count * CMP_ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16)

        cmp_work_valid = pl.create_tensor([cmp_gather_count, CMP_ATTN_K_TILE], dtype=pl.FP32)
        cmp_gather_blocks = cmp_gather_count
        if cmp_table_blocks >= 8:
            cmp_gather_blocks = (cmp_gather_count + CMP_GATHER_WORK_TILE - 1) // CMP_GATHER_WORK_TILE
        with pl.spmd(cmp_gather_blocks, name_hint="hca_cmp_work_gather", allow_early_resolve=True, deps=[cmp_cache_ready_dep]) as cmp_gather_tid:
            gather_block = pl.tile.get_block_idx()
            gather_begin = gather_block
            gather_end = gather_block + 1
            if cmp_table_blocks >= 8:
                gather_begin = gather_block * CMP_GATHER_WORK_TILE
                gather_end = pl.min(gather_begin + CMP_GATHER_WORK_TILE, cmp_gather_count)
            for gather_item in pl.range(gather_begin, gather_end):
                gather_request = gather_item // cmp_work_count
                gather_work = gather_item - gather_request * cmp_work_count
                gather_first_col = gather_work * CMP_PAGES_PER_WORK
                gather_dst0 = gather_item * CMP_ATTN_K_TILE
                gather_tile = pl.tile.full([CMP_ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                gather_mask = pl.tile.full([1, CMP_ATTN_K_TILE], dtype=pl.FP32, value=0.0)
                # Native 不初始化有效长度之后的 KV；零概率乘 NaN 仍会污染 PV。
                # 只搬本步已生成的压缩行，未搬入的片内位置保留为零。
                gather_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.max(
                    pl.read(kv_seq_lens, [gather_request]), 0,
                ) // COMPRESS_RATIO)
                for gather_page in pl.range(CMP_PAGES_PER_WORK):
                    gather_page_col = gather_first_col + gather_page
                    gather_valid_rows = pl.min(CMP_STORAGE_BLOCK_SIZE, pl.max(
                        gather_rows - gather_page_col * CMP_STORAGE_BLOCK_SIZE, 0,
                    ))
                    if gather_page_col < cmp_table_blocks and gather_valid_rows > 0:
                        gather_page_i32 = pl.read(cmp_block_table, [gather_request, gather_page_col])
                        if gather_page_i32 >= 0:
                            if gather_page_i32 < cmp_block_num:
                                gather_page_id = pl.cast(gather_page_i32, pl.INDEX)
                                gather_src = gather_page_id * CMP_STORAGE_BLOCK_SIZE
                                gather_col = gather_page * CMP_STORAGE_BLOCK_SIZE
                                gather_tile = pl.gather_row(
                                    gather_tile, cmp_kv_flat, [gather_col, 0], [gather_src, 0],
                                    [CMP_STORAGE_BLOCK_SIZE, HEAD_DIM], valid_shape=[gather_valid_rows, HEAD_DIM],
                                )
                                for gather_mask_col in pl.unroll(CMP_STORAGE_BLOCK_SIZE):
                                    if gather_mask_col < gather_valid_rows:
                                        pl.tile.write(gather_mask, [0, gather_col + gather_mask_col], 1.0)
                pl.store(gather_tile, [gather_dst0, 0], cmp_work_kv)
                pl.store(gather_mask, [gather_item, 0], cmp_work_valid)



        attn_sink_col = pl.reshape(attn_sink, [H, 1])
        transfer_slots = NUM_QK_CORES * QK_TRANSFER_SLOTS
        transfer_heads = transfer_slots * H
        score_transfer = pl.create_tensor([transfer_heads, CMP_ATTN_K_TILE], dtype=pl.FP32)
        probability_transfer = pl.create_tensor([transfer_heads, CMP_ATTN_K_TILE], dtype=pl.BF16)
        pv_transfer = pl.create_tensor([transfer_heads, HEAD_DIM], dtype=pl.FP32)
        mi_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
        li_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
        ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
        with pl.spmd(NUM_QK_CORES, name_hint="hca_cmp_qk_pv", deps=[cmp_gather_tid], allow_early_resolve=True) as cmp_qk_tid:
            qk_core = pl.tile.get_block_idx()
            pl.system.set_ffts(ffts_workspace)
            if cmp_work_count == 1:
                fast_count = pl.max((t_dim - qk_core + NUM_QK_CORES - 1) // NUM_QK_CORES, 0)
                for fast_tick in pl.range(fast_count + CMP_QUERY_LOOKAHEAD):
                    if fast_tick < fast_count:
                        fast_t = qk_core + fast_tick * NUM_QK_CORES
                        fast_request = fast_t // S
                        fast_position = pl.max(pl.read(position_ids, [fast_t]), -1)
                        fast_length = pl.max(pl.read(kv_seq_lens, [fast_request]), 0)
                        fast_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min((fast_position + 1) // COMPRESS_RATIO, fast_length // COMPRESS_RATIO))
                        if fast_rows > 0:
                            fast_page_ok = 1
                            if CMP_PAGES_PER_WORK == 1:
                                fast_page = pl.read(cmp_block_table, [fast_request, 0])
                                fast_page_ok = pl.min(fast_page + 1, cmp_block_num - fast_page)
                            if fast_page_ok > 0:
                                fast_slot = qk_core * QK_TRANSFER_SLOTS + fast_tick % QK_TRANSFER_SLOTS
                                fast_q = pl.load(q_flat, [fast_t * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                                fast_kv = pl.load(cmp_work_kv, [fast_request * CMP_ATTN_K_TILE, 0], [CMP_ATTN_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                                fast_scores = pl.matmul(fast_q, pl.tile.transpose_view(fast_kv), out_dtype=pl.FP32)
                                pl.store(fast_scores, [fast_slot * H, 0], score_transfer)
                                pl.system.sync_set(QK_SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                    if fast_tick >= CMP_QUERY_LOOKAHEAD:
                        fast_pv_item = fast_tick - CMP_QUERY_LOOKAHEAD
                        fast_pv_t = qk_core + fast_pv_item * NUM_QK_CORES
                        fast_pv_request = fast_pv_t // S
                        fast_pv_position = pl.max(pl.read(position_ids, [fast_pv_t]), -1)
                        fast_pv_length = pl.max(pl.read(kv_seq_lens, [fast_pv_request]), 0)
                        fast_pv_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min((fast_pv_position + 1) // COMPRESS_RATIO, fast_pv_length // COMPRESS_RATIO))
                        if fast_pv_rows > 0:
                            fast_pv_page_ok = 1
                            if CMP_PAGES_PER_WORK == 1:
                                fast_pv_page = pl.read(cmp_block_table, [fast_pv_request, 0])
                                fast_pv_page_ok = pl.min(fast_pv_page + 1, cmp_block_num - fast_pv_page)
                            if fast_pv_page_ok > 0:
                                fast_pv_slot = qk_core * QK_TRANSFER_SLOTS + fast_pv_item % QK_TRANSFER_SLOTS
                                pl.system.sync_wait(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                                fast_prob = pl.load(probability_transfer, [fast_pv_slot * H, 0], [H, CMP_ATTN_K_TILE], target_memory=pl.MemorySpace.Mat)
                                fast_value = pl.load(cmp_work_kv, [fast_pv_request * CMP_ATTN_K_TILE, 0], [CMP_ATTN_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                                fast_left = pl.tile.move(fast_prob, target_memory=pl.MemorySpace.Left)
                                fast_right = pl.tile.extract(
                                    fast_value, 0, 0, [CMP_ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right,
                                )
                                fast_previous = pl.tile.matmul(fast_left, fast_right)
                                for fast_pv_n in pl.unroll(1, HEAD_DIM // PV_N_TILE):
                                    fast_next = pl.tile.extract(
                                        fast_value, 0, fast_pv_n * PV_N_TILE,
                                        [CMP_ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right,
                                    )
                                    fast_current = pl.tile.matmul(fast_left, fast_next)
                                    pl.store(fast_previous, [fast_pv_t * H, (fast_pv_n - 1) * PV_N_TILE], cmp_partial_o)
                                    fast_previous = fast_current
                                pl.store(fast_previous, [fast_pv_t * H, HEAD_DIM - PV_N_TILE], cmp_partial_o)

                for fast_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    pl.system.set_ffts(ffts_workspace)
                    fast_head = fast_aiv * (H // 2)
                    fast_tmp = pl.create_tile([H // 2, CMP_ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    for fast_item in pl.range(fast_count):
                        fast_vec_t = qk_core + fast_item * NUM_QK_CORES
                        fast_vec_request = fast_vec_t // S
                        fast_vec_position = pl.max(pl.read(position_ids, [fast_vec_t]), -1)
                        fast_vec_length = pl.max(pl.read(kv_seq_lens, [fast_vec_request]), 0)
                        fast_vec_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min((fast_vec_position + 1) // COMPRESS_RATIO, fast_vec_length // COMPRESS_RATIO))
                        fast_valid = fast_vec_rows
                        if CMP_PAGES_PER_WORK == 1:
                            fast_vec_page = pl.read(cmp_block_table, [fast_vec_request, 0])
                            fast_valid = pl.min(fast_vec_rows, pl.min(fast_vec_page + 1, cmp_block_num - fast_vec_page))
                        if fast_valid > 0:
                            fast_vec_slot = qk_core * QK_TRANSFER_SLOTS + fast_item % QK_TRANSFER_SLOTS
                            fast_transfer_row = fast_vec_slot * H + fast_head
                            pl.system.sync_wait(QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            fast_vec_scores = pl.load(score_transfer, [fast_transfer_row, 0], [H // 2, CMP_ATTN_K_TILE], target_memory=pl.MemorySpace.Vec)
                            fast_scaled = pl.mul(fast_vec_scores, SOFTMAX_SCALE)
                            if CMP_PAGES_PER_WORK > 1:
                                fast_page_valid = pl.load(cmp_work_valid, [fast_vec_request, 0], [1, CMP_ATTN_K_TILE], target_memory=pl.MemorySpace.Vec)
                                fast_page_bias = pl.mul(pl.sub(fast_page_valid, 1.0), -NEG_INF)
                                fast_scaled = pl.col_expand_add(fast_scaled, fast_page_bias)
                                fast_shaped = pl.set_validshape(fast_scaled, H // 2, pl.min(CMP_ATTN_K_TILE, fast_vec_rows))
                                fast_masked = pl.fillpad(fast_shaped, pad_value=pl.PadValue.min)
                                fast_mi = pl.row_max(fast_masked, fast_tmp)
                                fast_exp = pl.exp(pl.row_expand_sub(fast_masked, fast_mi))
                                fast_exp = pl.col_expand_mul(fast_exp, fast_page_valid)
                            else:
                                fast_shaped = pl.set_validshape(fast_scaled, H // 2, pl.min(CMP_ATTN_K_TILE, fast_vec_rows))
                                fast_masked = pl.fillpad(fast_shaped, pad_value=pl.PadValue.min)
                                fast_mi = pl.row_max(fast_masked, fast_tmp)
                                fast_exp = pl.exp(pl.row_expand_sub(fast_masked, fast_mi))
                            fast_li = pl.row_sum(fast_exp, fast_tmp)
                            fast_probability = pl.cast(fast_exp, target_type=pl.BF16, mode="rint")
                            pl.store(fast_probability, [fast_transfer_row, 0], probability_transfer)
                            pl.store(fast_mi, [fast_vec_t * H + fast_head, 0], cmp_partial_m)
                            pl.store(fast_li, [fast_vec_t * H + fast_head, 0], cmp_partial_l)
                            pl.system.sync_set(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                        else:
                            fast_neutral_m = pl.load(attn_sink_col, [fast_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec)
                            fast_neutral_l = pl.tile.muls(fast_neutral_m, 0.0)
                            fast_neutral_o = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                            pl.store(fast_neutral_m, [fast_vec_t * H + fast_head, 0], cmp_partial_m)
                            pl.store(fast_neutral_l, [fast_vec_t * H + fast_head, 0], cmp_partial_l)
                            pl.store(fast_neutral_o, [fast_vec_t * H + fast_head, 0], cmp_partial_o)
                            pl.store(fast_neutral_o, [fast_vec_t * H + fast_head, HEAD_DIM // 2], cmp_partial_o)
            else:
                for qk_t in pl.range(qk_core, t_dim, NUM_QK_CORES):
                    qk_request = qk_t // S
                    qk_position = pl.max(pl.read(position_ids, [qk_t]), -1)
                    qk_kv_len = pl.max(pl.read(kv_seq_lens, [qk_request]), 0)
                    qk_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min((qk_position + 1) // COMPRESS_RATIO, qk_kv_len // COMPRESS_RATIO))
                    qk_blocks = pl.min(cmp_work_count, (qk_rows + CMP_ATTN_K_TILE - 1) // CMP_ATTN_K_TILE)
                    qk_q, qk_l1 = _cmp_query_kv(q, qk_t)
                    for qk_tick in pl.range(qk_blocks + QK_PRE_LAUNCH):
                        if qk_tick < qk_blocks:
                            qk_sb = qk_tick
                            qk_page_ok = 1
                            if CMP_PAGES_PER_WORK == 1:
                                qk_first_page = pl.read(cmp_block_table, [qk_request, qk_sb * CMP_PAGES_PER_WORK])
                                qk_page_ok = pl.min(qk_first_page + 1, cmp_block_num - qk_first_page)
                            if qk_page_ok > 0:
                                qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                                qk_kv_row = (qk_request * cmp_work_count + qk_sb) * CMP_ATTN_K_TILE
                                qk_transfer_row = qk_slot * H
                                if TP == 1:
                                    qk_l1_row = (qk_sb % QK_TRANSFER_SLOTS) * CMP_ATTN_K_TILE
                                    qk_l1 = pl.gather_row(qk_l1, cmp_work_kv, [qk_l1_row, 0], [qk_kv_row, 0], [CMP_ATTN_K_TILE, HEAD_DIM])
                                    qk_l1_t = pl.tile.transpose_view(qk_l1)
                                    qk_kv_t = pl.tile.slice(qk_l1_t, [HEAD_DIM, CMP_ATTN_K_TILE], [0, qk_l1_row])
                                else:
                                    qk_kv = pl.load(cmp_work_kv, [qk_kv_row, 0], [CMP_ATTN_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                                    qk_kv_t = pl.tile.transpose_view(qk_kv)
                                qk_scores = pl.matmul(qk_q, qk_kv_t, out_dtype=pl.FP32)
                                pl.store(qk_scores, [qk_transfer_row, 0], score_transfer)
                                pl.system.sync_set(
                                    QK_SCORE_READY_EVENT, pipe=pl.PipeType.FIX,
                                    ffts_mode=2, core_type=pl.KernelType.AIC,
                                )
                        if qk_tick >= QK_PRE_LAUNCH:
                            pv_sb = qk_tick - QK_PRE_LAUNCH
                            pv_page_ok = 1
                            if CMP_PAGES_PER_WORK == 1:
                                pv_first_page = pl.read(cmp_block_table, [qk_request, pv_sb * CMP_PAGES_PER_WORK])
                                pv_page_ok = pl.min(pv_first_page + 1, cmp_block_num - pv_first_page)
                            if pv_page_ok > 0:
                                pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                                pv_transfer_row = pv_slot * H
                                pl.system.sync_wait(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                                pv_probability = pl.load(
                                    probability_transfer, [pv_transfer_row, 0], [H, CMP_ATTN_K_TILE],
                                    target_memory=pl.MemorySpace.Mat,
                                )
                                if TP == 1:
                                    pv_l1_row = (pv_sb % QK_TRANSFER_SLOTS) * CMP_ATTN_K_TILE
                                    pv_kv = pl.tile.slice(qk_l1, [CMP_ATTN_K_TILE, HEAD_DIM], [pv_l1_row, 0])
                                else:
                                    pv_kv_row = (qk_request * cmp_work_count + pv_sb) * CMP_ATTN_K_TILE
                                    pv_kv = pl.load(cmp_work_kv, [pv_kv_row, 0], [CMP_ATTN_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                                pv_l0_left = pl.tile.move(pv_probability, target_memory=pl.MemorySpace.Left)
                                pv_l0_right = pl.tile.extract(
                                    pv_kv, 0, 0, [CMP_ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right,
                                )
                                pv_previous = pl.tile.matmul(pv_l0_left, pv_l0_right)
                                for pv_n in pl.unroll(1, HEAD_DIM // PV_N_TILE):
                                    pv_next = pl.tile.extract(
                                        pv_kv, 0, pv_n * PV_N_TILE,
                                        [CMP_ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right,
                                    )
                                    pv_current = pl.tile.matmul(pv_l0_left, pv_next)
                                    pl.store(pv_previous, [pv_transfer_row, (pv_n - 1) * PV_N_TILE], pv_transfer)
                                    pv_previous = pv_current
                                pl.store(pv_previous, [pv_transfer_row, HEAD_DIM - PV_N_TILE], pv_transfer)
                                pl.system.sync_set(
                                    QK_PV_READY_EVENT, pipe=pl.PipeType.FIX,
                                    ffts_mode=2, core_type=pl.KernelType.AIC,
                                )

                    for qk_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                        pl.system.set_ffts(ffts_workspace)
                        qk_lane_head = qk_aiv * (H // 2)
                        qk_reduce_tmp = pl.create_tile([H // 2, CMP_ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                        running_m = pl.load(attn_sink_col, [qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec)
                        running_l = pl.tile.muls(running_m, 0.0)
                        running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                        running_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                        for qk_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                            qk_blocks + QK_PRE_LAUNCH,
                            init_values=(running_m, running_l, running_left, running_right),
                        ):
                            if qk_tick < qk_blocks:
                                qk_sb = qk_tick
                                qk_page_ok = 1
                                if CMP_PAGES_PER_WORK == 1:
                                    qk_first_page = pl.read(cmp_block_table, [qk_request, qk_sb * CMP_PAGES_PER_WORK])
                                    qk_page_ok = pl.min(qk_first_page + 1, cmp_block_num - qk_first_page)
                                if qk_page_ok > 0:
                                    qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                                    qk_transfer_row = qk_slot * H
                                    pl.system.sync_wait(QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                                    qk_scores_half = pl.load(
                                        score_transfer, [qk_transfer_row + qk_lane_head, 0], [H // 2, CMP_ATTN_K_TILE],
                                        target_memory=pl.MemorySpace.Vec,
                                    )
                                    qk_scaled = pl.mul(qk_scores_half, SOFTMAX_SCALE)
                                    if CMP_PAGES_PER_WORK > 1:
                                        qk_valid_item = qk_request * cmp_work_count + qk_sb
                                        qk_page_valid = pl.load(cmp_work_valid, [qk_valid_item, 0], [1, CMP_ATTN_K_TILE], target_memory=pl.MemorySpace.Vec)
                                        qk_page_bias = pl.mul(pl.sub(qk_page_valid, 1.0), -NEG_INF)
                                        qk_scaled = pl.col_expand_add(qk_scaled, qk_page_bias)
                                        qk_valid_rows = pl.min(CMP_ATTN_K_TILE, qk_rows - qk_sb * CMP_ATTN_K_TILE)
                                        qk_valid_scores = pl.set_validshape(qk_scaled, H // 2, qk_valid_rows)
                                        qk_masked = pl.fillpad(qk_valid_scores, pad_value=pl.PadValue.min)
                                        qk_mi = pl.row_max(qk_masked, qk_reduce_tmp)
                                        qk_exp = pl.exp(pl.row_expand_sub(qk_masked, qk_mi))
                                        qk_exp = pl.col_expand_mul(qk_exp, qk_page_valid)
                                    else:
                                        qk_valid_rows = pl.min(CMP_ATTN_K_TILE, qk_rows - qk_sb * CMP_ATTN_K_TILE)
                                        qk_valid_scores = pl.set_validshape(qk_scaled, H // 2, qk_valid_rows)
                                        qk_masked = pl.fillpad(qk_valid_scores, pad_value=pl.PadValue.min)
                                        qk_mi = pl.row_max(qk_masked, qk_reduce_tmp)
                                        qk_exp = pl.exp(pl.row_expand_sub(qk_masked, qk_mi))
                                    qk_li = pl.row_sum(qk_exp, qk_reduce_tmp)
                                    qk_probability = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                                    pl.store(qk_probability, [qk_transfer_row + qk_lane_head, 0], probability_transfer)
                                    pl.store(qk_mi, [qk_transfer_row + qk_lane_head, 0], mi_transfer)
                                    pl.store(qk_li, [qk_transfer_row + qk_lane_head, 0], li_transfer)
                                    pl.system.sync_set(
                                        QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE3,
                                        ffts_mode=2, core_type=pl.KernelType.AIV,
                                    )
                            if qk_tick >= QK_PRE_LAUNCH:
                                pv_sb = qk_tick - QK_PRE_LAUNCH
                                pv_page_ok = 1
                                if CMP_PAGES_PER_WORK == 1:
                                    pv_first_page = pl.read(cmp_block_table, [qk_request, pv_sb * CMP_PAGES_PER_WORK])
                                    pv_page_ok = pl.min(pv_first_page + 1, cmp_block_num - pv_first_page)
                                if pv_page_ok > 0:
                                    pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                                    pv_transfer_row = pv_slot * H
                                    pl.system.sync_wait(QK_PV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                                    pv_m = pl.load(mi_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec)
                                    pv_l = pl.load(li_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec)
                                    next_m = pl.maximum(m_iter, pv_m)
                                    alpha = pl.exp(pl.sub(m_iter, next_m))
                                    beta = pl.exp(pl.sub(pv_m, next_m))
                                    next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, pv_l))
                                    pv_left = pl.load(
                                        pv_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, HEAD_DIM // 2],
                                        target_memory=pl.MemorySpace.Vec,
                                    )
                                    next_left = pl.add(pl.row_expand_mul(left_iter, alpha), pl.row_expand_mul(pv_left, beta))
                                    pv_right = pl.load(
                                        pv_transfer, [pv_transfer_row + qk_lane_head, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2],
                                        target_memory=pl.MemorySpace.Vec,
                                    )
                                    next_right = pl.add(pl.row_expand_mul(right_iter, alpha), pl.row_expand_mul(pv_right, beta))
                                    m_valid, l_valid, left_valid, right_valid = pl.yield_(next_m, next_l, next_left, next_right)
                                else:
                                    m_valid, l_valid, left_valid, right_valid = pl.yield_(m_iter, l_iter, left_iter, right_iter)
                                m_after, l_after, left_after, right_after = pl.yield_(m_valid, l_valid, left_valid, right_valid)
                            else:
                                m_after, l_after, left_after, right_after = pl.yield_(m_iter, l_iter, left_iter, right_iter)
                            running_m, running_l, running_left, running_right = pl.yield_(m_after, l_after, left_after, right_after)
                        qk_output_row = qk_t * H + qk_lane_head
                        pl.store(running_m, [qk_output_row, 0], cmp_partial_m)
                        pl.store(running_l, [qk_output_row, 0], cmp_partial_l)
                        pl.store(running_left, [qk_output_row, 0], cmp_partial_o)
                        pl.store(running_right, [qk_output_row, HEAD_DIM // 2], cmp_partial_o)

        cmp_branch_tids[0] = cmp_qk_tid

    return (
        stream_state_m, stream_state_l, stream_heads,
        cmp_partial_m, cmp_partial_l, cmp_partial_o,
        rope_cos_il, rope_sin_signed,
        raw_branch_tids[0], cmp_branch_tids[0], rope_cs_tids[0],
    )


@pl.jit.inline(auto_scope=False)
def _legacy_sparse_attn_hca_tp1(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B_DYN, ORI_TABLE_COLUMNS_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    raw_cache_ready_dep: pl.Scalar[pl.TASK_ID],
    cmp_cache_ready_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16], pl.Scalar[pl.TASK_ID]]:
    """归并滑窗与压缩分支，写入公共 O projection 的分组布局。"""
    (
        stream_state_m, stream_state_l, stream_heads,
        cmp_partial_m, cmp_partial_l, cmp_partial_o,
        rope_cos_il, rope_sin_signed,
        raw_tid, cmp_tid, rope_tid,
    ) = sparse_attn_hca(
        q, ori_kv, ori_block_table, cmp_kv, cmp_block_table, position_ids, kv_seq_lens, attn_sink, freqs_cos,
        freqs_sin, raw_cache_ready_dep, cmp_cache_ready_dep,
    )
    t_dim = pl.tensor.dim(stream_state_m, 0) // H
    stream_block_count = t_dim * (H // H_TILE)
    attn_sink_col = pl.reshape(attn_sink, [H, 1])

    with pl.spmd(
        MERGE_WORKERS, name_hint="hca_stream_merge_pack", allow_early_resolve=True, deps=[raw_tid, cmp_tid, rope_tid],
    ) as heads_tid:
        worker = pl.tile.get_block_idx()
        stream_swap_one = pl.tile.full([1, ROPE_DIM], dtype=pl.FP32, value=1.0)
        stream_swap_lane_ids = pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        stream_swap_index = pl.cast(stream_swap_lane_ids, target_type=pl.FP32)
        stream_swap_col = pl.col_expand_mul(stream_swap_one, stream_swap_index)
        stream_swap_half = pl.mul(stream_swap_col, 0.5)
        stream_swap_dup = pl.cast(stream_swap_half, target_type=pl.INT32, mode="trunc")
        stream_swap_dup_f = pl.cast(stream_swap_dup, target_type=pl.FP32)
        stream_swap_lane = pl.sub(stream_swap_col, pl.mul(stream_swap_dup_f, 2.0))
        stream_swap_next = pl.add(stream_swap_col, 1.0)
        stream_swap_back = pl.mul(stream_swap_lane, 2.0)
        stream_swap = pl.sub(stream_swap_next, stream_swap_back)
        stream_swap_zero = pl.tile.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
        stream_swap_source = pl.add(stream_swap, NOPE_DIM)
        stream_swap_grid = pl.col_expand_add(stream_swap_zero, stream_swap_source)
        stream_row_ids = pl.tile.arange(0, [1, H_TILE], dtype=pl.INT32)
        stream_row_ids_f = pl.cast(stream_row_ids, target_type=pl.FP32)
        stream_row_offsets = pl.mul(stream_row_ids_f, HEAD_DIM)
        stream_row_offsets_col = pl.reshape(stream_row_offsets, [H_TILE, 1])
        stream_swap_flat = pl.row_expand_add(stream_swap_grid, stream_row_offsets_col)
        stream_swap_idx = pl.cast(stream_swap_flat, target_type=pl.INT32)
        stream_gather_tmp = pl.create_tile([H_TILE, ROPE_DIM], dtype=pl.INT32)
        for stream_idx in pl.range(worker, stream_block_count, MERGE_WORKERS):
            merge_t = stream_idx // (H // H_TILE)
            merge_h_tile = stream_idx - merge_t * (H // H_TILE)
            merge_h0 = merge_h_tile * H_TILE
            merge_state_row = merge_t * H + merge_h0
            stream_m = pl.load(stream_state_m, [merge_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
            stream_l = pl.load(stream_state_l, [merge_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
            stream_o = pl.load(stream_heads, [merge_state_row, 0], [H_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)

            stream_cmp_m = pl.load(cmp_partial_m, [merge_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
            stream_cmp_l = pl.load(cmp_partial_l, [merge_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
            stream_cmp_o = pl.load(
                cmp_partial_o, [merge_state_row, 0], [H_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec,
            )
            stream_sink = pl.load(attn_sink_col, [merge_h0, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
            stream_m_new = pl.maximum(stream_m, stream_cmp_m)
            stream_m_new = pl.maximum(stream_m_new, stream_sink)
            stream_alpha = pl.exp(pl.sub(stream_m, stream_m_new))
            stream_beta = pl.exp(pl.sub(stream_cmp_m, stream_m_new))
            stream_l = pl.add(pl.mul(stream_alpha, stream_l), pl.mul(stream_beta, stream_cmp_l))
            stream_o_scaled = pl.row_expand_mul(stream_o, stream_alpha)
            stream_cmp_o_scaled = pl.row_expand_mul(stream_cmp_o, stream_beta)
            stream_o = pl.add(stream_o_scaled, stream_cmp_o_scaled)
            stream_m = stream_m_new
            stream_sink_tile = pl.add(pl.sub(stream_m, stream_m), stream_sink)
            stream_denom = pl.add(stream_l, pl.exp(pl.sub(stream_sink_tile, stream_m)))
            stream_output = pl.row_expand_div(stream_o, stream_denom)
            stream_bf16 = pl.cast(stream_output, target_type=pl.BF16, mode="rint")
            stream_rope = stream_output[0:H_TILE, NOPE_DIM:HEAD_DIM]
            stream_cos_il = pl.load(rope_cos_il, [merge_t, 0], [1, ROPE_DIM])
            stream_sin_signed = pl.load(rope_sin_signed, [merge_t, 0], [1, ROPE_DIM])
            stream_swapped = pl.tile.gather(stream_output, stream_swap_idx, stream_gather_tmp)
            stream_rope_cos = pl.col_expand_mul(stream_rope, stream_cos_il)
            stream_swap_sin = pl.col_expand_mul(stream_swapped, stream_sin_signed)
            stream_rot = pl.add(stream_rope_cos, stream_swap_sin)
            stream_rope_bf16 = pl.cast(stream_rot, target_type=pl.BF16, mode="rint")
            stream_full_bf16 = pl.concat(stream_bf16[0:H_TILE, 0:NOPE_DIM], stream_rope_bf16)
            stream_groups = pl.reshape(stream_full_bf16, [PUBLISH_GROUPS, O_GROUP_IN])
            stream_pack_first = stream_groups[0:1, 0:O_GROUP_IN]
            stream_pack_second = stream_groups[1:2, 0:O_GROUP_IN]
            stream_group0 = merge_h0 // HEADS_PER_GROUP
            stream_pack_row0 = stream_group0 * T_PAD + merge_t
            stream_pack_row1 = stream_pack_row0 + T_PAD
            pl.store(stream_pack_first, [stream_pack_row0, 0], o_packed_heads)
            pl.store(stream_pack_second, [stream_pack_row1, 0], o_packed_heads)

    return o_packed_heads, heads_tid


@pl.jit.inline(auto_scope=False)
def _short_sparse_attn_hca_tp1(
    q: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B_DYN, ORI_TABLE_COLUMNS_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    raw_cache_ready_dep: pl.Scalar[pl.TASK_ID],
    cmp_cache_ready_dep: pl.Scalar[pl.TASK_ID],
    q_ready: pl.Array[4, pl.TASK_ID],
) -> tuple[pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16], pl.Scalar[pl.TASK_ID]]:
    t_dim = pl.tensor.dim(q, 0)
    request_count = pl.tensor.dim(kv_seq_lens, 0)
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_table_blocks = pl.tensor.dim(cmp_block_table, 1)
    cmp_work_count = 1
    cmp_gather_count = request_count
    raw_gather_count = request_count
    ori_cache_ready_dep = raw_cache_ready_dep
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * CMP_STORAGE_BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [t_dim * H, HEAD_DIM])
    rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    raw_kv = pl.create_tensor([request_count * REQUEST_KV_ROWS, HEAD_DIM], dtype=pl.BF16)
    raw_valid = pl.create_tensor([t_dim, WIN], dtype=pl.FP32)
    # 按 Native 原始 KV 页表聚合一个请求的 128+5 行；不创建外部窗口索引。
    with pl.spmd(raw_gather_count, name_hint="hca_gather_kv", allow_early_resolve=True, deps=[ori_cache_ready_dep]) as raw_gather_tid:
        g_req = pl.tile.get_block_idx()
        g_t0 = g_req * S
        g_base = g_req * REQUEST_KV_ROWS
        g_length = pl.read(kv_seq_lens, [g_req])
        g_position = pl.read(position_ids, [g_t0])
        g_first_len = pl.min(WIN, pl.max(pl.min(g_position + 1, g_length), 0))
        g_start = pl.max(g_position + 1 - g_first_len, 0)
        raw_kv[g_base:g_base + REQUEST_KV_ROWS, :] = pl.full([REQUEST_KV_ROWS, HEAD_DIM], dtype=pl.BF16, value=0.0)
        if g_length > 0:
            # 一个 Native 页内的行物理连续；首尾页只复制有效行，不读取未初始化 KV。
            g_rows = pl.min(g_first_len + S - 1, pl.max(g_length - g_start, 0))
            g_pages = (g_start % BLOCK_SIZE + g_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
            for g_part in pl.range(g_pages):
                g_absolute_page = g_start // BLOCK_SIZE + g_part
                g_offset = pl.max(g_start - g_absolute_page * BLOCK_SIZE, 0)
                g_row = pl.max(g_absolute_page * BLOCK_SIZE - g_start, 0)
                g_count = pl.min(BLOCK_SIZE - g_offset, g_rows - g_row)
                if g_count > 0:
                    g_page = pl.read(ori_block_table, [g_req, g_absolute_page])
                    if g_page >= 0 and g_page < ori_block_num:
                        g_source = pl.cast(g_page, pl.INDEX) * BLOCK_SIZE + g_offset
                        g_chunk = pl.load(
                            ori_kv_flat, [g_source, 0], [BLOCK_SIZE, HEAD_DIM],
                            valid_shape=[g_count, HEAD_DIM],
                        )
                        pl.store(g_chunk, [g_base + g_row, 0], raw_kv)

    # B*6 不一定是 8 的倍数，尾块逐实际 token 写入；seq_lens=0 排除 dummy。
    with pl.spmd((t_dim + VALID_TOKEN_TILE - 1) // VALID_TOKEN_TILE, name_hint="hca_raw_valid", allow_early_resolve=True,
                 deps=[ori_cache_ready_dep]) as raw_valid_tid:
        valid_t0 = pl.tile.get_block_idx() * VALID_TOKEN_TILE
        valid_col = pl.cast(pl.tile.arange(0, [1, WIN], dtype=pl.INT32), pl.FP32)
        for valid_dt in pl.range(pl.min(VALID_TOKEN_TILE, t_dim - valid_t0)):
            valid_token = valid_t0 + valid_dt
            valid_length = pl.min(WIN, pl.max(pl.min(
                pl.read(position_ids, [valid_token]) + 1,
                pl.read(kv_seq_lens, [valid_token // S]),
            ), 0))
            valid_mask = pl.minimum(pl.maximum(pl.neg(pl.sub(valid_col, pl.cast(valid_length, pl.FP32))), 0.0), 1.0)
            pl.store(valid_mask, [valid_token, 0], raw_valid)

    # Native 已提供交错的 FP32 频率；inverse RoPE 仅在消费者内折叠符号。
    with pl.spmd(rope_cs_blocks, name_hint="hca_inverse_rope_sign", allow_early_resolve=True, deps=[ori_cache_ready_dep]) as rope_cs_tid:
        cs_t0 = pl.tile.get_block_idx() * ROPE_CS_T_TILE
        cs_index = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), pl.FP32)
        cs_pair = pl.cast(pl.cast(pl.mul(cs_index, 0.5), pl.INT32, mode="trunc"), pl.FP32)
        cs_odd = pl.sub(cs_index, pl.mul(cs_pair, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_odd, 2.0), 1.0))
        rope_cos_il[cs_t0:cs_t0 + ROPE_CS_T_TILE, :] = freqs_cos[cs_t0:cs_t0 + ROPE_CS_T_TILE, :]
        rope_sin_signed[cs_t0:cs_t0 + ROPE_CS_T_TILE, :] = pl.col_expand_mul(
            freqs_sin[cs_t0:cs_t0 + ROPE_CS_T_TILE, :], cs_sign,
        )


    cmp_work_kv = pl.create_tensor([cmp_gather_count * CMP_ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16)

    cmp_work_valid = pl.create_tensor([cmp_gather_count, CMP_ATTN_K_TILE], dtype=pl.FP32)
    cmp_gather_blocks = cmp_gather_count
    if cmp_table_blocks >= 8:
        cmp_gather_blocks = (cmp_gather_count + CMP_GATHER_WORK_TILE - 1) // CMP_GATHER_WORK_TILE
    with pl.spmd(cmp_gather_blocks, name_hint="hca_cmp_work_gather", allow_early_resolve=True, deps=[cmp_cache_ready_dep]) as cmp_gather_tid:
        gather_block = pl.tile.get_block_idx()
        gather_begin = gather_block
        gather_end = gather_block + 1
        if cmp_table_blocks >= 8:
            gather_begin = gather_block * CMP_GATHER_WORK_TILE
            gather_end = pl.min(gather_begin + CMP_GATHER_WORK_TILE, cmp_gather_count)
        for gather_item in pl.range(gather_begin, gather_end):
            gather_request = gather_item // cmp_work_count
            gather_work = gather_item - gather_request * cmp_work_count
            gather_first_col = gather_work * CMP_PAGES_PER_WORK
            gather_dst0 = gather_item * CMP_ATTN_K_TILE
            gather_tile = pl.tile.full([CMP_ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
            gather_mask = pl.tile.full([1, CMP_ATTN_K_TILE], dtype=pl.FP32, value=0.0)
            # Native 不初始化有效长度之后的 KV；零概率乘 NaN 仍会污染 PV。
            # 只搬本步已生成的压缩行，未搬入的片内位置保留为零。
            gather_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.max(
                pl.read(kv_seq_lens, [gather_request]), 0,
            ) // COMPRESS_RATIO)
            for gather_page in pl.range(CMP_PAGES_PER_WORK):
                gather_page_col = gather_first_col + gather_page
                gather_valid_rows = pl.min(CMP_STORAGE_BLOCK_SIZE, pl.max(
                    gather_rows - gather_page_col * CMP_STORAGE_BLOCK_SIZE, 0,
                ))
                if gather_page_col < cmp_table_blocks and gather_valid_rows > 0:
                    gather_page_i32 = pl.read(cmp_block_table, [gather_request, gather_page_col])
                    if gather_page_i32 >= 0:
                        if gather_page_i32 < cmp_block_num:
                            gather_page_id = pl.cast(gather_page_i32, pl.INDEX)
                            gather_src = gather_page_id * CMP_STORAGE_BLOCK_SIZE
                            gather_col = gather_page * CMP_STORAGE_BLOCK_SIZE
                            gather_tile = pl.gather_row(
                                gather_tile, cmp_kv_flat, [gather_col, 0], [gather_src, 0],
                                [CMP_STORAGE_BLOCK_SIZE, HEAD_DIM], valid_shape=[gather_valid_rows, HEAD_DIM],
                            )
                            for gather_mask_col in pl.unroll(CMP_STORAGE_BLOCK_SIZE):
                                if gather_mask_col < gather_valid_rows:
                                    pl.tile.write(gather_mask, [0, gather_col + gather_mask_col], 1.0)
            pl.store(gather_tile, [gather_dst0, 0], cmp_work_kv)
            pl.store(gather_mask, [gather_item, 0], cmp_work_valid)



    # 原始/压缩两路保持独立的 softmax 和 K128 PV，融合任务只消除启动与中间状态。
    short_workers = 24
    transfer_rows = short_workers * H
    short_scores = pl.create_tensor([transfer_rows, 2 * ATTN_K_TILE], dtype=pl.FP32)
    short_probs = pl.create_tensor([transfer_rows, 2 * ATTN_K_TILE], dtype=pl.BF16)
    short_raw = pl.create_tensor([transfer_rows, HEAD_DIM], dtype=pl.FP32)
    short_cmp = pl.create_tensor([transfer_rows, HEAD_DIM], dtype=pl.FP32)
    short_ffts = pl.create_tensor([256], dtype=pl.INT64)
    sink_column = pl.reshape(attn_sink, [H, 1])
    with pl.spmd(
        short_workers, name_hint="hca_short_attention_pack",
        deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid, q_ready], allow_early_resolve=True,
    ) as short_tid:
        short_worker = pl.tile.get_block_idx()
        pl.system.set_ffts(short_ffts)
        short_count = pl.max((t_dim - short_worker + short_workers - 1) // short_workers, 0)
        for item in pl.range(short_count):
            token = short_worker + item * short_workers
            request = token // S
            first_length = pl.min(WIN, pl.max(pl.min(
                pl.read(position_ids, [request * S]) + 1, pl.read(kv_seq_lens, [request]),
            ), 0))
            raw_drop = pl.max(first_length + token % S - WIN, 0)
            query = pl.load(q_flat, [token * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            raw_key = pl.load(raw_kv, [request * REQUEST_KV_ROWS + raw_drop, 0],
                              [ATTN_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            cmp_key = pl.load(cmp_work_kv, [request * ATTN_K_TILE, 0],
                              [ATTN_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            raw_qk = pl.matmul(query, pl.tile.transpose_view(raw_key), out_dtype=pl.FP32)
            cmp_qk = pl.matmul(query, pl.tile.transpose_view(cmp_key), out_dtype=pl.FP32)
            pl.store(raw_qk, [short_worker * H, 0], short_scores)
            pl.store(cmp_qk, [short_worker * H, ATTN_K_TILE], short_scores)
            pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
            raw_probability = pl.load(short_probs, [short_worker * H, 0], [H, ATTN_K_TILE],
                                      target_memory=pl.MemorySpace.Mat)
            cmp_probability = pl.load(short_probs, [short_worker * H, ATTN_K_TILE], [H, ATTN_K_TILE],
                                      target_memory=pl.MemorySpace.Mat)
            raw_left = pl.tile.move(raw_probability, target_memory=pl.MemorySpace.Left)
            cmp_left = pl.tile.move(cmp_probability, target_memory=pl.MemorySpace.Left)
            for col in pl.unroll(HEAD_DIM // PV_N_TILE):
                raw_right = pl.tile.extract(raw_key, 0, col * PV_N_TILE, [ATTN_K_TILE, PV_N_TILE],
                                            target_memory=pl.MemorySpace.Right)
                cmp_right = pl.tile.extract(cmp_key, 0, col * PV_N_TILE, [ATTN_K_TILE, PV_N_TILE],
                                            target_memory=pl.MemorySpace.Right)
                raw_pv = pl.tile.matmul(raw_left, raw_right)
                cmp_pv = pl.tile.matmul(cmp_left, cmp_right)
                pl.store(raw_pv, [short_worker * H, col * PV_N_TILE], short_raw)
                pl.store(cmp_pv, [short_worker * H, col * PV_N_TILE], short_cmp)
            pl.system.sync_set(2, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)

        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            pl.system.set_ffts(short_ffts)
            head0 = lane * (H // 2)
            reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32)
            sink_half = pl.load(sink_column, [head0, 0], [H // 2, 1])
            swap_one = pl.tile.full([1, ROPE_DIM], dtype=pl.FP32, value=1.0)
            swap_ids = pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), pl.FP32)
            swap_col = pl.col_expand_mul(swap_one, swap_ids)
            swap_pair = pl.cast(pl.cast(pl.mul(swap_col, 0.5), pl.INT32, mode="trunc"), pl.FP32)
            swap_lane = pl.sub(swap_col, pl.mul(swap_pair, 2.0))
            swap = pl.sub(pl.add(swap_col, 1.0), pl.mul(swap_lane, 2.0))
            swap_grid = pl.col_expand_add(pl.tile.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0),
                                          pl.add(swap, NOPE_DIM))
            row_ids = pl.cast(pl.tile.arange(0, [1, H_TILE], dtype=pl.INT32), pl.FP32)
            swap_index = pl.cast(pl.row_expand_add(swap_grid, pl.reshape(pl.mul(row_ids, HEAD_DIM),
                                                                                       [H_TILE, 1])), pl.INT32)
            gather_tmp = pl.create_tile([H_TILE, ROPE_DIM], dtype=pl.INT32)
            for vec_item in pl.range(short_count):
                vec_token = short_worker + vec_item * short_workers
                vec_request = vec_token // S
                vec_row = short_worker * H + head0
                vec_rows = pl.max(pl.min(
                    (pl.read(position_ids, [vec_token]) + 1) // COMPRESS_RATIO,
                    pl.read(kv_seq_lens, [vec_request]) // COMPRESS_RATIO,
                ), 0)
                pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                raw_score = pl.load(short_scores, [vec_row, 0], [H // 2, ATTN_K_TILE])
                raw_mask = pl.load(raw_valid, [vec_token, 0], [1, ATTN_K_TILE])
                raw_bias = pl.mul(pl.sub(raw_mask, 1.0), -NEG_INF)
                raw_scaled = pl.col_expand_add(pl.mul(raw_score, SOFTMAX_SCALE), raw_bias)
                raw_m = pl.row_max(raw_scaled, reduce_tmp)
                raw_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(raw_scaled, raw_m)), raw_mask)
                raw_l = pl.row_sum(raw_exp, reduce_tmp)
                pl.store(pl.cast(raw_exp, pl.BF16, mode="rint"), [vec_row, 0], short_probs)
                if vec_rows > 0:
                    cmp_score = pl.load(short_scores, [vec_row, ATTN_K_TILE], [H // 2, ATTN_K_TILE])
                    cmp_mask = pl.load(cmp_work_valid, [vec_request, 0], [1, ATTN_K_TILE])
                    cmp_bias = pl.mul(pl.sub(cmp_mask, 1.0), -NEG_INF)
                    cmp_scaled = pl.col_expand_add(pl.mul(cmp_score, SOFTMAX_SCALE), cmp_bias)
                    cmp_shaped = pl.set_validshape(cmp_scaled, H // 2, pl.min(ATTN_K_TILE, vec_rows))
                    cmp_masked = pl.fillpad(cmp_shaped, pad_value=pl.PadValue.min)
                    cmp_maximum = pl.row_max(cmp_masked, reduce_tmp)
                    cmp_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(cmp_masked, cmp_maximum)), cmp_mask)
                    cmp_total = pl.row_sum(cmp_exp, reduce_tmp)
                    cmp_prob = pl.cast(cmp_exp, pl.BF16, mode="rint")
                    pl.store(cmp_prob, [vec_row, ATTN_K_TILE], short_probs)
                    cmp_m, cmp_l = pl.yield_(cmp_maximum, cmp_total)
                else:
                    zero_probability = pl.tile.full([H // 2, ATTN_K_TILE], dtype=pl.BF16, value=0.0)
                    pl.store(zero_probability, [vec_row, ATTN_K_TILE], short_probs)
                    cmp_m, cmp_l = pl.yield_(sink_half, pl.mul(sink_half, 0.0))
                pl.system.sync_set(1, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                maximum = pl.maximum(pl.maximum(raw_m, cmp_m), sink_half)
                alpha = pl.exp(pl.sub(raw_m, maximum))
                beta = pl.exp(pl.sub(cmp_m, maximum))
                total = pl.add(pl.mul(alpha, raw_l), pl.mul(beta, cmp_l))
                sink_tile = pl.add(pl.sub(maximum, maximum), sink_half)
                denominator = pl.add(total, pl.exp(pl.sub(sink_tile, maximum)))
                cosine = pl.load(rope_cos_il, [vec_token, 0], [1, ROPE_DIM])
                sine = pl.load(rope_sin_signed, [vec_token, 0], [1, ROPE_DIM])
                pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                for head_part in pl.unroll(H // 2 // H_TILE):
                    offset = head_part * H_TILE
                    head = head0 + offset
                    raw_value = pl.load(short_raw, [short_worker * H + head, 0], [H_TILE, HEAD_DIM])
                    cmp_value = pl.load(short_cmp, [short_worker * H + head, 0], [H_TILE, HEAD_DIM])
                    merged = pl.add(pl.row_expand_mul(raw_value, pl.tile.slice(alpha, [H_TILE, 1], [offset, 0])),
                                    pl.row_expand_mul(cmp_value, pl.tile.slice(beta, [H_TILE, 1], [offset, 0])))
                    normalized = pl.row_expand_div(merged, pl.tile.slice(denominator, [H_TILE, 1], [offset, 0]))
                    bf16 = pl.cast(normalized, pl.BF16, mode="rint")
                    rope = normalized[0:H_TILE, NOPE_DIM:HEAD_DIM]
                    swapped = pl.tile.gather(normalized, swap_index, gather_tmp)
                    rotated = pl.add(pl.col_expand_mul(rope, cosine), pl.col_expand_mul(swapped, sine))
                    rope_bf16 = pl.cast(rotated, pl.BF16, mode="rint")
                    full = pl.concat(bf16[0:H_TILE, 0:NOPE_DIM], rope_bf16)
                    groups = pl.reshape(full, [PUBLISH_GROUPS, O_GROUP_IN])
                    packed_row = (head // HEADS_PER_GROUP) * T_PAD + vec_token
                    pl.store(groups[0:1, :], [packed_row, 0], o_packed_heads)
                    pl.store(groups[1:2, :], [packed_row + T_PAD, 0], o_packed_heads)
    return o_packed_heads, short_tid


@pl.jit.inline(auto_scope=False)
def _long_sparse_attn_hca_tp1(
    q: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B_DYN, ORI_TABLE_COLUMNS_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    raw_cache_ready_dep: pl.Scalar[pl.TASK_ID],
    cmp_cache_ready_dep: pl.Scalar[pl.TASK_ID],
    q_ready: pl.Array[4, pl.TASK_ID],
) -> tuple[pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16], pl.Scalar[pl.TASK_ID]]:
    t_dim = pl.tensor.dim(q, 0)
    request_count = pl.tensor.dim(kv_seq_lens, 0)
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_table_blocks = pl.tensor.dim(cmp_block_table, 1)
    cmp_work_count = (cmp_table_blocks + CMP_PAGES_PER_WORK - 1) // CMP_PAGES_PER_WORK
    cmp_gather_count = request_count * cmp_work_count
    raw_gather_count = request_count
    ori_cache_ready_dep = raw_cache_ready_dep
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * CMP_STORAGE_BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [t_dim * H, HEAD_DIM])
    rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    raw_kv = pl.create_tensor([request_count * REQUEST_KV_ROWS, HEAD_DIM], dtype=pl.BF16)
    raw_valid = pl.create_tensor([t_dim, WIN], dtype=pl.FP32)
    # 按 Native 原始 KV 页表聚合一个请求的 128+5 行；不创建外部窗口索引。
    with pl.spmd(raw_gather_count, name_hint="hca_gather_kv", allow_early_resolve=True, deps=[ori_cache_ready_dep]) as raw_gather_tid:
        g_req = pl.tile.get_block_idx()
        g_t0 = g_req * S
        g_base = g_req * REQUEST_KV_ROWS
        g_length = pl.read(kv_seq_lens, [g_req])
        g_position = pl.read(position_ids, [g_t0])
        g_first_len = pl.min(WIN, pl.max(pl.min(g_position + 1, g_length), 0))
        g_start = pl.max(g_position + 1 - g_first_len, 0)
        raw_kv[g_base:g_base + REQUEST_KV_ROWS, :] = pl.full([REQUEST_KV_ROWS, HEAD_DIM], dtype=pl.BF16, value=0.0)
        if g_length > 0:
            # 一个 Native 页内的行物理连续；首尾页只复制有效行，不读取未初始化 KV。
            g_rows = pl.min(g_first_len + S - 1, pl.max(g_length - g_start, 0))
            g_pages = (g_start % BLOCK_SIZE + g_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
            for g_part in pl.range(g_pages):
                g_absolute_page = g_start // BLOCK_SIZE + g_part
                g_offset = pl.max(g_start - g_absolute_page * BLOCK_SIZE, 0)
                g_row = pl.max(g_absolute_page * BLOCK_SIZE - g_start, 0)
                g_count = pl.min(BLOCK_SIZE - g_offset, g_rows - g_row)
                if g_count > 0:
                    g_page = pl.read(ori_block_table, [g_req, g_absolute_page])
                    if g_page >= 0 and g_page < ori_block_num:
                        g_source = pl.cast(g_page, pl.INDEX) * BLOCK_SIZE + g_offset
                        g_chunk = pl.load(
                            ori_kv_flat, [g_source, 0], [BLOCK_SIZE, HEAD_DIM],
                            valid_shape=[g_count, HEAD_DIM],
                        )
                        pl.store(g_chunk, [g_base + g_row, 0], raw_kv)

    # B*6 不一定是 8 的倍数，尾块逐实际 token 写入；seq_lens=0 排除 dummy。
    with pl.spmd((t_dim + VALID_TOKEN_TILE - 1) // VALID_TOKEN_TILE, name_hint="hca_raw_valid", allow_early_resolve=True,
                 deps=[ori_cache_ready_dep]) as raw_valid_tid:
        valid_t0 = pl.tile.get_block_idx() * VALID_TOKEN_TILE
        valid_col = pl.cast(pl.tile.arange(0, [1, WIN], dtype=pl.INT32), pl.FP32)
        for valid_dt in pl.range(pl.min(VALID_TOKEN_TILE, t_dim - valid_t0)):
            valid_token = valid_t0 + valid_dt
            valid_length = pl.min(WIN, pl.max(pl.min(
                pl.read(position_ids, [valid_token]) + 1,
                pl.read(kv_seq_lens, [valid_token // S]),
            ), 0))
            valid_mask = pl.minimum(pl.maximum(pl.neg(pl.sub(valid_col, pl.cast(valid_length, pl.FP32))), 0.0), 1.0)
            pl.store(valid_mask, [valid_token, 0], raw_valid)

    # Native 已提供交错的 FP32 频率；inverse RoPE 仅在消费者内折叠符号。
    with pl.spmd(rope_cs_blocks, name_hint="hca_inverse_rope_sign", allow_early_resolve=True, deps=[ori_cache_ready_dep]) as rope_cs_tid:
        cs_t0 = pl.tile.get_block_idx() * ROPE_CS_T_TILE
        cs_index = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), pl.FP32)
        cs_pair = pl.cast(pl.cast(pl.mul(cs_index, 0.5), pl.INT32, mode="trunc"), pl.FP32)
        cs_odd = pl.sub(cs_index, pl.mul(cs_pair, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_odd, 2.0), 1.0))
        rope_cos_il[cs_t0:cs_t0 + ROPE_CS_T_TILE, :] = freqs_cos[cs_t0:cs_t0 + ROPE_CS_T_TILE, :]
        rope_sin_signed[cs_t0:cs_t0 + ROPE_CS_T_TILE, :] = pl.col_expand_mul(
            freqs_sin[cs_t0:cs_t0 + ROPE_CS_T_TILE, :], cs_sign,
        )


    cmp_work_kv = pl.create_tensor([cmp_gather_count * CMP_ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16)

    cmp_work_valid = pl.create_tensor([cmp_gather_count, CMP_ATTN_K_TILE], dtype=pl.FP32)
    # 每块承担的搬运项数按实际项数反推，使块数不超过一个 AIV 波（48 块）。
    # 原先固定 CMP_GATHER_WORK_TILE 在 128K/B16 下算出约 64 块，是 1.5 波、白扔半波；
    # 实测该任务跨度 51.1 μs，而核时间只有 900 μs（48 核下限 18.8 μs）。
    # 划分只决定哪个块搬哪几项，搬运内容与顺序不变。
    gather_work_tile = (cmp_gather_count + CMP_GATHER_AIV_WAVE - 1) // CMP_GATHER_AIV_WAVE
    cmp_gather_blocks = (cmp_gather_count + gather_work_tile - 1) // gather_work_tile
    with pl.spmd(cmp_gather_blocks, name_hint="hca_cmp_work_gather", allow_early_resolve=True, deps=[cmp_cache_ready_dep]) as cmp_gather_tid:
        gather_block = pl.tile.get_block_idx()
        gather_begin = gather_block * gather_work_tile
        gather_end = pl.min(gather_begin + gather_work_tile, cmp_gather_count)
        for gather_item in pl.range(gather_begin, gather_end):
            gather_request = gather_item // cmp_work_count
            gather_work = gather_item - gather_request * cmp_work_count
            gather_first_col = gather_work * CMP_PAGES_PER_WORK
            gather_dst0 = gather_item * CMP_ATTN_K_TILE
            gather_tile = pl.tile.full([CMP_ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
            gather_mask = pl.tile.full([1, CMP_ATTN_K_TILE], dtype=pl.FP32, value=0.0)
            # Native 不初始化有效长度之后的 KV；零概率乘 NaN 仍会污染 PV。
            # 只搬本步已生成的压缩行，未搬入的片内位置保留为零。
            gather_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.max(
                pl.read(kv_seq_lens, [gather_request]), 0,
            ) // COMPRESS_RATIO)
            for gather_page in pl.range(CMP_PAGES_PER_WORK):
                gather_page_col = gather_first_col + gather_page
                gather_valid_rows = pl.min(CMP_STORAGE_BLOCK_SIZE, pl.max(
                    gather_rows - gather_page_col * CMP_STORAGE_BLOCK_SIZE, 0,
                ))
                if gather_page_col < cmp_table_blocks and gather_valid_rows > 0:
                    gather_page_i32 = pl.read(cmp_block_table, [gather_request, gather_page_col])
                    if gather_page_i32 >= 0:
                        if gather_page_i32 < cmp_block_num:
                            gather_page_id = pl.cast(gather_page_i32, pl.INDEX)
                            gather_src = gather_page_id * CMP_STORAGE_BLOCK_SIZE
                            gather_col = gather_page * CMP_STORAGE_BLOCK_SIZE
                            gather_tile = pl.gather_row(
                                gather_tile, cmp_kv_flat, [gather_col, 0], [gather_src, 0],
                                [CMP_STORAGE_BLOCK_SIZE, HEAD_DIM], valid_shape=[gather_valid_rows, HEAD_DIM],
                            )
                            for gather_mask_col in pl.unroll(CMP_STORAGE_BLOCK_SIZE):
                                if gather_mask_col < gather_valid_rows:
                                    pl.tile.write(gather_mask, [0, gather_col + gather_mask_col], 1.0)
            pl.store(gather_tile, [gather_dst0, 0], cmp_work_kv)
            pl.store(gather_mask, [gather_item, 0], cmp_work_valid)



    # 全部历史共用一组 MIX 任务：先处理压缩历史，最后一块处理 raw。
    # 保留原压缩分支归约及 raw + compressed 的合并顺序，不增加独立 merge 任务。
    transfer_rows = NUM_QK_CORES * QK_TRANSFER_SLOTS * H
    scores = pl.create_tensor([transfer_rows, ATTN_K_TILE], dtype=pl.FP32)
    probs = pl.create_tensor([transfer_rows, ATTN_K_TILE], dtype=pl.BF16)
    values = pl.create_tensor([transfer_rows, HEAD_DIM], dtype=pl.FP32)
    ffts = pl.create_tensor([256], dtype=pl.INT64)
    sink_col = pl.reshape(attn_sink, [H, 1])
    with pl.spmd(
        NUM_QK_CORES, name_hint="hca_unified_attention",
        deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid, q_ready], allow_early_resolve=True,
    ) as attention_tid:
        worker = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts)
        # QK进入下一query时保留上一query的KV槽；只在本worker最后一条query排空。
        kv_l1 = pl.create_tile([QK_TRANSFER_SLOTS * ATTN_K_TILE, HEAD_DIM],
                              dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
        for token, (work_offset,) in pl.range(
            worker, t_dim, NUM_QK_CORES, init_values=(pl.cast(0, pl.INDEX),),
        ):
            request = token // S
            length = pl.max(pl.read(kv_seq_lens, [request]), 0)
            position = pl.max(pl.read(position_ids, [token]), -1)
            cmp_rows = pl.cast(pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
                (position + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO,
            )), pl.INDEX)
            cmp_blocks = pl.min(cmp_work_count, (cmp_rows + ATTN_K_TILE - 1) // ATTN_K_TILE)
            work_count = 1 + cmp_blocks
            first_length = pl.min(WIN, pl.max(pl.min(pl.read(position_ids, [request * S]) + 1, length), 0))
            raw_drop = pl.max(first_length + token % S - WIN, 0)
            query = pl.load(q_flat, [token * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            if token + NUM_QK_CORES >= t_dim:
                extra = pl.yield_(QK_PRE_LAUNCH)
            else:
                extra = pl.yield_(0)
            for tick in pl.range(work_count + extra):
                global_tick = work_offset + tick
                if tick < work_count:
                    l1_row = (global_tick % QK_TRANSFER_SLOTS) * ATTN_K_TILE
                    if tick == cmp_blocks:
                        kv_l1 = pl.gather_row(kv_l1, raw_kv, [l1_row, 0],
                                             [request * REQUEST_KV_ROWS + raw_drop, 0], [ATTN_K_TILE, HEAD_DIM])
                    else:
                        cmp_row = (request * cmp_work_count + tick) * ATTN_K_TILE
                        kv_l1 = pl.gather_row(kv_l1, cmp_work_kv, [l1_row, 0], [cmp_row, 0],
                                             [ATTN_K_TILE, HEAD_DIM])
                    key_t = pl.tile.slice(pl.tile.transpose_view(kv_l1), [HEAD_DIM, ATTN_K_TILE], [0, l1_row])
                    qk = pl.matmul(query, key_t, out_dtype=pl.FP32)
                    row = (worker * QK_TRANSFER_SLOTS + global_tick % QK_TRANSFER_SLOTS) * H
                    pl.store(qk, [row, 0], scores)
                    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                if global_tick >= QK_PRE_LAUNCH:
                    pv_work = global_tick - QK_PRE_LAUNCH
                    pv_row = (worker * QK_TRANSFER_SLOTS + pv_work % QK_TRANSFER_SLOTS) * H
                    pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    probability = pl.load(probs, [pv_row, 0], [H, ATTN_K_TILE], target_memory=pl.MemorySpace.Mat)
                    left = pl.tile.move(probability, target_memory=pl.MemorySpace.Left)
                    pv_l1_row = (pv_work % QK_TRANSFER_SLOTS) * ATTN_K_TILE
                    right = pl.tile.extract(kv_l1, pv_l1_row, 0, [ATTN_K_TILE, PV_N_TILE],
                                            target_memory=pl.MemorySpace.Right)
                    previous = pl.tile.matmul(left, right)
                    for col in pl.unroll(1, HEAD_DIM // PV_N_TILE):
                        right_next = pl.tile.extract(kv_l1, pv_l1_row, col * PV_N_TILE,
                                                     [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                        current = pl.tile.matmul(left, right_next)
                        pl.store(previous, [pv_row, (col - 1) * PV_N_TILE], values)
                        previous = current
                    pl.store(previous, [pv_row, HEAD_DIM - PV_N_TILE], values)
                    pl.system.sync_set(2, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)

            work_offset_done = pl.yield_(work_offset + work_count)  # noqa: F841 -- PyPTO循环状态

        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            pl.system.set_ffts(ffts)
            head0 = lane * (H // 2)
            reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32)
            stream_seed_m = pl.load(sink_col, [head0, 0], [H // 2, 1])
            stream_seed_l = pl.mul(stream_seed_m, 0.0)
            stream_seed_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
            stream_seed_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
            # Native keeps softmaxMax/Sum in UB across the delayed PV stages.
            # These values never cross cores; only scores/probs/values need GM.
            max_ring_init = pl.tile.full([QK_TRANSFER_SLOTS, H // 2], dtype=pl.FP32, value=0.0)
            sum_ring_init = pl.tile.full([QK_TRANSFER_SLOTS, H // 2], dtype=pl.FP32, value=0.0)
            query_ring = pl.tile.full([QK_TRANSFER_SLOTS, 8], dtype=pl.INT32, value=0)
            for vec_token, (vec_offset, m_query, l_query, left_query, right_query, max_query, sum_query) in pl.range(
                worker, t_dim, NUM_QK_CORES,
                init_values=(pl.cast(0, pl.INDEX), stream_seed_m, stream_seed_l,
                             stream_seed_left, stream_seed_right, max_ring_init, sum_ring_init),
            ):
                request = vec_token // S
                length = pl.max(pl.read(kv_seq_lens, [request]), 0)
                position = pl.max(pl.read(position_ids, [vec_token]), -1)
                cmp_rows = pl.cast(pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
                    (position + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO,
                )), pl.INDEX)
                cmp_blocks = pl.min(cmp_work_count, (cmp_rows + ATTN_K_TILE - 1) // ATTN_K_TILE)
                work_count = 1 + cmp_blocks
                first_length = pl.min(WIN, pl.max(pl.min(pl.read(position_ids, [request * S]) + 1, length), 0))
                raw_drop = pl.max(first_length + vec_token % S - WIN, 0)
                if vec_token + NUM_QK_CORES >= t_dim:
                    vec_extra = pl.yield_(QK_PRE_LAUNCH)
                else:
                    vec_extra = pl.yield_(0)
                for vec_tick, (m_iter, l_iter, left_iter, right_iter, max_ring, sum_ring) in pl.range(
                    work_count + vec_extra,
                    init_values=(m_query, l_query, left_query, right_query, max_query, sum_query),
                ):
                    vec_global = vec_offset + vec_tick
                    if vec_tick < work_count:
                        vec_row = (worker * QK_TRANSFER_SLOTS + vec_global % QK_TRANSFER_SLOTS) * H + head0
                        pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        score = pl.load(scores, [vec_row, 0], [H // 2, ATTN_K_TILE])
                        if vec_tick == cmp_blocks:
                            raw_mask = pl.load(raw_valid, [vec_token, 0], [1, ATTN_K_TILE])
                            raw_bias = pl.mul(pl.sub(raw_mask, 1.0), -NEG_INF)
                            raw_scaled = pl.col_expand_add(pl.mul(score, SOFTMAX_SCALE), raw_bias)
                            raw_m = pl.row_max(raw_scaled, reduce_tmp)
                            raw_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(raw_scaled, raw_m)), raw_mask)
                            maximum, exponent = pl.yield_(raw_m, raw_exp)
                        else:
                            cmp_work = vec_tick
                            cmp_mask = pl.load(cmp_work_valid, [request * cmp_work_count + cmp_work, 0],
                                               [1, ATTN_K_TILE])
                            cmp_bias = pl.mul(pl.sub(cmp_mask, 1.0), -NEG_INF)
                            cmp_scaled = pl.col_expand_add(pl.mul(score, SOFTMAX_SCALE), cmp_bias)
                            valid_rows = pl.min(ATTN_K_TILE, cmp_rows - cmp_work * ATTN_K_TILE)
                            cmp_shaped = pl.set_validshape(cmp_scaled, H // 2, valid_rows)
                            cmp_masked = pl.fillpad(cmp_shaped, pad_value=pl.PadValue.min)
                            cmp_m = pl.row_max(cmp_masked, reduce_tmp)
                            cmp_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(cmp_masked, cmp_m)), cmp_mask)
                            maximum, exponent = pl.yield_(cmp_m, cmp_exp)
                        total = pl.row_sum(exponent, reduce_tmp)
                        vec_probability = pl.cast(exponent, pl.BF16, mode="rint")
                        pl.store(vec_probability, [vec_row, 0], probs)
                        next_max_ring = pl.tile.assemble(
                            max_ring, pl.reshape(maximum, [1, H // 2]), [vec_global % QK_TRANSFER_SLOTS, 0],
                        )
                        next_sum_ring = pl.tile.assemble(
                            sum_ring, pl.reshape(total, [1, H // 2]), [vec_global % QK_TRANSFER_SLOTS, 0],
                        )
                        # 描述符仅供本AIV消费；PV延迟两步时仍指向正确query，不增加GM任务。
                        pl.tile.write(query_ring, [vec_global % QK_TRANSFER_SLOTS, 0], pl.cast(vec_token, pl.INT32))
                        pl.tile.write(query_ring, [vec_global % QK_TRANSFER_SLOTS, 1], pl.cast(vec_tick, pl.INT32))
                        pl.tile.write(query_ring, [vec_global % QK_TRANSFER_SLOTS, 2],
                                      pl.cast(work_count - vec_tick - 1, pl.INT32))
                        pl.system.sync_set(1, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                        updated_max_ring, updated_sum_ring = pl.yield_(next_max_ring, next_sum_ring)
                    else:
                        updated_max_ring, updated_sum_ring = pl.yield_(max_ring, sum_ring)
                    if vec_global >= QK_PRE_LAUNCH:
                        out_work = vec_global - QK_PRE_LAUNCH
                        out_row = (worker * QK_TRANSFER_SLOTS + out_work % QK_TRANSFER_SLOTS) * H + head0
                        pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        out_m = pl.reshape(pl.tile.extract(
                            updated_max_ring, out_work % QK_TRANSFER_SLOTS, 0, [1, H // 2],
                            target_memory=pl.MemorySpace.Vec,
                        ), [H // 2, 1])
                        out_l = pl.reshape(pl.tile.extract(
                            updated_sum_ring, out_work % QK_TRANSFER_SLOTS, 0, [1, H // 2],
                            target_memory=pl.MemorySpace.Vec,
                        ), [H // 2, 1])
                        out_local = pl.tile.read(query_ring, [out_work % QK_TRANSFER_SLOTS, 1])
                        if out_local == 0:
                            fresh_m = pl.load(sink_col, [head0, 0], [H // 2, 1])
                            fresh_l = pl.mul(fresh_m, 0.0)
                            fresh_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                            fresh_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                            current_m, current_l, current_left, current_right = pl.yield_(
                                fresh_m, fresh_l, fresh_left, fresh_right,
                            )
                        else:
                            current_m, current_l, current_left, current_right = pl.yield_(
                                m_iter, l_iter, left_iter, right_iter,
                            )
                        # 长历史的 running_m 从 sink 开始，只增不减；最后 raw 合并不需要另设分支。
                        # 归约状态跟随PV消费的query；QK提前进入下一query不改变这里的归约顺序。
                        next_m = pl.maximum(current_m, out_m)
                        alpha = pl.exp(pl.sub(current_m, next_m))
                        beta = pl.exp(pl.sub(out_m, next_m))
                        next_l = pl.add(pl.mul(alpha, current_l), pl.mul(beta, out_l))
                        # 两个乘积先各自舍入为 FP32；有限值加法两侧交换不改变结果。
                        pv_left = pl.load(values, [out_row, 0], [H // 2, HEAD_DIM // 2])
                        next_left = pl.add(pl.row_expand_mul(current_left, alpha),
                                           pl.row_expand_mul(pv_left, beta))
                        pv_right = pl.load(values, [out_row, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2])
                        next_right = pl.add(pl.row_expand_mul(current_right, alpha),
                                            pl.row_expand_mul(pv_right, beta))
                        remaining = pl.tile.read(query_ring, [out_work % QK_TRANSFER_SLOTS, 2])
                        if remaining == 0:
                            publish_token = pl.cast(pl.tile.read(
                                query_ring, [out_work % QK_TRANSFER_SLOTS, 0],
                            ), pl.INDEX)
                            final_sink = pl.load(sink_col, [head0, 0], [H // 2, 1])
                            sink_value = pl.add(pl.sub(next_m, next_m), final_sink)
                            denominator = pl.add(next_l, pl.exp(pl.sub(sink_value, next_m)))
                            cosine = pl.load(rope_cos_il, [publish_token, 0], [1, ROPE_DIM])
                            sine = pl.load(rope_sin_signed, [publish_token, 0], [1, ROPE_DIM])
                            swap_one = pl.tile.full([1, ROPE_DIM], dtype=pl.FP32, value=1.0)
                            swap_ids = pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), pl.FP32)
                            swap_col = pl.col_expand_mul(swap_one, swap_ids)
                            swap_pair = pl.cast(pl.cast(pl.mul(swap_col, 0.5), pl.INT32, mode="trunc"), pl.FP32)
                            swap_lane = pl.sub(swap_col, pl.mul(swap_pair, 2.0))
                            swap = pl.sub(pl.add(swap_col, 1.0), pl.mul(swap_lane, 2.0))
                            swap_grid = pl.col_expand_add(pl.tile.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0),
                                                          pl.add(swap, NOPE_DIM))
                            row_ids = pl.cast(pl.tile.arange(0, [1, H_TILE], dtype=pl.INT32), pl.FP32)
                            swap_index = pl.cast(pl.row_expand_add(swap_grid, pl.reshape(pl.mul(row_ids, HEAD_DIM),
                                                                                        [H_TILE, 1])), pl.INT32)
                            gather_tmp = pl.create_tile([H_TILE, ROPE_DIM], dtype=pl.INT32)
                            for head_part in pl.unroll(H // 2 // H_TILE):
                                offset = head_part * H_TILE
                                head = head0 + offset
                                merged = pl.concat(pl.tile.slice(next_left, [H_TILE, HEAD_DIM // 2], [offset, 0]),
                                                    pl.tile.slice(next_right, [H_TILE, HEAD_DIM // 2], [offset, 0]))
                                normalized = pl.row_expand_div(
                                    merged, pl.tile.slice(denominator, [H_TILE, 1], [offset, 0]),
                                )
                                bf16 = pl.cast(normalized, pl.BF16, mode="rint")
                                rope = normalized[0:H_TILE, NOPE_DIM:HEAD_DIM]
                                swapped = pl.tile.gather(normalized, swap_index, gather_tmp)
                                rotated = pl.add(pl.col_expand_mul(rope, cosine), pl.col_expand_mul(swapped, sine))
                                rope_bf16 = pl.cast(rotated, pl.BF16, mode="rint")
                                full = pl.concat(bf16[0:H_TILE, 0:NOPE_DIM], rope_bf16)
                                groups = pl.reshape(full, [PUBLISH_GROUPS, O_GROUP_IN])
                                packed_row = (head // HEADS_PER_GROUP) * T_PAD + publish_token
                                pl.store(groups[0:1, :], [packed_row, 0], o_packed_heads)
                                pl.store(groups[1:2, :], [packed_row + T_PAD, 0], o_packed_heads)
                        m_after, l_after, left_after, right_after = pl.yield_(
                            next_m, next_l, next_left, next_right,
                        )
                    else:
                        m_after, l_after, left_after, right_after = pl.yield_(m_iter, l_iter, left_iter, right_iter)
                    running_m, running_l, running_left, running_right, max_ring_done, sum_ring_done = pl.yield_(
                        m_after, l_after, left_after, right_after, updated_max_ring, updated_sum_ring,
                    )
                offset_done, m_done, l_done, left_done, right_done, max_done, sum_done = pl.yield_(
                    vec_offset + work_count, running_m, running_l, running_left, running_right,
                    max_ring_done, sum_ring_done,
                )
    return o_packed_heads, attention_tid


@pl.jit.inline(auto_scope=False)
def sparse_attn_hca_tp1(
    q: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B_DYN, ORI_TABLE_COLUMNS_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    raw_cache_ready_dep: pl.Scalar[pl.TASK_ID],
    cmp_cache_ready_dep: pl.Scalar[pl.TASK_ID],
    q_ready: pl.Array[4, pl.TASK_ID],
) -> tuple[pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16], pl.Scalar[pl.TASK_ID]]:
    # 页表容量不超过四页时，全部压缩行确定在一个 K128 块内；更宽页表保持原路径。
    ready = pl.array.create(1, pl.TASK_ID)
    if pl.tensor.dim(cmp_block_table, 1) <= CMP_PAGES_PER_WORK:
        packed, short_ready = _short_sparse_attn_hca_tp1(
            q, ori_kv, ori_block_table, cmp_kv, cmp_block_table, position_ids,
            kv_seq_lens, attn_sink, freqs_cos, freqs_sin, o_packed_heads,
            raw_cache_ready_dep, cmp_cache_ready_dep, q_ready,
        )
        ready[0] = short_ready
    else:
        packed, long_ready = _long_sparse_attn_hca_tp1(
            q, ori_kv, ori_block_table, cmp_kv, cmp_block_table, position_ids,
            kv_seq_lens, attn_sink, freqs_cos, freqs_sin, o_packed_heads,
            raw_cache_ready_dep, cmp_cache_ready_dep, q_ready,
        )
        ready[0] = long_ready
    return packed, ready[0]
