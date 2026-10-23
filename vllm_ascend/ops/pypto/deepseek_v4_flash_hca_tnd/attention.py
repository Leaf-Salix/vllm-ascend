# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TND HCA attention：请求级 KV 聚合，token 级 QK/Softmax/PV 流水。

本版复用已验证 HCA 的三槽环形流水：AIC 提前发起 QK，两个 AIV 各处理一半
head 的 softmax 与输出归约；硬件事件连接 QK→概率→PV。跨 query 保留流水，
每个 worker 最后一条 query 才排空。不同 incore task 仍由 AICPU 管理依赖。

请求归属和 query 起点来自设备 token_geometry；不使用 token//6、token%6。
raw KV 的 133 行是 S≤6 的窗口容量，实际搬运行数取设备 Q/L；compressed
历史按各 token 的因果位置裁剪。失活 token 使用零有效长度，不写业务 cache。
短历史/小T采用K128并在L1复用KV；长历史较大T采用K512，减少跨块更新与PV写回。
两档共用请求gather、device Q/L、FFTS及输出/逆RoPE，仅编译期绑定不同计算块。
K512改变浮点舍入，使用原Native输出容差验证；选择区间见native_adapter.py。
"""


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
BOUNDS_DYN = pl.dynamic("HCA_QUERY_BOUNDS")


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
ROPE_CS_T_TILE = 8
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



def _make_sparse_attention(MACRO_K):
    """编译期策略工厂，不接收本轮Host数据，不产生TMR运行时策略分支。

    K128保留3份完整KV的L1环以供QK/PV复用；K512按D256供数、PV重载KV，
    用四份L0C完成宏块归约再写GM。两者均保留三槽、两步QK提前执行、跨query流水。
    每AIV的K128 softmax一次H32；K512用两次H16限制UB容量。输出累积与RoPE共用。
    """
    MACRO_PARTS = MACRO_K // ATTN_K_TILE
    @pl.jit.inline(auto_scope=False)
    def sparse_attn_hca_tp1(
        q: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
        ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
        ori_block_table: pl.Tensor[[B_DYN, ORI_TABLE_COLUMNS_DYN], pl.INT32],
        cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
        cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
        position_ids: pl.Tensor[[T_DYN], pl.INT64],
        kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
        query_bounds: pl.Tensor[[BOUNDS_DYN], pl.INT32],
        token_geometry: pl.Tensor[[T_DYN, 16], pl.INT32],
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
        rope_cs_blocks = (t_dim + ROPE_CS_T_TILE - 1) // ROPE_CS_T_TILE
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
            g_t0 = pl.cast(pl.read(query_bounds, [g_req]), pl.INDEX)
            g_t1 = pl.min(pl.cast(pl.read(query_bounds, [g_req + 1]), pl.INDEX), t_dim)
            g_queries = pl.max(g_t1 - g_t0, 0)
            g_base = g_req * REQUEST_KV_ROWS
            g_length = pl.read(kv_seq_lens, [g_req])
            g_position = pl.read(position_ids, [pl.min(g_t0, t_dim - 1)])
            g_first_len = pl.min(WIN, pl.max(pl.min(g_position + 1, g_length), 0))
            g_start = pl.max(g_position + 1 - g_first_len, 0)
            raw_kv[g_base:g_base + REQUEST_KV_ROWS, :] = pl.full([REQUEST_KV_ROWS, HEAD_DIM], dtype=pl.BF16, value=0.0)
            if g_length > 0 and g_queries > 0:
                # 一个 Native 页内的行物理连续；首尾页只复制有效行，不读取未初始化 KV。
                g_rows = pl.min(g_first_len + g_queries - 1, pl.max(g_length - g_start, 0))
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

        # T 与请求数量独立；尾块只发布真实容量内行，metadata 的有效长度排除 dummy。
        with pl.spmd((t_dim + VALID_TOKEN_TILE - 1) // VALID_TOKEN_TILE, name_hint="hca_raw_valid", allow_early_resolve=True,
                     deps=[ori_cache_ready_dep]) as raw_valid_tid:
            valid_t0 = pl.tile.get_block_idx() * VALID_TOKEN_TILE
            valid_col = pl.cast(pl.tile.arange(0, [1, WIN], dtype=pl.INT32), pl.FP32)
            for valid_dt in pl.range(pl.min(VALID_TOKEN_TILE, t_dim - valid_t0)):
                valid_token = valid_t0 + valid_dt
                valid_length = pl.min(WIN, pl.max(pl.min(
                    pl.read(position_ids, [valid_token]) + 1,
                    pl.read(token_geometry, [valid_token, 3]),
                ), 0))
                valid_count_f32 = pl.cast(pl.cast(valid_length, pl.INT32), pl.FP32)
                valid_mask = pl.minimum(pl.maximum(pl.neg(pl.sub(valid_col, valid_count_f32)), 0.0), 1.0)
                pl.store(valid_mask, [valid_token, 0], raw_valid)

        # Native 已提供交错的 FP32 频率；inverse RoPE 仅在消费者内折叠符号。
        with pl.spmd(rope_cs_blocks, name_hint="hca_inverse_rope_sign", allow_early_resolve=True, deps=[ori_cache_ready_dep]) as rope_cs_tid:
            cs_t0 = pl.tile.get_block_idx() * ROPE_CS_T_TILE
            cs_index = pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), pl.FP32)
            cs_pair = pl.cast(pl.cast(pl.mul(cs_index, 0.5), pl.INT32, mode="trunc"), pl.FP32)
            cs_odd = pl.sub(cs_index, pl.mul(cs_pair, 2.0))
            cs_sign = pl.neg(pl.sub(pl.mul(cs_odd, 2.0), 1.0))
            cs_count = pl.min(ROPE_CS_T_TILE, t_dim - cs_t0)
            cs_cosine = pl.load(freqs_cos, [cs_t0, 0], [ROPE_CS_T_TILE, ROPE_DIM], valid_shape=[cs_count, ROPE_DIM])
            cs_sine = pl.load(freqs_sin, [cs_t0, 0], [ROPE_CS_T_TILE, ROPE_DIM], valid_shape=[cs_count, ROPE_DIM])
            pl.store(cs_cosine, [cs_t0, 0], rope_cos_il)
            pl.store(pl.col_expand_mul(cs_sine, cs_sign), [cs_t0, 0], rope_sin_signed)


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
        scores = pl.create_tensor([transfer_rows, MACRO_K], dtype=pl.FP32)
        probs = pl.create_tensor([transfer_rows, MACRO_K], dtype=pl.BF16)
        values = pl.create_tensor([transfer_rows, HEAD_DIM], dtype=pl.FP32)
        ffts = pl.create_tensor([256], dtype=pl.INT64)
        sink_col = pl.reshape(attn_sink, [H, 1])
        with pl.spmd(
            NUM_QK_CORES, name_hint="hca_unified_attention",
            deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid, q_ready], allow_early_resolve=True,
        ) as attention_tid:
            worker = pl.tile.get_block_idx()
            pl.system.set_ffts(ffts)
            # 编译期只保留选中计算体，不在核内增加策略判断或额外任务。
            if MACRO_K == ATTN_K_TILE:
                # QK进入下一query时保留上一query的KV槽；只在本worker最后一条query排空。
                kv_l1 = pl.create_tile([QK_TRANSFER_SLOTS * ATTN_K_TILE, HEAD_DIM],
                                      dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
                for token, (work_offset,) in pl.range(
                    worker, t_dim, NUM_QK_CORES, init_values=(pl.cast(0, pl.INDEX),),
                ):
                    request = pl.cast(pl.read(token_geometry, [token, 0]), pl.INDEX)
                    query_begin = pl.cast(pl.read(token_geometry, [token, 1]), pl.INDEX)
                    length = pl.max(pl.read(token_geometry, [token, 3]), 0)
                    position = pl.max(pl.read(position_ids, [token]), -1)
                    cmp_rows = pl.cast(pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
                        (position + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO,
                    )), pl.INDEX)
                    cmp_blocks = pl.min(cmp_work_count, (cmp_rows + ATTN_K_TILE - 1) // ATTN_K_TILE)
                    work_count = 1 + cmp_blocks
                    first_length = pl.min(WIN, pl.max(pl.min(pl.read(position_ids, [pl.min(query_begin, t_dim - 1)]) + 1, length), 0))
                    raw_drop = pl.min(S - 1, pl.max(first_length + (token - query_begin) - WIN, 0))
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
            else:
                # 两级Scalar FIFO保存前两项KV来源；都是同一AIC的局部循环变量。
                # 它们不是GM完成标志，不涉及跨核Scalar缓存一致性；CV交接仍用硬件事件。
                for token, (work_offset, older_src, older_parts, older_raw,
                            recent_src, recent_parts, recent_raw) in pl.range(
                    worker, t_dim, NUM_QK_CORES,
                    init_values=(pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX),
                                 pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX)),
                ):
                    request = pl.cast(pl.read(token_geometry, [token, 0]), pl.INDEX)
                    query_begin = pl.cast(pl.read(token_geometry, [token, 1]), pl.INDEX)
                    length = pl.max(pl.read(token_geometry, [token, 3]), 0)
                    position = pl.max(pl.read(position_ids, [token]), -1)
                    cmp_rows = pl.cast(pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
                        (position + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO,
                    )), pl.INDEX)
                    cmp_rows = pl.min(cmp_rows, cmp_work_count * ATTN_K_TILE)
                    cmp_blocks = (cmp_rows + MACRO_K - 1) // MACRO_K
                    work_count = cmp_blocks + 1
                    first_length = pl.min(WIN, pl.max(pl.min(pl.read(position_ids, [pl.min(query_begin, t_dim - 1)]) + 1, length), 0))
                    raw_drop = pl.min(S - 1, pl.max(first_length + (token - query_begin) - WIN, 0))
                    query = pl.load(q_flat, [token * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                    if token + NUM_QK_CORES >= t_dim:
                        extra = pl.yield_(QK_PRE_LAUNCH)
                    else:
                        extra = pl.yield_(0)
                    for tick, (old_src, old_parts, old_raw, new_src, new_parts, new_raw) in pl.range(
                        work_count + extra,
                        init_values=(older_src, older_parts, older_raw, recent_src, recent_parts, recent_raw),
                    ):
                        global_tick = work_offset + tick
                        # FIFO每拍前移，来源用直接Scalar表达式计算，避免条件yield的地址别名。
                        # 排空拍将描述符夹到最后raw块；只有已生产的前两拍才会被PV消费。
                        is_raw = pl.min(1, pl.max(0, tick - cmp_blocks + 1))
                        cmp_source = (request * cmp_work_count + tick * MACRO_PARTS) * ATTN_K_TILE
                        raw_source = pl.cast(request * REQUEST_KV_ROWS + raw_drop, pl.INDEX)
                        source = cmp_source * (1 - is_raw) + raw_source * is_raw
                        compressed_parts = pl.max(0, pl.min(MACRO_PARTS, (cmp_rows - tick * MACRO_K + ATTN_K_TILE - 1) // ATTN_K_TILE))
                        parts = compressed_parts * (1 - is_raw) + is_raw
                        if tick < work_count:
                            row = (worker * QK_TRANSFER_SLOTS + global_tick % QK_TRANSFER_SLOTS) * H
                            # 宏块内仍是N128微块。D分成两个256列L1块，分别用K128 L0双缓冲。
                            # 不再用一个D512的L1临时块阻塞整段QK供数。
                            for part in pl.range(parts):
                                seed = pl.create_tile([H, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                                for half, (acc_half,) in pl.pipeline(2, stage=2, init_values=(seed,)):
                                    dim0 = half * 256
                                    if is_raw == 1:
                                        raw_piece = pl.load(raw_kv, [request * REQUEST_KV_ROWS + raw_drop + part * ATTN_K_TILE, dim0],
                                                            [ATTN_K_TILE, 256], target_memory=pl.MemorySpace.Mat)
                                        piece = pl.yield_(raw_piece)
                                    else:
                                        cmp_piece = pl.load(cmp_work_kv, [(request * cmp_work_count + tick * MACRO_PARTS + part) * ATTN_K_TILE, dim0],
                                                            [ATTN_K_TILE, 256], target_memory=pl.MemorySpace.Mat)
                                        piece = pl.yield_(cmp_piece)
                                    transposed = pl.tile.transpose_view(piece)
                                    for inner, (acc_inner,) in pl.pipeline(2, stage=2, init_values=(acc_half,)):
                                        lhs = pl.tile.extract(query, 0, dim0 + inner * 128, [H, 128],
                                                              target_memory=pl.MemorySpace.Left)
                                        rhs = pl.tile.extract(transposed, inner * 128, 0, [128, ATTN_K_TILE],
                                                              target_memory=pl.MemorySpace.Right)
                                        updated = pl.tile.matmul_acc(acc_inner, lhs, rhs, init_cond=(half == 0 and inner == 0))
                                        inner_done = pl.yield_(updated)
                                    half_done = pl.yield_(inner_done)
                                pl.store(half_done, [row, part * ATTN_K_TILE], scores)
                            pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                        if global_tick >= QK_PRE_LAUNCH:
                            pv_row = (worker * QK_TRANSFER_SLOTS + (global_tick - QK_PRE_LAUNCH) % QK_TRANSFER_SLOTS) * H
                            pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                            # 四个N128结果覆盖D512；完整宏块的K累加完成后才写GM。
                            # 每个Acc32KiB，四份合计128KiB，不另保留一份全宽临时结果。
                            seed0 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                            seed1 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                            seed2 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                            seed3 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                            for part, (acc0, acc1, acc2, acc3) in pl.range(
                                old_parts, init_values=(seed0, seed1, seed2, seed3),
                            ):
                                probability = pl.load(probs, [pv_row, part * ATTN_K_TILE], [H, ATTN_K_TILE],
                                                      target_memory=pl.MemorySpace.Mat)
                                left = pl.tile.move(probability, target_memory=pl.MemorySpace.Left)
                                # 先提供前半D，让MTE1可以提前开始；后半D使用另一个64KiB L1块。
                                if old_raw == 1:
                                    raw0 = pl.load(raw_kv, [old_src + part * ATTN_K_TILE, 0], [ATTN_K_TILE, 256],
                                                   target_memory=pl.MemorySpace.Mat)
                                    value0 = pl.yield_(raw0)
                                else:
                                    cmp0 = pl.load(cmp_work_kv, [old_src + part * ATTN_K_TILE, 0], [ATTN_K_TILE, 256],
                                                   target_memory=pl.MemorySpace.Mat)
                                    value0 = pl.yield_(cmp0)
                                right0 = pl.tile.extract(value0, 0, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                                if old_raw == 1:
                                    raw1 = pl.load(raw_kv, [old_src + part * ATTN_K_TILE, 256], [ATTN_K_TILE, 256],
                                                   target_memory=pl.MemorySpace.Mat)
                                    value1 = pl.yield_(raw1)
                                else:
                                    cmp1 = pl.load(cmp_work_kv, [old_src + part * ATTN_K_TILE, 256], [ATTN_K_TILE, 256],
                                                   target_memory=pl.MemorySpace.Mat)
                                    value1 = pl.yield_(cmp1)
                                right1 = pl.tile.extract(value0, 0, PV_N_TILE, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                                update0 = pl.tile.matmul_acc(acc0, left, right0, init_cond=(part == 0))
                                right2 = pl.tile.extract(value1, 0, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                                update1 = pl.tile.matmul_acc(acc1, left, right1, init_cond=(part == 0))
                                right3 = pl.tile.extract(value1, 0, PV_N_TILE, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                                update2 = pl.tile.matmul_acc(acc2, left, right2, init_cond=(part == 0))
                                update3 = pl.tile.matmul_acc(acc3, left, right3, init_cond=(part == 0))
                                done0, done1, done2, done3 = pl.yield_(update0, update1, update2, update3)
                            pl.store(done0, [pv_row, 0], values)
                            pl.store(done1, [pv_row, PV_N_TILE], values)
                            pl.store(done2, [pv_row, 2 * PV_N_TILE], values)
                            pl.store(done3, [pv_row, 3 * PV_N_TILE], values)
                            pl.system.sync_set(2, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                        old_src_done, old_parts_done, old_raw_done, new_src_done, new_parts_done, new_raw_done = pl.yield_(
                            new_src, new_parts, new_raw, source, parts, is_raw,
                        )
                    offset_done, older_src_done, older_parts_done, older_raw_done, recent_src_done, recent_parts_done, recent_raw_done = pl.yield_(
                        work_offset + work_count, old_src_done, old_parts_done, old_raw_done,
                        new_src_done, new_parts_done, new_raw_done,
                    )

            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(ffts)
                head0 = lane * (H // 2)
                reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32)
                stream_seed_m = pl.load(sink_col, [head0, 0], [H // 2, 1])
                stream_seed_l = pl.mul(stream_seed_m, 0.0)
                stream_seed_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                stream_seed_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                # 延迟 PV 期间，softmax 最大值/分母保留在本 AIV 的 UB 环内。
                # 这些归约状态不跨核；只有 score、概率和 PV 结果通过 GM 交接。
                max_ring_init = pl.tile.full([QK_TRANSFER_SLOTS * (H // 2 // H_TILE), H_TILE], dtype=pl.FP32, value=0.0)
                sum_ring_init = pl.tile.full([QK_TRANSFER_SLOTS * (H // 2 // H_TILE), H_TILE], dtype=pl.FP32, value=0.0)
                query_ring = pl.tile.full([QK_TRANSFER_SLOTS, 8], dtype=pl.INT32, value=0)
                for vec_token, (vec_offset, m_query, l_query, left_query, right_query, max_query, sum_query) in pl.range(
                    worker, t_dim, NUM_QK_CORES,
                    init_values=(pl.cast(0, pl.INDEX), stream_seed_m, stream_seed_l,
                                 stream_seed_left, stream_seed_right, max_ring_init, sum_ring_init),
                ):
                    request = pl.cast(pl.read(token_geometry, [vec_token, 0]), pl.INDEX)
                    query_begin = pl.cast(pl.read(token_geometry, [vec_token, 1]), pl.INDEX)
                    length = pl.max(pl.read(token_geometry, [vec_token, 3]), 0)
                    position = pl.max(pl.read(position_ids, [vec_token]), -1)
                    cmp_rows = pl.cast(pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(
                        (position + 1) // COMPRESS_RATIO, length // COMPRESS_RATIO,
                    )), pl.INDEX)
                    cmp_rows = pl.min(cmp_rows, cmp_work_count * ATTN_K_TILE)
                    cmp_blocks = (cmp_rows + MACRO_K - 1) // MACRO_K
                    work_count = 1 + cmp_blocks
                    first_length = pl.min(WIN, pl.max(pl.min(pl.read(position_ids, [pl.min(query_begin, t_dim - 1)]) + 1, length), 0))
                    raw_drop = pl.min(S - 1, pl.max(first_length + (vec_token - query_begin) - WIN, 0))
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
                            if MACRO_K == ATTN_K_TILE:
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
                                    max_ring, pl.reshape(maximum, [H // 2 // H_TILE, H_TILE]), [(vec_global % QK_TRANSFER_SLOTS) * (H // 2 // H_TILE), 0],
                                )
                                next_sum_ring = pl.tile.assemble(
                                    sum_ring, pl.reshape(total, [H // 2 // H_TILE, H_TILE]), [(vec_global % QK_TRANSFER_SLOTS) * (H // 2 // H_TILE), 0],
                                )
                                ready_max, ready_sum = pl.yield_(next_max_ring, next_sum_ring)
                            else:
                                # 每个AIV逐次处理两个H16片，限制大列块的UB工作集。
                                for head_part, (head_max_ring, head_sum_ring) in pl.range(
                                    H // 2 // H_TILE, init_values=(max_ring, sum_ring),
                                ):
                                    head_offset = head_part * H_TILE
                                    if vec_tick == cmp_blocks:
                                        raw_tmp = pl.create_tile([H_TILE, ATTN_K_TILE], dtype=pl.FP32)
                                        raw_score = pl.load(scores, [vec_row + head_offset, 0], [H_TILE, ATTN_K_TILE])
                                        raw_mask = pl.load(raw_valid, [vec_token, 0], [1, ATTN_K_TILE])
                                        raw_scaled = pl.col_expand_add(pl.mul(raw_score, SOFTMAX_SCALE),
                                                                      pl.mul(pl.sub(raw_mask, 1.0), -NEG_INF))
                                        raw_max = pl.row_max(raw_scaled, raw_tmp)
                                        raw_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(raw_scaled, raw_max)), raw_mask)
                                        raw_sum = pl.row_sum(raw_exp, raw_tmp)
                                        pl.store(pl.cast(raw_exp, pl.BF16, mode="rint"), [vec_row + head_offset, 0], probs)
                                        maximum, total = pl.yield_(raw_max, raw_sum)
                                    else:
                                        cmp_tmp = pl.create_tile([H_TILE, MACRO_K], dtype=pl.FP32)
                                        valid_cols = pl.min(MACRO_K, cmp_rows - vec_tick * MACRO_K)
                                        mask_parts = (valid_cols + ATTN_K_TILE - 1) // ATTN_K_TILE
                                        mask_grid = pl.load(cmp_work_valid,
                                                            [request * cmp_work_count + vec_tick * MACRO_PARTS, 0],
                                                            [MACRO_PARTS, ATTN_K_TILE],
                                                            valid_shape=[mask_parts, ATTN_K_TILE])
                                        cmp_mask = pl.reshape(pl.fillpad(mask_grid, pad_value=pl.PadValue.zero), [1, MACRO_K])
                                        score = pl.load(scores, [vec_row + head_offset, 0], [H_TILE, MACRO_K],
                                                        valid_shape=[H_TILE, valid_cols])
                                        scaled = pl.col_expand_add(pl.mul(score, SOFTMAX_SCALE),
                                                                   pl.mul(pl.sub(cmp_mask, 1.0), -NEG_INF))
                                        masked = pl.fillpad(pl.set_validshape(scaled, H_TILE, valid_cols), pad_value=pl.PadValue.min)
                                        cmp_max = pl.row_max(masked, cmp_tmp)
                                        cmp_exp = pl.col_expand_mul(pl.exp(pl.row_expand_sub(masked, cmp_max)), cmp_mask)
                                        cmp_sum = pl.row_sum(cmp_exp, cmp_tmp)
                                        pl.store(pl.cast(cmp_exp, pl.BF16, mode="rint"), [vec_row + head_offset, 0], probs)
                                        maximum, total = pl.yield_(cmp_max, cmp_sum)
                                    row_slot = (vec_global % QK_TRANSFER_SLOTS) * (H // 2 // H_TILE) + head_part
                                    next_max_ring = pl.tile.assemble(
                                        head_max_ring, pl.reshape(maximum, [1, H_TILE]), [row_slot, 0],
                                    )
                                    next_sum_ring = pl.tile.assemble(
                                        head_sum_ring, pl.reshape(total, [1, H_TILE]), [row_slot, 0],
                                    )
                                    completed_max, completed_sum = pl.yield_(next_max_ring, next_sum_ring)
                                ready_max, ready_sum = pl.yield_(completed_max, completed_sum)
                            # 描述符仅供本AIV消费；PV延迟两步时仍指向正确query，不增加GM任务。
                            pl.tile.write(query_ring, [vec_global % QK_TRANSFER_SLOTS, 0], pl.cast(vec_token, pl.INT32))
                            pl.tile.write(query_ring, [vec_global % QK_TRANSFER_SLOTS, 1], pl.cast(vec_tick, pl.INT32))
                            pl.tile.write(query_ring, [vec_global % QK_TRANSFER_SLOTS, 2],
                                          pl.cast(work_count - vec_tick - 1, pl.INT32))
                            pl.system.sync_set(1, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                            updated_max_ring, updated_sum_ring = pl.yield_(ready_max, ready_sum)
                        else:
                            updated_max_ring, updated_sum_ring = pl.yield_(max_ring, sum_ring)
                        if vec_global >= QK_PRE_LAUNCH:
                            out_work = vec_global - QK_PRE_LAUNCH
                            out_row = (worker * QK_TRANSFER_SLOTS + out_work % QK_TRANSFER_SLOTS) * H + head0
                            pl.system.sync_wait(2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                            out_m = pl.reshape(pl.tile.extract(
                                updated_max_ring, (out_work % QK_TRANSFER_SLOTS) * (H // 2 // H_TILE), 0, [H // 2 // H_TILE, H_TILE],
                                target_memory=pl.MemorySpace.Vec,
                            ), [H // 2, 1])
                            out_l = pl.reshape(pl.tile.extract(
                                updated_sum_ring, (out_work % QK_TRANSFER_SLOTS) * (H // 2 // H_TILE), 0, [H // 2 // H_TILE, H_TILE],
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

    return sparse_attn_hca_tp1


sparse_attn_hca_tp1 = _make_sparse_attention(ATTN_K_TILE)
sparse_attn_hca_tp1_long = _make_sparse_attention(512)
