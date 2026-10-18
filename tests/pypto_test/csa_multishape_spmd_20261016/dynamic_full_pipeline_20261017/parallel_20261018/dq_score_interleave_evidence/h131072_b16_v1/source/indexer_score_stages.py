# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""动态CSA Indexer的成对Cube/Vector score流水：请求内分组、两槽READY/ACK。

阅读顺序与范围
    indexer_score_cube_stage在AIC执行INT8 QK、FIX缩放/ReLU至FP16，再用
    FP16 head系数在Cube归约为FP32；indexer_score_vector_lane_stage由同组
    两AIV读取分数、乘Native key scale并生成叶Top512根。二者必须在一个
    MIX内并发推进，不能先等完整AIC阶段后才调用AIV。本模块不含short Vector
    score、最终query Top-K发布或完整CSA入口；不创建spmd/task/阶段通知。

动态布局和分组
    Q/L/positions与group_offsets/plan来自正式indexer_group_stages，T/B、
    每请求query数、历史和所有arena容量保留动态。组在单请求内切G=2/6，
    最后live=min(G,request_end-first)，不读下一请求或padding query来凑整组。
    AIC query INT8[HEAD_ROWS,128]按原token*64排列；系数FP16[COEFF_ROWS,G*64]
    每组16行。尾组只从GM读取live条query，在L1用首条有效query填充其余
    片上行；对应系数已由生产者置零，不产生额外输出。QK始终使用完整物理M，
    head归约使用完整物理K，避免动态缩M产生紧凑L0C后，A3 TINSERT仍按物理
    行距读回造成错位。这是片上tile补齐，不能读取下一请求/越界GM来补齐。
    完整组保留原QK/WS顺序；尾组仍仅发布live条query的score/Top-K。
    cache沿Native INT8[pages,page_bytes]布局：4096B key后是64B FP16 scale。
    key0/key64是外层绑定的只读二维别名；page_bytes>=4160且64B对齐，不在
    incore内reshape整块GM。页表列和物理页先检查再读取；无效页保护不表示
    非法metadata具有已定义的正确Top-K数学，真实visible历史页必须合法。

策略、分核与工作区
    G、QK panel、score_tile768/1024、prefetch和balanced均为显式策略，
    必须与plan的G/balanced/roots配置一致，不能按T重新猜。常驻核组数固定24。
    所有角色用同一work_item/leaf_range计算core按24条带领取的(group,leaf)。
    score_transfer FP32[>=24*2*G,score_tile]每core双槽；仅AIC写、两AIV读。
    score_arena FP32[>=24*2*G,8192]是非balanced AIV私有半叶scratch，
    行号core*2*G+query_lane*2+lane，各自覆盖有效前缀后才排序，不读取旧尾。
    pair_arena FP32[>=T*64,1024]每query固定64行：两根布局写leaf*2+lane，
    一根布局写leaf；后继按同一plan和真实query visibility只合并有效根。
    balanced AIV各管G/2个完整query，按两个1024连续块排序再以前新后旧次序
    合并根，保留同分规则。未实际产出的arena容量尾部不能作为有效数据消费。

主Query穿插候选
    INTERLEAVE_QUERY_DQ开启时，在balanced偶数步和非balanced每步已ACK后，
    用原DQ owner推进最多8head；余量在本核Score结束后补齐。只携带两个Scalar
    进度，共享query_projection_stages的完整数学，不增加GM队列或核间同步。
    此候选尚未设备验收，完整前驱获取和QUERY发布仍由外层负责。

同步、资源与边界
    外层先等query、系数、Native cache的key/scale和plan全部发布，并在同组
    AIC/双AIV使用同一FFTS事件域。测试壳显式set_ffts；当前PyPTO预处理会
    删除set_ffts_base_addr，实际base由Simpler入口设置，不能把形参workspace
    当作一块独立硬件队列。READY事件由FIX发布；两AIV MTE2等待
    后读取自己的FIFO片段，再以MTE2各发ACK。AIC第2轮起等ACK后复用槽，
    每leaf末排空min(steps,2)。尾组中无有效query的AIV也必须读取FIFO并ACK，
    只能跳过query数学/根写回，否则AIC死锁。外层完成phase不是这个消费ACK。
    同时存活的QK/WS与两Key L0B沿原tile布局，不增加永久L1/L0 reserve。
    所有输入metadata在本次阶段中固定；工作区只有协议规定的owner写。
    返回别名不等于跨核完成。当前源码/CPU证据不能代替设备同步、精度和性能。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.config import FP32_NEG_INF
from .indexer_group_stages import (
    COEFF_COLS_DYN,
    COEFF_ROWS_DYN,
    OFFSET_ROWS_DYN,
    PLAN_WORDS,
    TOKENS_DYN,
    indexer_decode_group,
    indexer_leaf_range,
    indexer_query_visibility,
    indexer_work_item,
)
from .indexer_topk_stages import PAIR_ROWS_DYN, SCORE_ROWS_DYN, indexer_topk_half_leaf
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN
from .query_projection_stages import (
    H as Q_HEADS, HEAD_DIM as Q_HEAD_DIM, ROPE_DIM as Q_ROPE_DIM,
    T_DYN as Q_T_DYN, PAD_ROWS_DYN as Q_PAD_ROWS_DYN, QPROJ_ROWS_DYN,
    Q_ROPE_T_TILE, q_dequant_head_group_size, q_dequant_rope_block_stage,
    q_dequant_rope_chunk_stage,
)

QUERY_HEAD_ROWS_DYN = pl.dynamic("CSA_SCORE_QUERY_HEAD_ROWS")
CACHE_PAGES_DYN = pl.dynamic("CSA_SCORE_CACHE_PAGES")
PAGE_BYTES_DYN = pl.dynamic("CSA_SCORE_PAGE_BYTES")
KEY0_ROWS_DYN = pl.dynamic("CSA_SCORE_KEY0_ROWS")
KEY64_ROWS_DYN = pl.dynamic("CSA_SCORE_KEY64_ROWS")
TABLE_COLUMNS_DYN = pl.dynamic("CSA_SCORE_TABLE_COLUMNS")
TRANSFER_ROWS_DYN = pl.dynamic("CSA_SCORE_TRANSFER_ROWS")
TRANSFER_COLS_DYN = pl.dynamic("CSA_SCORE_TRANSFER_COLS")
IDX_N_HEADS = 64
IDX_HEAD_DIM = 128
BLOCK_SIZE = 32
COMPRESS_RATIO = 4
INDEXER_KEY_BYTES = 4096
TOPK_CANDIDATES_PER_LEAF = 8192
TOPK_PAIR_WIDTH = 1024
TOPK_ROWS_PER_QUERY = 64
TOPK_SCORE_WORKERS = 24
BUFFERED_LONG_SCORE_TILE = 1024
BUFFERED_SCORE_SCALE = 1024.0
NATIVE_QLI_WEIGHT_ROWS = 16


@pl.jit.inline(auto_scope=False)
def indexer_score_cube_stage(
    qr_hadamard_i8: pl.Tensor[[QUERY_HEAD_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    coefficients: pl.Tensor[[COEFF_ROWS_DYN, COEFF_COLS_DYN], pl.FP16],
    idx_kv_cache: pl.Tensor[[KEY0_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_cache_shift64: pl.Tensor[[KEY64_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_native_kv_cache: pl.Tensor[[CACHE_PAGES_DYN, PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    position_ids: pl.Tensor[[TOKENS_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    buf_score_transfer: pl.InOut[pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32]],
    buf_worker: pl.Scalar[pl.INDEX],
    query_group_size: pl.constexpr,
    qk_panel_cols: pl.constexpr,
    score_tile: pl.constexpr,
    key_prefetch_panels: pl.constexpr,
    key_prefetch_to_l0: pl.constexpr,
    balance_leaves: pl.constexpr,
    SCORE_READY_EVENT: pl.constexpr,
    SCORE_CONSUMED_EVENT: pl.constexpr,
):
    """AIC生产FP32 score双槽；须与两个vector_lane同MIX并发，外层/运行时已设置同一FFTS域。"""
    native_page_bytes = pl.tensor.dim(idx_native_kv_cache, 1)
    groups = pl.cast(pl.read(plan, [0]), pl.INDEX)
    buf_max_leaves = pl.cast(pl.read(plan, [2]), pl.INDEX)
    buf_leaf_tiles = pl.cast(pl.read(plan, [3]), pl.INDEX)
    buf_extra_leaves = pl.cast(pl.read(plan, [4]), pl.INDEX)
    # 每个core独立读取同一device计划；空core不触发任何FFTS事件。
    for buf_item in pl.range(buf_worker, groups * buf_max_leaves, TOPK_SCORE_WORKERS):
        buf_group, buf_leaf = indexer_work_item(buf_item, groups, buf_max_leaves)
        buf_batch_idx, buf_query, buf_live, buf_visible_count = indexer_decode_group(
            bounds, kv_seq_lens, position_ids, offsets, buf_group, query_group_size
        )
        buf_logical_begin, buf_leaf_capacity, work_valid = indexer_leaf_range(
            buf_leaf, buf_leaf_tiles, buf_extra_leaves, buf_visible_count
        )
        if work_valid > 0:
            buf_cache_len = pl.cast(pl.read(kv_seq_lens, [buf_batch_idx]), pl.INDEX) // COMPRESS_RATIO
            buf_valid_count = pl.min(buf_leaf_capacity, buf_visible_count - buf_logical_begin)
            buf_tile_count = (buf_valid_count + score_tile - 1) // score_tile
            buf_lane_span = pl.min(buf_tile_count * (score_tile // 2), TOPK_CANDIDATES_PER_LEAF // 2)
            buf_score_iters = (buf_lane_span + score_tile // 2 - 1) // (score_tile // 2)
            buf_query_vector = pl.load(
                qr_hadamard_i8,
                [buf_query * IDX_N_HEADS, 0],
                [query_group_size * IDX_N_HEADS, IDX_HEAD_DIM],
                target_memory=pl.MemorySpace.Mat,
                valid_shape=[buf_live * IDX_N_HEADS, IDX_HEAD_DIM],
            )
            # A3当前TINSERT(Acc→Mat)忽略紧凑L0C的动态行距：若QK仅算
            # live*64行而物理M=G*64，每个16列分形之后都会读错源地址。
            # 因此在L1安全补满M，再做完整QK。补齐行只复制本组首条有效
            # query，绝不访问下一请求；对应head系数为0，整数点积/ReLU
            # 均有限，补齐行不会影响任何真实query。无需Host分支或GM搬运。
            buf_query_vector = pl.set_validshape(
                buf_query_vector, query_group_size * IDX_N_HEADS, IDX_HEAD_DIM
            )
            for padding_lane in pl.unroll(query_group_size):
                if padding_lane >= buf_live:
                    buf_query_vector = pl.gather_row(
                        buf_query_vector,
                        qr_hadamard_i8,
                        [padding_lane * IDX_N_HEADS, 0],
                        [buf_query * IDX_N_HEADS, 0],
                        [IDX_N_HEADS, IDX_HEAD_DIM],
                    )
            buf_coefficients_l1 = pl.load(
                coefficients,
                [buf_group * NATIVE_QLI_WEIGHT_ROWS, 0],
                [NATIVE_QLI_WEIGHT_ROWS, query_group_size * IDX_N_HEADS],
                target_memory=pl.MemorySpace.Mat,
            )
            buf_query_left = pl.tile.move(buf_query_vector, target_memory=pl.MemorySpace.Left)
            buf_coefficient_pair = pl.tile.move(buf_coefficients_l1, target_memory=pl.MemorySpace.Left)
            # 双槽最多允许两份未消费分数；第2轮起等两个AIV的MTE2 ACK后才重用。
            for buf_score_step in pl.range(buf_score_iters):
                if buf_score_step >= 2:
                    pl.system.sync_wait(SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                buf_score_begin = buf_score_step * (score_tile // 2)
                if balance_leaves:
                    buf_score_begin = buf_score_step * score_tile
                buf_transfer_row = buf_worker * 2 * query_group_size + buf_score_step % 2 * query_group_size
                buf_previous_l1 = pl.tile.create(
                    [query_group_size * IDX_N_HEADS, qk_panel_cols], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat
                )
                buf_current_key = pl.tile.create(
                    [IDX_HEAD_DIM, qk_panel_cols], dtype=pl.INT8, target_memory=pl.MemorySpace.Right
                )
                buf_prefetched_key = pl.tile.create(
                    [IDX_HEAD_DIM, qk_panel_cols], dtype=pl.INT8, target_memory=pl.MemorySpace.Right
                )
                buf_key_l1_pool = pl.tile.create(
                    [2 * qk_panel_cols, IDX_HEAD_DIM], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat
                )
                if key_prefetch_panels > 0:
                    buf_prime_col = 0
                    for buf_prime_key_page in pl.unroll(qk_panel_cols // BLOCK_SIZE):
                        buf_prime_key_row = (
                            buf_logical_begin
                            + buf_score_begin
                            + buf_prime_col // (score_tile // 2) * buf_lane_span
                            + buf_prime_col % (score_tile // 2)
                            + buf_prime_key_page * BLOCK_SIZE
                        )
                        if balance_leaves:
                            buf_prime_key_row = (
                                buf_logical_begin + buf_score_begin + buf_prime_col + buf_prime_key_page * BLOCK_SIZE
                            )
                        buf_prime_safe_page = pl.min(
                            buf_prime_key_row // BLOCK_SIZE, pl.max((buf_cache_len - 1) // BLOCK_SIZE, 0)
                        )
                        if buf_prime_safe_page >= 0 and buf_prime_safe_page < pl.tensor.dim(idx_block_table, 1):
                            buf_prime_physical_page = pl.cast(
                                pl.read(idx_block_table, [buf_batch_idx, buf_prime_safe_page]), pl.INDEX
                            )
                            if buf_prime_physical_page >= 0 and buf_prime_physical_page < pl.tensor.dim(
                                idx_native_kv_cache, 0
                            ):
                                buf_prime_native_byte = buf_prime_physical_page * native_page_bytes
                                if buf_prime_native_byte % IDX_HEAD_DIM == 0:
                                    buf_key_l1_pool = pl.gather_row(
                                        buf_key_l1_pool,
                                        idx_kv_cache,
                                        [buf_prime_key_page * BLOCK_SIZE, 0],
                                        [buf_prime_native_byte // IDX_HEAD_DIM, 0],
                                        [BLOCK_SIZE, IDX_HEAD_DIM],
                                    )
                                else:
                                    buf_key_l1_pool = pl.gather_row(
                                        buf_key_l1_pool,
                                        idx_kv_cache_shift64,
                                        [buf_prime_key_page * BLOCK_SIZE, 0],
                                        [buf_prime_native_byte // IDX_HEAD_DIM, 0],
                                        [BLOCK_SIZE, IDX_HEAD_DIM],
                                    )
                    if key_prefetch_to_l0:
                        buf_current_key = pl.tile.extract(
                            pl.tile.transpose_view(buf_key_l1_pool),
                            0,
                            0,
                            [IDX_HEAD_DIM, qk_panel_cols],
                            target_memory=pl.MemorySpace.Right,
                        )
                # 固定panel顺序：本panel QK与前panel head归约同时存活，保留FP16舍入位置。
                for buf_qk_panel in pl.unroll(score_tile // qk_panel_cols):
                    buf_panel_col = buf_qk_panel * qk_panel_cols
                    buf_load_col = (buf_qk_panel + key_prefetch_panels) * qk_panel_cols
                    buf_load_slot = (buf_qk_panel + key_prefetch_panels) % 2 * qk_panel_cols
                    if key_prefetch_panels > 0:
                        if not key_prefetch_to_l0:
                            buf_current_key = pl.tile.extract(
                                pl.tile.transpose_view(buf_key_l1_pool),
                                0,
                                buf_qk_panel % 2 * qk_panel_cols,
                                [IDX_HEAD_DIM, qk_panel_cols],
                                target_memory=pl.MemorySpace.Right,
                            )
                    if buf_qk_panel + key_prefetch_panels < score_tile // qk_panel_cols:
                        buf_panel_keys = pl.tile.create(
                            [qk_panel_cols, IDX_HEAD_DIM], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat
                        )
                        for buf_key_page in pl.unroll(qk_panel_cols // BLOCK_SIZE):
                            buf_key_row = (
                                buf_logical_begin
                                + buf_score_begin
                                + buf_load_col // (score_tile // 2) * buf_lane_span
                                + buf_load_col % (score_tile // 2)
                                + buf_key_page * BLOCK_SIZE
                            )
                            if balance_leaves:
                                buf_key_row = (
                                    buf_logical_begin + buf_score_begin + buf_load_col + buf_key_page * BLOCK_SIZE
                                )
                            buf_safe_page = pl.min(
                                buf_key_row // BLOCK_SIZE, pl.max((buf_cache_len - 1) // BLOCK_SIZE, 0)
                            )
                            if buf_safe_page >= 0 and buf_safe_page < pl.tensor.dim(idx_block_table, 1):
                                buf_physical_page = pl.cast(
                                    pl.read(idx_block_table, [buf_batch_idx, buf_safe_page]), pl.INDEX
                                )
                                if buf_physical_page >= 0 and buf_physical_page < pl.tensor.dim(idx_native_kv_cache, 0):
                                    buf_native_byte = buf_physical_page * native_page_bytes
                                    if buf_native_byte % IDX_HEAD_DIM == 0:
                                        if key_prefetch_panels > 0:
                                            buf_key_l1_pool = pl.gather_row(
                                                buf_key_l1_pool,
                                                idx_kv_cache,
                                                [buf_load_slot + buf_key_page * BLOCK_SIZE, 0],
                                                [buf_native_byte // IDX_HEAD_DIM, 0],
                                                [BLOCK_SIZE, IDX_HEAD_DIM],
                                            )
                                        else:
                                            buf_panel_keys = pl.gather_row(
                                                buf_panel_keys,
                                                idx_kv_cache,
                                                [buf_key_page * BLOCK_SIZE, 0],
                                                [buf_native_byte // IDX_HEAD_DIM, 0],
                                                [BLOCK_SIZE, IDX_HEAD_DIM],
                                            )
                                    elif key_prefetch_panels > 0:
                                        buf_key_l1_pool = pl.gather_row(
                                            buf_key_l1_pool,
                                            idx_kv_cache_shift64,
                                            [buf_load_slot + buf_key_page * BLOCK_SIZE, 0],
                                            [buf_native_byte // IDX_HEAD_DIM, 0],
                                            [BLOCK_SIZE, IDX_HEAD_DIM],
                                        )
                                    else:
                                        buf_panel_keys = pl.gather_row(
                                            buf_panel_keys,
                                            idx_kv_cache_shift64,
                                            [buf_key_page * BLOCK_SIZE, 0],
                                            [buf_native_byte // IDX_HEAD_DIM, 0],
                                            [BLOCK_SIZE, IDX_HEAD_DIM],
                                        )
                        if key_prefetch_panels > 0:
                            if key_prefetch_to_l0:
                                buf_prefetched_key = pl.tile.extract(
                                    pl.tile.transpose_view(buf_key_l1_pool),
                                    0,
                                    buf_load_slot,
                                    [IDX_HEAD_DIM, qk_panel_cols],
                                    target_memory=pl.MemorySpace.Right,
                                )
                        else:
                            buf_prefetched_key = pl.tile.move(
                                pl.tile.transpose_view(buf_panel_keys), target_memory=pl.MemorySpace.Right
                            )
                    if key_prefetch_panels > 0:
                        buf_key_panel = buf_current_key
                    else:
                        buf_key_panel = buf_prefetched_key
                    buf_scores_l1 = pl.tile.create(
                        [query_group_size * IDX_N_HEADS, qk_panel_cols], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat
                    )
                    if buf_qk_panel > 0:
                        buf_previous_pair = pl.tile.extract(
                            buf_previous_l1,
                            0,
                            0,
                            [query_group_size * IDX_N_HEADS, qk_panel_cols],
                            target_memory=pl.MemorySpace.Right,
                        )
                        buf_scores = pl.tile.matmul(buf_query_left, buf_key_panel)
                        buf_reduced_pair = pl.tile.matmul(buf_coefficient_pair, buf_previous_pair)
                        buf_scores_l1 = pl.tile.assemble(
                            buf_scores_l1, buf_scores, [0, 0], pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True
                        )
                        pl.store(
                            pl.set_validshape(buf_reduced_pair, query_group_size, qk_panel_cols),
                            [buf_transfer_row, buf_panel_col - qk_panel_cols],
                            buf_score_transfer,
                        )
                    else:
                        buf_scores = pl.tile.matmul(buf_query_left, buf_key_panel)
                        buf_scores_l1 = pl.tile.assemble(
                            buf_scores_l1, buf_scores, [0, 0], pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True
                        )
                    buf_previous_l1 = buf_scores_l1
                    if key_prefetch_panels > 0:
                        if key_prefetch_to_l0:
                            if buf_qk_panel + 1 < score_tile // qk_panel_cols:
                                buf_current_key = buf_prefetched_key
                buf_last_pair = pl.tile.extract(
                    buf_previous_l1,
                    0,
                    0,
                    [query_group_size * IDX_N_HEADS, qk_panel_cols],
                    target_memory=pl.MemorySpace.Right,
                )
                buf_last_scores = pl.tile.matmul(buf_coefficient_pair, buf_last_pair)
                pl.store(
                    pl.set_validshape(buf_last_scores, query_group_size, qk_panel_cols),
                    [buf_transfer_row, score_tile - qk_panel_cols],
                    buf_score_transfer,
                )
                pl.system.sync_set(SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            # 即使只有1轮也排空已发消息；下个leaf才能重新从slot0开始，不携带旧ACK。
            for buf_score_drain in pl.range(pl.min(buf_score_iters, 2)):
                pl.system.sync_wait(SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
    return buf_score_transfer


@pl.jit.inline(auto_scope=False)
def indexer_score_vector_lane_stage(
    idx_native_kv_cache: pl.Tensor[[CACHE_PAGES_DYN, PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    position_ids: pl.Tensor[[TOKENS_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    buf_score_transfer: pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32],
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
    q_weight_scale: pl.Tensor[[Q_HEADS * Q_HEAD_DIM], pl.FP32],
    q_cos: pl.Tensor[[Q_T_DYN, Q_ROPE_DIM], pl.FP32],
    q_sin: pl.Tensor[[Q_T_DYN, Q_ROPE_DIM], pl.FP32],
    q_swap: pl.Tensor[[Q_T_DYN, Q_ROPE_DIM], pl.INT32],
    q_flat: pl.InOut[pl.Tensor[[Q_T_DYN, Q_HEADS * Q_HEAD_DIM], pl.BF16]],
    q_scale: pl.Tensor[[Q_PAD_ROWS_DYN, 1], pl.FP32],
    q_acc: pl.Tensor[[QPROJ_ROWS_DYN, Q_HEADS * Q_HEAD_DIM], pl.INT32],
    buf_score_lane: pl.Scalar[pl.INDEX],
    buf_worker: pl.Scalar[pl.INDEX],
    query_group_size: pl.constexpr,
    qk_panel_cols: pl.constexpr,
    score_tile: pl.constexpr,
    key_prefetch_panels: pl.constexpr,
    key_prefetch_to_l0: pl.constexpr,
    balance_leaves: pl.constexpr,
    SCORE_READY_EVENT: pl.constexpr,
    SCORE_CONSUMED_EVENT: pl.constexpr,
    INTERLEAVE_QUERY_DQ: pl.constexpr = False,
):
    """AIV lane0/1都完整消费同一core事件；只有真实query可写根，空尾lane仍ACK。"""
    # 原DQ owner=group*2+lane；进度跨所有Score工作携带，不按leaf重置。
    # 所需QB/QR/RoPE由外层在进入Score之前获取，持有FIFO槽时绝不增加前驱等待。
    q_dq_work = buf_worker * 2 + buf_score_lane
    q_dq_head_offset = 0
    groups = pl.cast(pl.read(plan, [0]), pl.INDEX)
    buf_max_leaves = pl.cast(pl.read(plan, [2]), pl.INDEX)
    buf_leaf_tiles = pl.cast(pl.read(plan, [3]), pl.INDEX)
    buf_extra_leaves = pl.cast(pl.read(plan, [4]), pl.INDEX)
    # 每个core独立读取同一device计划；空core不触发任何FFTS事件。
    for buf_item in pl.range(buf_worker, groups * buf_max_leaves, TOPK_SCORE_WORKERS):
        buf_group, buf_leaf = indexer_work_item(buf_item, groups, buf_max_leaves)
        buf_batch_idx, buf_query, buf_live, buf_visible_count = indexer_decode_group(
            bounds, kv_seq_lens, position_ids, offsets, buf_group, query_group_size
        )
        buf_logical_begin, buf_leaf_capacity, work_valid = indexer_leaf_range(
            buf_leaf, buf_leaf_tiles, buf_extra_leaves, buf_visible_count
        )
        if work_valid > 0:
            buf_cache_len = pl.cast(pl.read(kv_seq_lens, [buf_batch_idx]), pl.INDEX) // COMPRESS_RATIO
            if balance_leaves:
                single_query = buf_query
                single_leaf = buf_leaf
                single_batch = buf_batch_idx
                single_cache_len = buf_cache_len
                single_visible = buf_visible_count
                single_begin = buf_logical_begin
                single_capacity = buf_leaf_capacity
                single_valid = pl.min(single_capacity, single_visible - single_begin)
                single_steps = (single_valid + score_tile - 1) // score_tile
                single_roots = pl.tile.full([query_group_size // 2, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF)
                single_segments = pl.tile.full(
                    [query_group_size, BUFFERED_LONG_SCORE_TILE], dtype=pl.FP32, value=FP32_NEG_INF
                )
                # balanced每AIV持G/2条完整query；连续两轮凑2048候选，尾轮仍按实际valid排序。
                for single_step in pl.range(single_steps):
                    single_score_begin = single_begin + single_step * score_tile
                    single_transfer_row = (
                        buf_worker * 2 * query_group_size
                        + single_step % 2 * query_group_size
                        + buf_score_lane * (query_group_size // 2)
                    )
                    # A3没有INT8常量填充指令；INT16零按位展开为相同全零字节，不做数值转换。
                    single_scale_bytes = pl.tile.reinterpret_view(
                        pl.tile.full([1, score_tile], dtype=pl.INT16, value=0), pl.INT8
                    )
                    # 两AIV读取同一候选区间的scale；页表与slab容量检查先于gather。
                    for single_scale_page in pl.range(score_tile // BLOCK_SIZE):
                        single_scale_logical = pl.min(
                            single_score_begin // BLOCK_SIZE + single_scale_page,
                            pl.max((single_cache_len - 1) // BLOCK_SIZE, 0),
                        )
                        if single_scale_logical >= 0 and single_scale_logical < pl.tensor.dim(idx_block_table, 1):
                            single_scale_physical = pl.cast(
                                pl.read(idx_block_table, [single_batch, single_scale_logical]), pl.INDEX
                            )
                            if single_scale_physical >= 0 and single_scale_physical < pl.tensor.dim(
                                idx_native_kv_cache, 0
                            ):
                                single_scale_bytes = pl.gather_row(
                                    single_scale_bytes,
                                    idx_native_kv_cache,
                                    [0, single_scale_page * BLOCK_SIZE * 2],
                                    [single_scale_physical, INDEXER_KEY_BYTES],
                                    [1, BLOCK_SIZE * 2],
                                )
                    single_scale_half = pl.tile.reinterpret_view(single_scale_bytes, pl.FP16)
                    single_kv_scale = pl.cast(single_scale_half, pl.FP32)
                    pl.system.sync_wait(SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    single_scores = pl.load(
                        buf_score_transfer, [single_transfer_row, 0], [query_group_size // 2, score_tile]
                    )
                    pl.system.sync_set(
                        SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV
                    )
                    for single_local in pl.unroll(query_group_size // 2):
                        single_score_row = pl.tile.extract(
                            single_scores, single_local, 0, [1, score_tile], target_memory=pl.MemorySpace.Vec
                        )
                        single_scaled = pl.mul(single_score_row, single_kv_scale)
                        single_segments = pl.tile.assemble(
                            single_segments, single_scaled, [single_local * 2 + single_step % 2, 0]
                        )
                    # ACK已发布、本步scale/segment已落位，偶数步没有2048排序；
                    # 在下一次READY等待前推进本核一段DQ，让供数与独立Query数学重叠。
                    if INTERLEAVE_QUERY_DQ and single_step % 2 == 0:
                        q_flat, q_dq_work, q_dq_head_offset = q_dequant_rope_chunk_stage(
                            q_weight_scale, q_cos, q_sin, q_swap, q_flat, q_scale, q_acc,
                            q_dq_work, q_dq_head_offset, TOPK_SCORE_WORKERS * 2,
                        )
                    if single_step % 2 == 1 or single_step + 1 == single_steps:
                        single_segment_begin = single_begin + single_step // 2 * (2 * score_tile)
                        single_segment_span = (single_step % 2 + 1) * score_tile
                        single_segment_rows = pl.tile.reshape(
                            single_segments, [query_group_size // 2, 2 * BUFFERED_LONG_SCORE_TILE]
                        )
                        for single_local in pl.unroll(query_group_size // 2):
                            single_query_lane = buf_score_lane * (query_group_size // 2) + single_local
                            if single_query_lane < buf_live:
                                (single_query_visible, _causal) = indexer_query_visibility(
                                    bounds, kv_seq_lens, position_ids, single_batch, single_query + single_query_lane
                                )
                                single_segment_valid = pl.max(
                                    pl.min(single_query_visible - single_segment_begin, single_segment_span), 0
                                )
                                if single_segment_valid > 0:
                                    single_segment_data = pl.tile.extract(
                                        single_segment_rows,
                                        single_local,
                                        0,
                                        [1, 2 * BUFFERED_LONG_SCORE_TILE],
                                        target_memory=pl.MemorySpace.Vec,
                                    )
                                    logical_i32 = pl.cast(single_segment_begin, pl.INT32)
                                    if single_segment_valid <= 512:
                                        tiny_indices = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), logical_i32)
                                        tiny_tile = pl.tile.extract(
                                            single_segment_data, 0, 0, [1, 512], target_memory=pl.MemorySpace.Vec
                                        )
                                        tiny_raw = pl.set_validshape(tiny_tile, 1, single_segment_valid)
                                        tiny_scores = pl.maximum(
                                            pl.tile.fillpad(tiny_raw, pad_value=pl.PadValue.min), FP32_NEG_INF
                                        )
                                        tiny_pairs = pl.sort32(
                                            tiny_scores, pl.reinterpret_view(tiny_indices, pl.UINT32)
                                        )
                                        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=64)
                                        tiny_pairs = pl.mrgsort(tiny_pairs, block_len=256)
                                        # 提取512对(score,index)，限制循环root存储寿命；保留原sort32→merge输入顺序。
                                        single_new_root = pl.tile.extract(
                                            tiny_pairs, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                                        )
                                    elif single_segment_valid <= 1024:
                                        small_indices = pl.add(
                                            pl.tile.arange(0, [1, 1024], dtype=pl.INT32), logical_i32
                                        )
                                        small_tile = pl.tile.extract(
                                            single_segment_data, 0, 0, [1, 1024], target_memory=pl.MemorySpace.Vec
                                        )
                                        small_raw = pl.set_validshape(small_tile, 1, single_segment_valid)
                                        small_scores = pl.maximum(
                                            pl.tile.fillpad(small_raw, pad_value=pl.PadValue.min), FP32_NEG_INF
                                        )
                                        small_pairs = pl.sort32(
                                            small_scores, pl.reinterpret_view(small_indices, pl.UINT32)
                                        )
                                        small_pairs = pl.mrgsort(small_pairs, block_len=64)
                                        small_pairs = pl.mrgsort(small_pairs, block_len=256)
                                        small_left = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
                                        small_right = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 1024])
                                        small_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                                        small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_tmp)
                                        single_new_root = pl.tile.extract(
                                            small_merged, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                                        )
                                    else:
                                        large_indices = pl.add(
                                            pl.tile.arange(0, [1, 2048], dtype=pl.INT32), logical_i32
                                        )
                                        large_tile = pl.tile.extract(
                                            single_segment_data, 0, 0, [1, 2048], target_memory=pl.MemorySpace.Vec
                                        )
                                        large_raw = pl.set_validshape(large_tile, 1, single_segment_valid)
                                        large_scores = pl.maximum(
                                            pl.tile.fillpad(large_raw, pad_value=pl.PadValue.min), FP32_NEG_INF
                                        )
                                        large_pairs = pl.sort32(
                                            large_scores, pl.reinterpret_view(large_indices, pl.UINT32)
                                        )
                                        large_pairs = pl.mrgsort(large_pairs, block_len=64)
                                        large_pairs = pl.mrgsort(large_pairs, block_len=256)
                                        large_pairs = pl.mrgsort(large_pairs, block_len=1024)
                                        single_new_root = pl.tile.extract(
                                            large_pairs, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                                        )
                                    if single_step < 2:
                                        single_roots = pl.tile.assemble(
                                            single_roots, single_new_root, [single_local, 0]
                                        )
                                    else:
                                        single_old_root = pl.tile.extract(
                                            single_roots,
                                            single_local,
                                            0,
                                            [1, TOPK_PAIR_WIDTH],
                                            target_memory=pl.MemorySpace.Vec,
                                        )
                                        single_merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                                        # 新2048段在旧根之前，保留同分时后到段优先，不交换两输入。
                                        single_merged = pl.tile.mrgsort(
                                            single_new_root, single_old_root, tmp=single_merge_tmp
                                        )
                                        single_next_root = pl.tile.extract(
                                            single_merged, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                                        )
                                        single_roots = pl.tile.assemble(
                                            single_roots, single_next_root, [single_local, 0]
                                        )
                for single_local in pl.unroll(query_group_size // 2):
                    single_query_lane = buf_score_lane * (query_group_size // 2) + single_local
                    if single_query_lane < buf_live:
                        single_root = pl.tile.extract(
                            single_roots, single_local, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                        )
                        pl.store(
                            single_root,
                            [(single_query + single_query_lane) * TOPK_ROWS_PER_QUERY + single_leaf, 0],
                            pair_arena,
                        )
            else:
                buf_valid_count = pl.min(buf_leaf_capacity, buf_visible_count - buf_logical_begin)
                buf_tile_count = (buf_valid_count + score_tile - 1) // score_tile
                buf_lane_span = pl.min(buf_tile_count * (score_tile // 2), TOPK_CANDIDATES_PER_LEAF // 2)
                buf_score_iters = (buf_lane_span + score_tile // 2 - 1) // (score_tile // 2)
                buf_lane_begin = buf_score_lane * buf_lane_span
                # 双槽最多允许两份未消费分数；第2轮起等两个AIV的MTE2 ACK后才重用。
                for buf_score_step in pl.range(buf_score_iters):
                    buf_score_begin = buf_score_step * (score_tile // 2)
                    buf_transfer_row = buf_worker * 2 * query_group_size + buf_score_step % 2 * query_group_size
                    # 同上，INT16零视图保证未gather页字节已初始化，保持Native FP16重解释。
                    buf_scale_bytes = pl.tile.reinterpret_view(
                        pl.tile.full([1, score_tile // 2], dtype=pl.INT16, value=0), pl.INT8
                    )
                    # scale读取独立于Cube结果，放在wait之前；无效页不解引用，输入合同仍须保证历史页合法。
                    for buf_scale_page in pl.range(score_tile // 2 // BLOCK_SIZE):
                        buf_scale_row = (
                            buf_logical_begin + buf_score_begin + buf_lane_begin + buf_scale_page * BLOCK_SIZE
                        )
                        buf_scale_safe_page = pl.min(
                            buf_scale_row // BLOCK_SIZE, pl.max((buf_cache_len - 1) // BLOCK_SIZE, 0)
                        )
                        if buf_scale_safe_page >= 0 and buf_scale_safe_page < pl.tensor.dim(idx_block_table, 1):
                            buf_scale_physical_page = pl.cast(
                                pl.read(idx_block_table, [buf_batch_idx, buf_scale_safe_page]), pl.INDEX
                            )
                            if buf_scale_physical_page >= 0 and buf_scale_physical_page < pl.tensor.dim(
                                idx_native_kv_cache, 0
                            ):
                                buf_scale_bytes = pl.gather_row(
                                    buf_scale_bytes,
                                    idx_native_kv_cache,
                                    [0, buf_scale_page * BLOCK_SIZE * 2],
                                    [buf_scale_physical_page, INDEXER_KEY_BYTES],
                                    [1, BLOCK_SIZE * 2],
                                )
                    buf_scale_half = pl.tile.reinterpret_view(buf_scale_bytes, pl.FP16)
                    buf_kv_scale = pl.cast(buf_scale_half, target_type=pl.FP32)
                    pl.system.sync_wait(SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    buf_score_sums = pl.load(
                        buf_score_transfer,
                        [buf_transfer_row, buf_score_lane * (score_tile // 2)],
                        [query_group_size, score_tile // 2],
                    )
                    pl.system.sync_set(
                        SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV
                    )
                    # 每query单独causal/visible；不能把组内最大历史当成所有query的有效候选数。
                    for buf_query_lane in pl.unroll(query_group_size):
                        if buf_query_lane < buf_live:
                            (buf_query_visible, _causal) = indexer_query_visibility(
                                bounds, kv_seq_lens, position_ids, buf_batch_idx, buf_query + buf_query_lane
                            )
                            buf_lane_valid_rows = pl.max(
                                pl.min(
                                    buf_query_visible - buf_logical_begin - buf_score_begin - buf_lane_begin,
                                    score_tile // 2,
                                ),
                                0,
                            )
                            buf_score_sum = pl.tile.slice(buf_score_sums, [1, score_tile // 2], [buf_query_lane, 0])
                            buf_score_row = pl.mul(buf_score_sum, buf_kv_scale)
                            if buf_lane_valid_rows > 0:
                                pl.store(
                                    pl.set_validshape(buf_score_row, 1, buf_lane_valid_rows),
                                    [
                                        buf_worker * 2 * query_group_size + buf_query_lane * 2 + buf_score_lane,
                                        buf_score_begin,
                                    ],
                                    score_arena,
                                )
                    # 非balanced每步只写自己的半叶前缀；根排序在后面进行。
                    # 该步的FIFO数据已消费并ACK，可插入一段DQ，不改变score数学。
                    if INTERLEAVE_QUERY_DQ:
                        q_flat, q_dq_work, q_dq_head_offset = q_dequant_rope_chunk_stage(
                            q_weight_scale, q_cos, q_sin, q_swap, q_flat, q_scale, q_acc,
                            q_dq_work, q_dq_head_offset, TOPK_SCORE_WORKERS * 2,
                        )
                # 每query单独causal/visible；不能把组内最大历史当成所有query的有效候选数。
                for buf_query_lane in pl.unroll(query_group_size):
                    if buf_query_lane < buf_live:
                        (buf_query_visible, _causal) = indexer_query_visibility(
                            bounds, kv_seq_lens, position_ids, buf_batch_idx, buf_query + buf_query_lane
                        )
                        buf_half_begin = buf_logical_begin + buf_lane_begin
                        buf_half_valid = pl.max(pl.min(buf_query_visible - buf_half_begin, buf_lane_span), 0)
                        buf_half_slot = (
                            (buf_query + buf_query_lane) * TOPK_ROWS_PER_QUERY + buf_leaf * 2 + buf_score_lane
                        )
                        if buf_half_valid > 0:
                            indexer_topk_half_leaf(
                                score_arena,
                                pair_arena,
                                buf_worker * 2 * query_group_size + buf_query_lane * 2 + buf_score_lane,
                                buf_half_begin,
                                buf_half_valid,
                                buf_half_slot,
                            )
                        else:
                            pl.store(
                                pl.tile.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF),
                                [buf_half_slot, 0],
                                pair_arena,
                            )
    if INTERLEAVE_QUERY_DQ:
        # Score为零或可穿插窗口不足的owner仍必须完成自己的全部DQ。
        # Score环槽已按原协议归还；尾段不持有槽，也不等待SCORE_CUBE整段完成。
        q_tokens = pl.tensor.dim(q_flat, 0)
        q_head_tile = q_dequant_head_group_size(q_tokens)
        q_work_count = (q_tokens + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE * (Q_HEADS // q_head_tile)
        for q_tail_work in pl.range(q_dq_work, q_work_count, TOPK_SCORE_WORKERS * 2):
            q_tail_offset = 0
            if q_tail_work == q_dq_work:
                q_tail_offset = q_dq_head_offset
            q_tg = q_tail_work // (Q_HEADS // q_head_tile) * Q_ROPE_T_TILE
            q_hg = q_tail_work % (Q_HEADS // q_head_tile) * q_head_tile + q_tail_offset
            q_flat = q_dequant_rope_block_stage(
                q_weight_scale, q_cos, q_sin, q_swap, q_flat, q_scale, q_acc,
                q_tokens, q_tg, q_hg, q_head_tile - q_tail_offset,
            )
    return score_arena, pair_arena, q_flat
