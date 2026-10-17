# SPDX-License-Identifier: Apache-2.0
"""只编译动态Indexer score单个24组MIX，不启动NPU，不代表完整CSA或性能。

外部先准备统一Q/L计划、query、coeff与Native页表；测试壳保持所有容量动态。
Cube和两个AIV用同一FFTS缓冲和事件0/1并发；空核不触发事件，尾query不足
G时空AIV仍完整消费FIFO并ACK。FFTS不是phase完成信号，不能用串行task替代。
四种显式策略取自canonical调用分支，所有helper和数学来自正式源码。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_score_stages import (
    CACHE_PAGES_DYN,
    COEFF_COLS_DYN,
    COEFF_ROWS_DYN,
    IDX_HEAD_DIM,
    KEY0_ROWS_DYN,
    KEY64_ROWS_DYN,
    OFFSET_ROWS_DYN,
    PAGE_BYTES_DYN,
    PAIR_ROWS_DYN,
    PLAN_WORDS,
    QUERY_BOUNDS_DYN,
    QUERY_HEAD_ROWS_DYN,
    REQUESTS_DYN,
    SCORE_ROWS_DYN,
    TABLE_COLUMNS_DYN,
    TOKENS_DYN,
    TOPK_CANDIDATES_PER_LEAF,
    TOPK_PAIR_WIDTH,
    TRANSFER_COLS_DYN,
    TRANSFER_ROWS_DYN,
    indexer_score_cube_stage,
    indexer_score_vector_lane_stage,
)


@pl.jit.incore
def cube(
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
    query_group_size: pl.constexpr,
    qk_panel_cols: pl.constexpr,
    score_tile: pl.constexpr,
    key_prefetch_panels: pl.constexpr,
    key_prefetch_to_l0: pl.constexpr,
    balance_leaves: pl.constexpr,
    SCORE_READY_EVENT: pl.constexpr,
    SCORE_CONSUMED_EVENT: pl.constexpr,
    ffts: pl.InOut[pl.Tensor[[256], pl.INT64]],
):
    worker = pl.tile.get_block_idx()
    pl.system.set_ffts(ffts)
    original = buf_score_transfer
    buf_score_transfer = indexer_score_cube_stage(
        qr_hadamard_i8,
        coefficients,
        idx_kv_cache,
        idx_kv_cache_shift64,
        idx_native_kv_cache,
        idx_block_table,
        position_ids,
        kv_seq_lens,
        bounds,
        offsets,
        plan,
        buf_score_transfer,
        worker,
        query_group_size,
        qk_panel_cols,
        score_tile,
        key_prefetch_panels,
        key_prefetch_to_l0,
        balance_leaves,
        SCORE_READY_EVENT,
        SCORE_CONSUMED_EVENT,
    )
    return original, ffts


@pl.jit.incore
def vector(
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
    query_group_size: pl.constexpr,
    qk_panel_cols: pl.constexpr,
    score_tile: pl.constexpr,
    key_prefetch_panels: pl.constexpr,
    key_prefetch_to_l0: pl.constexpr,
    balance_leaves: pl.constexpr,
    SCORE_READY_EVENT: pl.constexpr,
    SCORE_CONSUMED_EVENT: pl.constexpr,
    ffts: pl.InOut[pl.Tensor[[256], pl.INT64]],
):
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx()
    lane = pl.tile.get_subblock_idx()
    pl.system.set_ffts(ffts)
    score_arena, pair_arena = indexer_score_vector_lane_stage(
        idx_native_kv_cache,
        idx_block_table,
        position_ids,
        kv_seq_lens,
        bounds,
        offsets,
        plan,
        buf_score_transfer,
        score_arena,
        pair_arena,
        lane,
        worker,
        query_group_size,
        qk_panel_cols,
        score_tile,
        key_prefetch_panels,
        key_prefetch_to_l0,
        balance_leaves,
        SCORE_READY_EVENT,
        SCORE_CONSUMED_EVENT,
    )
    return score_arena, pair_arena, ffts


@pl.jit.inline(auto_scope=False)
def group(
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
    ffts: pl.InOut[pl.Tensor[[256], pl.INT64]],
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
    query_group_size: pl.constexpr,
    qk_panel_cols: pl.constexpr,
    score_tile: pl.constexpr,
    key_prefetch_panels: pl.constexpr,
    key_prefetch_to_l0: pl.constexpr,
    balance_leaves: pl.constexpr,
    SCORE_READY_EVENT: pl.constexpr,
    SCORE_CONSUMED_EVENT: pl.constexpr,
):
    with pl.cluster(name_hint="dynamic_indexer_score"):
        buf_score_transfer, ffts = cube(
            qr_hadamard_i8,
            coefficients,
            idx_kv_cache,
            idx_kv_cache_shift64,
            idx_native_kv_cache,
            idx_block_table,
            position_ids,
            kv_seq_lens,
            bounds,
            offsets,
            plan,
            buf_score_transfer,
            query_group_size,
            qk_panel_cols,
            score_tile,
            key_prefetch_panels,
            key_prefetch_to_l0,
            balance_leaves,
            SCORE_READY_EVENT,
            SCORE_CONSUMED_EVENT,
            ffts,
        )
        score_arena, pair_arena, ffts = vector(
            idx_native_kv_cache,
            idx_block_table,
            position_ids,
            kv_seq_lens,
            bounds,
            offsets,
            plan,
            buf_score_transfer,
            score_arena,
            pair_arena,
            query_group_size,
            qk_panel_cols,
            score_tile,
            key_prefetch_panels,
            key_prefetch_to_l0,
            balance_leaves,
            SCORE_READY_EVENT,
            SCORE_CONSUMED_EVENT,
            ffts,
        )
    return buf_score_transfer, score_arena, pair_arena, ffts


@pl.jit
def g2_768_probe(
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
    ffts: pl.InOut[pl.Tensor[[256], pl.INT64]],
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
):
    with pl.spmd(24, sync_start=True):
        buf_score_transfer, score_arena, pair_arena, ffts = group(
            qr_hadamard_i8,
            coefficients,
            idx_kv_cache,
            idx_kv_cache_shift64,
            idx_native_kv_cache,
            idx_block_table,
            position_ids,
            kv_seq_lens,
            bounds,
            offsets,
            plan,
            buf_score_transfer,
            ffts,
            score_arena,
            pair_arena,
            2,
            128,
            768,
            0,
            True,
            False,
            0,
            1,
        )
    return buf_score_transfer, score_arena, pair_arena, ffts


@pl.jit
def g6_768_probe(
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
    ffts: pl.InOut[pl.Tensor[[256], pl.INT64]],
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
):
    with pl.spmd(24, sync_start=True):
        buf_score_transfer, score_arena, pair_arena, ffts = group(
            qr_hadamard_i8,
            coefficients,
            idx_kv_cache,
            idx_kv_cache_shift64,
            idx_native_kv_cache,
            idx_block_table,
            position_ids,
            kv_seq_lens,
            bounds,
            offsets,
            plan,
            buf_score_transfer,
            ffts,
            score_arena,
            pair_arena,
            6,
            64,
            768,
            0,
            True,
            False,
            0,
            1,
        )
    return buf_score_transfer, score_arena, pair_arena, ffts


@pl.jit
def g2_1024_probe(
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
    ffts: pl.InOut[pl.Tensor[[256], pl.INT64]],
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
):
    with pl.spmd(24, sync_start=True):
        buf_score_transfer, score_arena, pair_arena, ffts = group(
            qr_hadamard_i8,
            coefficients,
            idx_kv_cache,
            idx_kv_cache_shift64,
            idx_native_kv_cache,
            idx_block_table,
            position_ids,
            kv_seq_lens,
            bounds,
            offsets,
            plan,
            buf_score_transfer,
            ffts,
            score_arena,
            pair_arena,
            2,
            128,
            1024,
            1,
            True,
            False,
            0,
            1,
        )
    return buf_score_transfer, score_arena, pair_arena, ffts


@pl.jit
def g6_balanced_probe(
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
    ffts: pl.InOut[pl.Tensor[[256], pl.INT64]],
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
):
    with pl.spmd(24, sync_start=True):
        buf_score_transfer, score_arena, pair_arena, ffts = group(
            qr_hadamard_i8,
            coefficients,
            idx_kv_cache,
            idx_kv_cache_shift64,
            idx_native_kv_cache,
            idx_block_table,
            position_ids,
            kv_seq_lens,
            bounds,
            offsets,
            plan,
            buf_score_transfer,
            ffts,
            score_arena,
            pair_arena,
            6,
            64,
            1024,
            1,
            True,
            True,
            0,
            1,
        )
    return buf_score_transfer, score_arena, pair_arena, ffts
