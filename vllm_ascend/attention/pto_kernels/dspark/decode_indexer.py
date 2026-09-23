# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CSA integration dependency adapted from pypto-lib 205255b4/decode_indexer.py."""

import pypto.language as pl

from .config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    FP32_NEG_INF,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    TP,
)
from .config import (
    FLASH as M,
)
from .layout import INDEXER_KEY_BYTES, INDEXER_PAGE_BYTES_DYN, INDEXER_TABLE_COLUMNS_DYN

B_DYN = pl.dynamic("B_DYN")

T_DYN = pl.dynamic("T_DYN")  # T = B * S

B = DECODE_BATCH // TP

S = DECODE_SEQ

T = B * S

D = M.hidden_size

Q_LORA = M.q_lora_rank

ROPE_HEAD_DIM = M.qk_rope_head_dim

IDX_N_HEADS = M.index_n_heads

IDX_HEAD_DIM = M.index_head_dim

HADAMARD_SCALE = IDX_HEAD_DIM**-0.5

# A3 Native QLI FixpSToL1 uses DEQF16 with 0x3a800000 (1/1024).
NATIVE_QLI_QK_SCALE = 1.0 / 1024

NATIVE_QLI_WEIGHT_ROWS = 16

IDX_NOPE_HEAD_DIM = M.index_nope_head_dim

WEIGHTS_SCALE = M.index_weights_scale

MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 4  # the indexer only runs on ratio-4 layers

IDX_TOPK = M.index_topk

INNER_OVERLAP = COMPRESS_RATIO == 4

INNER_COFF = 1 + int(INNER_OVERLAP)

INNER_HEAD_DIM = IDX_HEAD_DIM

INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM

INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE

INNER_STATE_LEN = INNER_COFF * COMPRESS_RATIO

INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")

INNER_STATE_DIM = 2 * INNER_OUT_DIM

IDX_MAX_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO

IDX_MAX_BLOCKS = (IDX_MAX_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE

IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")

CACHE_TILE = min(64, BLOCK_SIZE)

Q_TILE = 256

Q_OUT_TILE = 1024  # Query-projection output tile

T_PAD = ((T + 16 - 1) // 16) * 16

MM_ROW_TILE = 16

MM_N_TILE = min(512, (128 * 1024) // (MM_ROW_TILE * 4))

DEQUANT_T_TILE = min(T, 8)

HEAD_DIM_TILE = 32

WEIGHTS_PROJECTION_N_TILE = 16
WEIGHTS_PROJECTION_K_TILE = 256

QH_QUANT_TILE = 64

QH_QUANT_WORKERS = 48  # Query Hadamard quantization workers

QR_PROJ_WORKERS = 24  # Query projection workers

QH_MM_TILE = 64  # Query Hadamard cube tile

QH_WORKERS = 24  # Query Hadamard matmul workers

WEIGHTS_WORKERS = 24  # Weights-projection workers

TP1_WEIGHTS_WORKERS = 8  # TP1 weights-projection workers

QH_HEAD_DIM_TILE = 64

DQ_ROPE_H_TILE = 4  # heads per fused query dequant + RoPE unit

DQ_ROPE_WORKERS = 48  # fused query dequant + RoPE workers

TOPK_PAIR_WIDTH = 2 * IDX_TOPK

TOPK_CANDIDATES_PER_LEAF = 8192

TOPK_MAX_CANDIDATES = IDX_MAX_ROWS

TOPK_MAX_LEAVES = (TOPK_MAX_CANDIDATES + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF

TOPK_ROWS_PER_QUERY = TOPK_MAX_LEAVES * 2

TOPK_QUERY_WORKERS = 48  # Top-K query-merge workers

TOPK_ARENA_ROWS = T_PAD * TOPK_ROWS_PER_QUERY

TOPK_SCORE_WORKERS = 24  # Top-K score workers

SCORE_TILE = 384

SCORE_LANE_ROWS = SCORE_TILE // 2

SCORE_ARENA_ROWS = max(T_PAD, TOPK_SCORE_WORKERS * 2)


@pl.jit.inline
def merge2_top512_pairs(
    pair_arena: pl.Tensor,
    left_slot: pl.Scalar[pl.INDEX],
    right_slot: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    """Merge two arena rows and store their exact Top-512 pair row."""
    left = pl.load(pair_arena, [left_slot, 0], [1, TOPK_PAIR_WIDTH])
    right = pl.load(pair_arena, [right_slot, 0], [1, TOPK_PAIR_WIDTH])
    merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
    # Native QLI MergeSort places the incoming chunk before the accumulated
    # list, which determines the ordering when scores are exactly equal.
    merged_all = pl.tile.mrgsort(right, left, tmp=merge_tmp)
    merged = pl.tile.slice(merged_all, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output_slot, 0], pair_arena)


@pl.jit.inline
def indexer_topk_half_leaf(
    score_arena: pl.Tensor[[SCORE_ARENA_ROWS, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    score_row: pl.Scalar[pl.INDEX],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    """Sort a contiguous half-leaf and store its Top-512 pairs."""
    logical_begin_i32 = pl.cast(logical_begin, pl.INT32)
    if valid_count <= 512:
        short_indices = pl.add(pl.tile.arange(0, [1, 512], dtype=pl.INT32), logical_begin_i32)
        short_raw = pl.load(score_arena, [score_row, 0], [1, 512], valid_shape=[1, valid_count])
        short_scores = pl.tile.fillpad(short_raw, pad_value=pl.PadValue.min)
        short_scores = pl.maximum(short_scores, FP32_NEG_INF)
        short_pairs = pl.sort32(short_scores, pl.reinterpret_view(short_indices, pl.UINT32))
        short_pairs = pl.mrgsort(short_pairs, block_len=64)
        short_pairs = pl.mrgsort(short_pairs, block_len=256)
        pl.store(short_pairs, [output_slot, 0], pair_arena)
    elif valid_count <= 1024:
        small_indices = pl.add(pl.tile.arange(0, [1, 1024], dtype=pl.INT32), logical_begin_i32)
        small_raw = pl.load(score_arena, [score_row, 0], [1, 1024], valid_shape=[1, valid_count])
        small_scores = pl.tile.fillpad(small_raw, pad_value=pl.PadValue.min)
        small_scores = pl.maximum(small_scores, FP32_NEG_INF)
        small_pairs = pl.sort32(small_scores, pl.reinterpret_view(small_indices, pl.UINT32))
        small_pairs = pl.mrgsort(small_pairs, block_len=64)
        small_pairs = pl.mrgsort(small_pairs, block_len=256)
        small_left = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        small_right = pl.tile.slice(small_pairs, [1, TOPK_PAIR_WIDTH], [0, 1024])
        small_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        small_merged = pl.tile.mrgsort(small_left, small_right, tmp=small_tmp)
        small_top = pl.tile.slice(small_merged, [1, TOPK_PAIR_WIDTH], [0, 0])
        pl.store(small_top, [output_slot, 0], pair_arena)
    elif valid_count <= 2048:
        leaf_indices = pl.add(pl.tile.arange(0, [1, 2048], dtype=pl.INT32), logical_begin_i32)
        leaf_scores_raw = pl.load(score_arena, [score_row, 0], [1, 2048], valid_shape=[1, valid_count])
        leaf_scores = pl.tile.fillpad(leaf_scores_raw, pad_value=pl.PadValue.min)
        leaf_scores = pl.maximum(leaf_scores, FP32_NEG_INF)
        pairs = pl.sort32(leaf_scores, pl.reinterpret_view(leaf_indices, pl.UINT32))
        pairs = pl.mrgsort(pairs, block_len=64)
        pairs = pl.mrgsort(pairs, block_len=256)
        pairs = pl.mrgsort(pairs, block_len=1024)
        pl.store(pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)
    elif valid_count <= 4096:
        medium_indices = pl.add(pl.tile.arange(0, [1, 4096], dtype=pl.INT32), logical_begin_i32)
        medium_scores_raw = pl.load(score_arena, [score_row, 0], [1, 4096], valid_shape=[1, valid_count])
        medium_scores = pl.tile.fillpad(medium_scores_raw, pad_value=pl.PadValue.min)
        medium_scores = pl.maximum(medium_scores, FP32_NEG_INF)
        medium_pairs = pl.sort32(medium_scores, pl.reinterpret_view(medium_indices, pl.UINT32))
        medium_pairs = pl.mrgsort(medium_pairs, block_len=64)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=256)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=1024)
        medium_left = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        medium_right = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        medium_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        # Native sorts 2048 candidates at a time, then merges the new chunk first.
        medium_merged = pl.tile.mrgsort(medium_right, medium_left, tmp=medium_tmp)
        pl.store(pl.tile.slice(medium_merged, [1, TOPK_PAIR_WIDTH], [0, 0]), [output_slot, 0], pair_arena)


@pl.jit.inline
def indexer_topk_query_merge_one(
    query: pl.Scalar[pl.INDEX],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
):
    """Merge half-leaf roots and materialize one query's Top-512."""
    batch_idx = query // S
    position = pl.read(position_ids, [query])
    cache_len = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
    cache_bound = pl.min(cache_len, (position + 1) // COMPRESS_RATIO)
    visible_count = pl.min(cache_bound, TOPK_MAX_CANDIDATES)
    if visible_count > 0:
        leaf_count = (visible_count + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF
        half_count = leaf_count * 2
        arena_base = query * TOPK_ROWS_PER_QUERY
        for child in pl.range(1, half_count):
            merge2_top512_pairs(pair_arena, arena_base, arena_base + child, arena_base)

        root_slot = arena_base
        root_pairs = pl.load(pair_arena, [root_slot, 0], [1, TOPK_PAIR_WIDTH])
        root_scores = pl.tile.gather_mask(root_pairs, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        pl.store(root_scores, [query, 0], topk_scores)
        root_indices = pl.tile.gather_mask(root_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        if visible_count >= IDX_TOPK:
            pl.store(root_indices, [query, 0], topk_indices)
        else:
            output_indices = pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(visible_count):
                pl.tile.write(output_indices, [0, lane], pl.tile.read(root_indices, [0, lane]))
            pl.store(output_indices, [query, 0], topk_indices)
    else:
        pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.FP32, value=FP32_NEG_INF), [query, 0], topk_scores)
        pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1), [query, 0], topk_indices)


@pl.jit.incore
def indexer_topk_query_merge(
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
):
    """Merge query roots on one persistent worker per physical AIV."""
    worker = pl.tile.get_block_idx()
    query_count = pl.tensor.dim(position_ids, 0)
    for query in pl.range(worker, query_count, TOPK_QUERY_WORKERS):
        indexer_topk_query_merge_one(
            query,
            position_ids,
            kv_seq_lens,
            pair_arena,
            topk_scores,
            topk_indices,
        )


@pl.jit.inline
def indexer_topk_leaf_publish(
    score_arena: pl.Tensor[[SCORE_ARENA_ROWS, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    query: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
) -> None:
    """Sort one populated leaf and directly publish its Top-512 pairs."""
    if valid_count <= 2048:
        short_indices = pl.tile.arange(0, [1, 2048], dtype=pl.INT32)
        short_raw = pl.load(score_arena, [query, 0], [1, 2048], valid_shape=[1, valid_count])
        short_scores = pl.tile.fillpad(short_raw, pad_value=pl.PadValue.min)
        short_scores = pl.maximum(short_scores, FP32_NEG_INF)
        short_pairs = pl.sort32(short_scores, pl.reinterpret_view(short_indices, pl.UINT32))
        short_pairs = pl.mrgsort(short_pairs, block_len=64)
        short_pairs = pl.mrgsort(short_pairs, block_len=256)
        short_pairs = pl.mrgsort(short_pairs, block_len=1024)
        short_top = pl.tile.slice(short_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        short_values = pl.tile.gather_mask(short_top, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        short_selected = pl.tile.gather_mask(short_top, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        pl.store(short_values, [query, 0], topk_scores)
        if valid_count >= IDX_TOPK:
            pl.store(short_selected, [query, 0], topk_indices)
        else:
            short_output = pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(valid_count):
                pl.tile.write(short_output, [0, lane], pl.tile.read(short_selected, [0, lane]))
            pl.store(short_output, [query, 0], topk_indices)
    elif valid_count <= 4096:
        medium_indices = pl.tile.arange(0, [1, 4096], dtype=pl.INT32)
        medium_raw = pl.load(score_arena, [query, 0], [1, 4096], valid_shape=[1, valid_count])
        medium_scores = pl.tile.fillpad(medium_raw, pad_value=pl.PadValue.min)
        medium_scores = pl.maximum(medium_scores, FP32_NEG_INF)
        medium_pairs = pl.sort32(medium_scores, pl.reinterpret_view(medium_indices, pl.UINT32))
        medium_pairs = pl.mrgsort(medium_pairs, block_len=64)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=256)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=1024)
        medium_left = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        medium_right = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        medium_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        medium_merged = pl.tile.mrgsort(medium_right, medium_left, tmp=medium_tmp)
        medium_top = pl.tile.slice(medium_merged, [1, TOPK_PAIR_WIDTH], [0, 0])
        medium_values = pl.tile.gather_mask(medium_top, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        medium_selected = pl.tile.gather_mask(medium_top, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        pl.store(medium_values, [query, 0], topk_scores)
        pl.store(medium_selected, [query, 0], topk_indices)
    else:
        full_indices = pl.tile.arange(0, [1, TOPK_CANDIDATES_PER_LEAF], dtype=pl.INT32)
        full_raw = pl.load(score_arena, [query, 0], [1, TOPK_CANDIDATES_PER_LEAF], valid_shape=[1, valid_count])
        full_scores = pl.tile.fillpad(full_raw, pad_value=pl.PadValue.min)
        full_min = pl.tile.full([1, TOPK_CANDIDATES_PER_LEAF], dtype=pl.FP32, value=FP32_NEG_INF)
        full_scores = pl.maximum(full_scores, full_min)
        full_pairs = pl.sort32(full_scores, pl.reinterpret_view(full_indices, pl.UINT32))
        full_pairs = pl.mrgsort(full_pairs, block_len=64)
        full_pairs = pl.mrgsort(full_pairs, block_len=256)
        full_pairs = pl.mrgsort(full_pairs, block_len=1024)
        # Preserve the same later-chunk tie priority in the single-leaf path.
        full_first = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        full_second = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        full_third = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 8192])
        full_fourth = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 12288])
        full_tmp = pl.tile.create([1, 4 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        full_merged = pl.tile.mrgsort(full_fourth, full_third, full_second, full_first, tmp=full_tmp)
        full_top = pl.tile.slice(full_merged, [1, TOPK_PAIR_WIDTH], [0, 0])
        full_values = pl.tile.gather_mask(full_top, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        full_selected = pl.tile.gather_mask(full_top, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        pl.store(full_values, [query, 0], topk_scores)
        pl.store(full_selected, [query, 0], topk_indices)


@pl.jit.incore
def indexer_topk_single_leaf_publish(
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    score_arena: pl.Tensor[[SCORE_ARENA_ROWS, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
):
    """Sort a single leaf and publish each query's Top-512 directly."""
    worker = pl.tile.get_block_idx()
    query_count = pl.tensor.dim(position_ids, 0)
    for query in pl.range(worker, query_count, TOPK_QUERY_WORKERS):
        position = pl.read(position_ids, [query])
        cache_len = pl.read(kv_seq_lens, [query // S]) // COMPRESS_RATIO
        visible_count = pl.max(pl.min(cache_len, (position + 1) // COMPRESS_RATIO), 0)
        if visible_count > 0:
            indexer_topk_leaf_publish(score_arena, query, visible_count, topk_scores, topk_indices)
        else:
            pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.FP32, value=FP32_NEG_INF), [query, 0], topk_scores)
            pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1), [query, 0], topk_indices)


@pl.jit.inline(auto_scope=False)
def indexer_head_coefficients(
    qr_hadamard_scale_dq: pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32],
    weights: pl.Tensor[[T_PAD, IDX_N_HEADS], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    qh_quant_tid: pl.Scalar[pl.TASK_ID],
    weights_tid: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor[[T_PAD * NATIVE_QLI_WEIGHT_ROWS, IDX_N_HEADS], pl.FP16], pl.Scalar[pl.TASK_ID]]:
    """Shared production FP16 coefficients used by the QLI Cube reduction."""
    # Keep coefficients in owned GM/L1 storage: a cross-core pipe slot cannot
    # remain live while the same slot transports each candidate score tile.
    coefficients = pl.create_tensor([T_PAD * NATIVE_QLI_WEIGHT_ROWS, IDX_N_HEADS], dtype=pl.FP16)
    with pl.spmd(
        TOPK_QUERY_WORKERS,
        name_hint="indexer_head_coefficients",
        deps=[qh_quant_tid, weights_tid],
        allow_early_resolve=True,
    ) as coefficients_tid:
        coefficient_worker = pl.tile.get_block_idx()
        coefficient_count = pl.tensor.dim(position_ids, 0)
        for coefficient_query in pl.range(coefficient_worker, coefficient_count, TOPK_QUERY_WORKERS):
            coefficient_head_begin = coefficient_query * IDX_N_HEADS
            query_scale = pl.reshape(
                qr_hadamard_scale_dq[coefficient_head_begin : coefficient_head_begin + IDX_N_HEADS, 0:1],
                [1, IDX_N_HEADS],
            )
            query_weight = weights[coefficient_query : coefficient_query + 1, 0:IDX_N_HEADS]
            # Native QLI ProcessVec0 materializes this product in FP16.
            head_coefficient = pl.cast(pl.mul(query_scale, query_weight), pl.FP16, mode="rint")
            coefficient_rows = pl.col_expand_mul(
                pl.full([NATIVE_QLI_WEIGHT_ROWS, IDX_N_HEADS], dtype=pl.FP32, value=1.0),
                pl.cast(head_coefficient, pl.FP32),
            )
            coefficient_row = coefficient_query * NATIVE_QLI_WEIGHT_ROWS
            coefficients[coefficient_row : coefficient_row + NATIVE_QLI_WEIGHT_ROWS, :] = pl.cast(
                coefficient_rows, pl.FP16
            )
    return coefficients, coefficients_tid


@pl.jit.inline(auto_scope=False)
def indexer_score_topk_forest(
    qr_hadamard_i8: pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8],
    qr_hadamard_scale_dq: pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32],
    weights: pl.Tensor[[T_PAD, IDX_N_HEADS], pl.FP32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    qh_quant_tid: pl.Scalar[pl.TASK_ID],
    weights_tid: pl.Scalar[pl.TASK_ID],
    cache_write_tid: pl.Scalar[pl.TASK_ID],
):
    """Score and select half-leaves, then merge their exact Top-K rows."""
    b_dim = pl.tensor.dim(idx_block_table, 0)
    table_columns = pl.tensor.dim(idx_block_table, 1)
    idx_table_len = b_dim * table_columns
    # Native packs each page's INT8 keys and FP16 scales into one allocation.
    idx_block_table_flat = pl.reshape(
        idx_block_table,
        [idx_table_len],
    )
    pair_arena = pl.create_tensor([TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], dtype=pl.FP32)
    # The whole batch uses query rows for one leaf, or private lane rows for multiple leaves.
    score_arena = pl.create_tensor([SCORE_ARENA_ROWS, TOPK_CANDIDATES_PER_LEAF], dtype=pl.FP32)
    coefficients, coefficients_tid = indexer_head_coefficients(
        qr_hadamard_scale_dq, weights, position_ids, qh_quant_tid, weights_tid
    )
    with pl.spmd(
        TOPK_SCORE_WORKERS,
        name_hint="indexer_score_topk_leaf",
        deps=[coefficients_tid, cache_write_tid],
        allow_early_resolve=True,
        optimizations=[pl.cross_core_slot(slot_num=1)],
    ) as score_tid:
        worker = pl.tile.get_block_idx()
        query_count = pl.tensor.dim(position_ids, 0)
        max_cache_len = 0
        for batch in pl.range(query_count // S):
            batch_cache_len = pl.read(kv_seq_lens, [batch]) // COMPRESS_RATIO
            max_cache_len = pl.max(max_cache_len, batch_cache_len)
        max_leaves = pl.max(
            (pl.min(max_cache_len, TOPK_MAX_CANDIDATES) + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF, 1
        )
        single_leaf = pl.cast(max_leaves == 1, pl.INDEX)
        for item in pl.range(worker, query_count * max_leaves, TOPK_SCORE_WORKERS):
            query = item // max_leaves
            leaf = item % max_leaves
            batch_idx = query // S
            position = pl.read(position_ids, [query])
            cache_len = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
            cache_bound = pl.min(cache_len, (position + 1) // COMPRESS_RATIO)
            visible_count = pl.max(pl.min(cache_bound, TOPK_MAX_CANDIDATES), 0)
            logical_begin = leaf * TOPK_CANDIDATES_PER_LEAF
            if logical_begin < visible_count:
                valid_count = pl.min(TOPK_CANDIDATES_PER_LEAF, visible_count - logical_begin)
                # Each half-leaf must fit the 4096-candidate sort path.
                lane_span = pl.min(
                    ((valid_count + SCORE_TILE - 1) // SCORE_TILE) * SCORE_LANE_ROWS,
                    TOPK_CANDIDATES_PER_LEAF // 2,
                )
                lane_stride = single_leaf * SCORE_LANE_ROWS + (1 - single_leaf) * lane_span
                query_head_begin = query * IDX_N_HEADS
                query_vector = qr_hadamard_i8[query_head_begin : query_head_begin + IDX_N_HEADS, 0:IDX_HEAD_DIM]
                coefficient_begin = query * NATIVE_QLI_WEIGHT_ROWS
                coefficients_l1 = coefficients[
                    coefficient_begin : coefficient_begin + NATIVE_QLI_WEIGHT_ROWS, 0:IDX_N_HEADS
                ]
                for score_begin in pl.pipeline(0, lane_span, SCORE_LANE_ROWS, stage=2):
                    read_begin = score_begin * (1 + single_leaf)
                    # Native packs 4096 contiguous key bytes before the FP16
                    # scales in each page. Load that region once per page in
                    # each AIV lane, then hand the assembled rows to Cube in
                    # L1. This avoids 32 small GM DMAs per page and needs no
                    # GM staging allocation or additional root arguments.
                    for key_aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        key_bytes = pl.create_tensor([1, SCORE_LANE_ROWS * IDX_HEAD_DIM], dtype=pl.INT8)
                        for page in pl.unroll(SCORE_LANE_ROWS // BLOCK_SIZE):
                            page_begin = page * BLOCK_SIZE
                            lane_page = key_aiv * lane_stride + page_begin
                            safe_page_begin = pl.min(
                                read_begin + lane_page, ((valid_count - 1) // BLOCK_SIZE) * BLOCK_SIZE
                            )
                            logical_page = (logical_begin + safe_page_begin) // BLOCK_SIZE
                            physical_block = pl.cast(
                                pl.read(idx_block_table_flat, [batch_idx * table_columns + logical_page]), pl.INDEX
                            )
                            key_bytes = pl.gather_row(
                                key_bytes,
                                idx_kv_cache,
                                [0, page_begin * IDX_HEAD_DIM],
                                [physical_block, 0],
                                [1, INDEXER_KEY_BYTES],
                            )
                        key_rows = pl.reshape(key_bytes, [SCORE_LANE_ROWS, IDX_HEAD_DIM])
                        kv_i8 = pl.aic_gather(key_rows)
                    score_i32 = pl.matmul(query_vector, kv_i8, out_dtype=pl.INT32, b_trans=True)
                    # Match Native's FP16 QK tile and FP32 Cube reduction.
                    for score_aiv in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                        score_shard = pl.aiv_shard(score_i32)
                        score_fp32 = pl.maximum(pl.cast(score_shard, pl.FP32), 0.0)
                        score_half = pl.cast(pl.mul(score_fp32, NATIVE_QLI_QK_SCALE), pl.FP16, mode="rint")
                        scores_l1 = pl.aic_gather(score_half)
                    weighted_scores = pl.matmul(coefficients_l1, scores_l1, out_dtype=pl.FP32)
                    # Each lane owns a contiguous candidate-column range.
                    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                        lane_begin = aiv_id * lane_stride
                        lane_valid_rows = pl.max(pl.min(valid_count - read_begin - lane_begin, SCORE_LANE_ROWS), 0)
                        kv_scale_bytes = pl.create_tensor([1, SCORE_LANE_ROWS * 2], dtype=pl.INT8)
                        for scale_page in pl.unroll(SCORE_TILE // (2 * BLOCK_SIZE)):
                            scale_page_begin = scale_page * BLOCK_SIZE
                            safe_scale_begin = pl.min(
                                read_begin + lane_begin + scale_page_begin,
                                ((valid_count - 1) // BLOCK_SIZE) * BLOCK_SIZE,
                            )
                            scale_logical_page = (logical_begin + safe_scale_begin) // BLOCK_SIZE
                            scale_block = pl.cast(
                                pl.read(idx_block_table_flat, [batch_idx * table_columns + scale_logical_page]),
                                pl.INDEX,
                            )
                            kv_scale_bytes = pl.gather_row(
                                kv_scale_bytes,
                                idx_kv_cache,
                                [0, scale_page_begin * 2],
                                [scale_block, INDEXER_KEY_BYTES],
                                [1, BLOCK_SIZE * 2],
                            )
                        kv_scale = pl.reinterpret_view(kv_scale_bytes, pl.FP16)
                        weighted_shard = pl.aiv_shard(weighted_scores)
                        score_row = weighted_shard[0:1, :]
                        score_row = pl.mul(score_row, pl.cast(kv_scale, pl.FP32))
                        score_row_id = single_leaf * query + (1 - single_leaf) * (worker * 2 + aiv_id)
                        score_col = single_leaf * (read_begin + lane_begin) + (1 - single_leaf) * score_begin
                        # Store only valid scores; the Top-K load pads its own tail.
                        if lane_valid_rows > 0:
                            score_valid = pl.set_validshape(score_row, 1, lane_valid_rows)
                            score_arena[score_row_id : score_row_id + 1, score_col : score_col + SCORE_LANE_ROWS] = (
                                score_valid
                            )

                if single_leaf == 0:
                    for sort_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                        half_begin = logical_begin + sort_lane * lane_span
                        half_valid = pl.max(pl.min(valid_count - sort_lane * lane_span, lane_span), 0)
                        half_slot = query * TOPK_ROWS_PER_QUERY + leaf * 2 + sort_lane
                        if half_valid > 0:
                            indexer_topk_half_leaf(
                                score_arena, pair_arena, worker * 2 + sort_lane, half_begin, half_valid, half_slot
                            )
                        else:
                            empty_pairs = pl.tile.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF)
                            pl.store(empty_pairs, [half_slot, 0], pair_arena)

    max_topk_cache_len = 0
    for topk_batch in pl.range(b_dim):
        topk_cache_len = pl.read(kv_seq_lens, [topk_batch]) // COMPRESS_RATIO
        max_topk_cache_len = pl.max(max_topk_cache_len, topk_cache_len)
    with pl.scope():
        if max_topk_cache_len <= TOPK_CANDIDATES_PER_LEAF:
            with pl.spmd(
                TOPK_QUERY_WORKERS,
                name_hint="indexer_topk_single_leaf_publish",
                deps=[score_tid],
                allow_early_resolve=True,
            ):
                indexer_topk_single_leaf_publish(position_ids, kv_seq_lens, score_arena, topk_scores, topk_idxs)
        else:
            with pl.spmd(
                TOPK_QUERY_WORKERS, name_hint="indexer_topk_query_merge", deps=[score_tid], allow_early_resolve=True
            ):
                indexer_topk_query_merge(position_ids, kv_seq_lens, pair_arena, topk_scores, topk_idxs)

    return topk_scores, topk_idxs, score_tid


@pl.jit.inline(auto_scope=False)
def indexer_qr_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    qr_bf16: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.BF16]],
) -> pl.Scalar[pl.TASK_ID]:
    """Indexer query projection, dequant and RoPE -- everything before the hadamard."""

    bs = pl.tensor.dim(x, 0)
    row_blocks = (bs + MM_ROW_TILE - 1) // MM_ROW_TILE
    qr_acc_pad = pl.create_tensor([T_PAD, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.INT32)
    with pl.spmd(
        QR_PROJ_WORKERS,
        name_hint="idx_qr_proj_matmul",
        allow_early_resolve=True,
    ) as idx_qr_mm_tid:
        qr_proj_worker = pl.tile.get_block_idx()
        for qr_unit in pl.range(qr_proj_worker, IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE * row_blocks, QR_PROJ_WORKERS):
            qr_rb = qr_unit // (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE)
            output_tile = qr_unit - qr_rb * (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE)
            qr_r0 = qr_rb * MM_ROW_TILE
            qr_rows = pl.min(MM_ROW_TILE, bs - qr_r0)
            o_base = output_tile * Q_OUT_TILE
            for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
                qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
                for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                    q0 = kb * Q_TILE
                    qr_tile = pl.slice(qr, [MM_ROW_TILE, Q_TILE], [qr_r0, q0], valid_shape=[qr_rows, Q_TILE])
                    wq_tile = wq_b[q0 : q0 + Q_TILE, o_base + ns : o_base + ns + MM_N_TILE]
                    qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile, init_cond=(q0 == 0))
                qr_acc_pad[qr_r0 : qr_r0 + MM_ROW_TILE, o_base + ns : o_base + ns + MM_N_TILE] = qr_acc
    # Fused dequant + RoPE: one unit is DEQUANT_T_TILE tokens x DQ_ROPE_H_TILE heads.
    qr_bf16_2d = pl.reshape(qr_bf16, [T_PAD, IDX_N_HEADS * IDX_HEAD_DIM])
    dq_rope_units = ((bs + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE) * (IDX_N_HEADS // DQ_ROPE_H_TILE)
    dq_rope_workers = pl.min(dq_rope_units, DQ_ROPE_WORKERS)
    for dq_rope_worker in pl.spmd(dq_rope_workers, name_hint="idx_qr_dequant_rope", allow_early_resolve=True):
        sw_ones = pl.full([DEQUANT_T_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        sw_index = pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32)
        sw_col = pl.col_expand_mul(sw_ones, sw_index)
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        rope_swap_idx = pl.cast(pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)), target_type=pl.INT32)
        for dq_unit in pl.range(dq_rope_worker, dq_rope_units, dq_rope_workers):
            hg = (dq_unit % (IDX_N_HEADS // DQ_ROPE_H_TILE)) * DQ_ROPE_H_TILE
            dq_t0 = (dq_unit // (IDX_N_HEADS // DQ_ROPE_H_TILE)) * DEQUANT_T_TILE
            if dq_t0 + DEQUANT_T_TILE <= bs:
                qr_scale_tile = qr_scale[dq_t0 : dq_t0 + DEQUANT_T_TILE, :]
                cos_tile = cos[dq_t0 : dq_t0 + DEQUANT_T_TILE, 0:ROPE_HEAD_DIM]
                sin_tile = sin[dq_t0 : dq_t0 + DEQUANT_T_TILE, 0:ROPE_HEAD_DIM]
                for h_inner in pl.pipeline(DQ_ROPE_H_TILE, stage=2):
                    h0 = (hg + h_inner) * IDX_HEAD_DIM
                    wq_scale = pl.reshape(wq_b_scale[h0 : h0 + IDX_HEAD_DIM], [1, IDX_HEAD_DIM])
                    acc_fp32 = pl.cast(
                        qr_acc_pad[dq_t0 : dq_t0 + DEQUANT_T_TILE, h0 : h0 + IDX_HEAD_DIM],
                        target_type=pl.FP32,
                        mode="none",
                    )
                    # Native combines the activation and weight scales before
                    # multiplying the INT32 accumulator converted to FP32.
                    qr_dequant_scale = pl.col_expand_mul(
                        pl.row_expand_mul(
                            pl.full([DEQUANT_T_TILE, IDX_HEAD_DIM], dtype=pl.FP32, value=1.0), qr_scale_tile
                        ),
                        wq_scale,
                    )
                    qr_dequant = pl.mul(acc_fp32, qr_dequant_scale)
                    # Native quantized projection rounds to BF16 before RoPE.
                    qr_dequant = pl.cast(pl.cast(qr_dequant, pl.BF16, mode="rint"), pl.FP32)
                    qr_nope_bf16 = pl.cast(qr_dequant[:, 0:IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode="rint")
                    qr_rope_slice = qr_dequant[:, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
                    qr_swapped = pl.gather(qr_rope_slice, dim=-1, index=rope_swap_idx)
                    rope_rot = pl.add(pl.mul(qr_rope_slice, cos_tile), pl.mul(qr_swapped, sin_tile))
                    rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
                    qr_bf16_2d[dq_t0 : dq_t0 + DEQUANT_T_TILE, h0 : h0 + IDX_NOPE_HEAD_DIM] = qr_nope_bf16
                    qr_bf16_2d[dq_t0 : dq_t0 + DEQUANT_T_TILE, h0 + IDX_NOPE_HEAD_DIM : h0 + IDX_HEAD_DIM] = rope_bf16
            else:
                # At most seven rows. Keep all broadcast operands at the same
                # extent; a partial scale tile cannot broadcast into eight rows.
                tail_swap_idx = rope_swap_idx[0:1, :]
                for tail_t0 in pl.range(dq_t0, bs):
                    tail_qr_scale_value = pl.read(qr_scale, [tail_t0, 0])
                    tail_cos_tile = cos[tail_t0 : tail_t0 + 1, 0:ROPE_HEAD_DIM]
                    tail_sin_tile = sin[tail_t0 : tail_t0 + 1, 0:ROPE_HEAD_DIM]
                    for tail_h_inner in pl.pipeline(DQ_ROPE_H_TILE, stage=2):
                        tail_h0 = (hg + tail_h_inner) * IDX_HEAD_DIM
                        tail_wq_scale = pl.reshape(wq_b_scale[tail_h0 : tail_h0 + IDX_HEAD_DIM], [1, IDX_HEAD_DIM])
                        tail_acc_fp32 = pl.cast(
                            qr_acc_pad[tail_t0 : tail_t0 + 1, tail_h0 : tail_h0 + IDX_HEAD_DIM],
                            target_type=pl.FP32,
                            mode="none",
                        )
                        # Native combines the activation and weight scales before
                        # multiplying the INT32 accumulator converted to FP32.
                        tail_qr_dequant_scale = pl.mul(tail_wq_scale, tail_qr_scale_value)
                        tail_qr_dequant = pl.mul(tail_acc_fp32, tail_qr_dequant_scale)
                        # Native quantized projection rounds to BF16 before RoPE.
                        tail_qr_dequant = pl.cast(pl.cast(tail_qr_dequant, pl.BF16, mode="rint"), pl.FP32)
                        tail_qr_nope_bf16 = pl.cast(
                            tail_qr_dequant[:, 0:IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode="rint"
                        )
                        tail_qr_rope_slice = tail_qr_dequant[:, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
                        tail_qr_swapped = pl.gather(tail_qr_rope_slice, dim=-1, index=tail_swap_idx)
                        tail_rope_rot = pl.add(
                            pl.mul(tail_qr_rope_slice, tail_cos_tile), pl.mul(tail_qr_swapped, tail_sin_tile)
                        )
                        tail_rope_bf16 = pl.cast(tail_rope_rot, target_type=pl.BF16, mode="rint")
                        qr_bf16_2d[tail_t0 : tail_t0 + 1, tail_h0 : tail_h0 + IDX_NOPE_HEAD_DIM] = tail_qr_nope_bf16
                        qr_bf16_2d[tail_t0 : tail_t0 + 1, tail_h0 + IDX_NOPE_HEAD_DIM : tail_h0 + IDX_HEAD_DIM] = (
                            tail_rope_bf16
                        )

    return idx_qr_mm_tid


@pl.jit.inline(auto_scope=False)
def indexer_qr_hadamard_mm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr_bf16: pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    qr_hadamard_i8: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8]],
    qr_hadamard_scale_dq: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32]],
    qh_mm_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Scalar[pl.TASK_ID], pl.Scalar[pl.TASK_ID]]:
    """q @ hadamard and its INT8 quant, fenced behind the caller's cube ordering."""
    bs = pl.tensor.dim(x, 0)
    bs_heads = bs * IDX_N_HEADS

    # Query Hadamard FP32 intermediate.
    qh_acc_gm = pl.create_tensor([bs_heads, IDX_HEAD_DIM], dtype=pl.FP32)
    with pl.spmd(
        QH_WORKERS,
        name_hint="qr_hadamard_matmul",
        deps=[qh_mm_dep],
        allow_early_resolve=True,
    ) as qh_mm_tid:
        qh_worker = pl.tile.get_block_idx()
        # Shared Hadamard matrix.
        qh_hadamard = hadamard[0:IDX_HEAD_DIM, 0:IDX_HEAD_DIM]
        for idx in pl.range(qh_worker, bs_heads // QH_MM_TILE, QH_WORKERS):
            o0 = idx * QH_MM_TILE
            qh_acc = pl.matmul(qr_bf16[o0 : o0 + QH_MM_TILE, :], qh_hadamard, out_dtype=pl.FP32)
            qh_acc_gm[o0 : o0 + QH_MM_TILE, :] = qh_acc

    with pl.spmd(
        QH_QUANT_WORKERS,
        name_hint="qr_hadamard_quant",
        allow_early_resolve=True,
    ) as qh_quant_tid:
        qh_quant_worker = pl.tile.get_block_idx()
        for idx in pl.range(qh_quant_worker, bs_heads // QH_QUANT_TILE, QH_QUANT_WORKERS):
            o0 = idx * QH_QUANT_TILE
            qh_full_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, 0:IDX_HEAD_DIM]
            # Native Hadamard: unscaled BF16 matmul, then BF16 scaling.
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
            qr_hadamard_scale_dq[o0 : o0 + QH_QUANT_TILE, :] = pl.cast(
                pl.cast(qh_scale_dq, pl.FP16, mode="rint"), pl.FP32
            )
            qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
            qh_q_scaled = pl.row_expand_mul(qh_full_f32, qh_scale_quant)
            qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
            qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
            qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
            qr_hadamard_i8[o0 : o0 + QH_QUANT_TILE, 0:IDX_HEAD_DIM] = qh_i8

    return qh_mm_tid, qh_quant_tid


@pl.jit.inline(auto_scope=False)
def indexer_qr_hadamard(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    qr_hadamard_i8: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8]],
    qr_hadamard_scale_dq: pl.Out[pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32]],
):
    """Query half of the indexer: qr projection, rope, hadamard, and INT8 quant."""
    qr_bf16 = pl.create_tensor([T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    idx_qr_mm_tid = indexer_qr_rope(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        cos,
        sin,
        qr_bf16,
    )
    qh_mm_tid, qh_quant_tid = indexer_qr_hadamard_mm(
        x,
        qr_bf16,
        hadamard,
        qr_hadamard_i8,
        qr_hadamard_scale_dq,
        idx_qr_mm_tid,
    )
    return qh_mm_tid, qh_quant_tid, idx_qr_mm_tid


@pl.jit.inline(auto_scope=False)
def indexer_weights_project(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    weights_gate_dep: pl.Scalar[pl.TASK_ID],
    weights_workers: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor[[T_PAD, IDX_N_HEADS], pl.FP32], pl.Scalar[pl.TASK_ID]]:
    """Shared production head-weight projection and dtype publication."""
    bs = pl.tensor.dim(x, 0)
    row_blocks = (bs + MM_ROW_TILE - 1) // MM_ROW_TILE

    x_flat = x
    weights = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    weights_partial = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
    # Caller-ordered weights projection.
    with pl.spmd(
        weights_workers, name_hint="weights_proj", deps=[weights_gate_dep], allow_early_resolve=True
    ) as _weights_tid:
        w_worker = pl.tile.get_block_idx()
        for w_unit in pl.range(w_worker, row_blocks * (IDX_N_HEADS // WEIGHTS_PROJECTION_N_TILE), weights_workers):
            w_rb = w_unit // (IDX_N_HEADS // WEIGHTS_PROJECTION_N_TILE)
            w_ng = w_unit % (IDX_N_HEADS // WEIGHTS_PROJECTION_N_TILE)
            w_n0 = w_ng * WEIGHTS_PROJECTION_N_TILE
            w_r0 = w_rb * MM_ROW_TILE
            w_rows = pl.min(MM_ROW_TILE, bs - w_r0)
            weights_acc = pl.create_tensor([MM_ROW_TILE, WEIGHTS_PROJECTION_N_TILE], dtype=pl.FP32)
            for db in pl.range(D // WEIGHTS_PROJECTION_K_TILE):
                # Installed CANN9.0 A3 MatMulV2 traverses K in a shape-specific
                # order. Keep each FP32 accumulator live across that sequence;
                # M24 reverses 512-block order while retaining each block's
                # internal order. Larger formal shapes use 256-block order.
                # Evidence: weights_native_codegen_{v1,remaining_v1}.
                k_order = db
                if bs == 24:
                    k_direction = db // 2
                    if w_ng % 2 == 1:
                        k_direction = 7 - db // 2
                    k_order = ((w_ng // 2 * 4 + k_direction) % 8) * 2 + db % 2
                elif bs == 48:
                    k_direction = db
                    if w_ng % 2 == 1:
                        k_direction = 15 - db
                    k_order = (w_ng // 2 * 8 + k_direction) % 16
                elif bs == 96 or bs == 144 or bs == 192 or bs == 240:
                    k_direction = db
                    if w_rb % 2 == 1:
                        k_direction = 15 - db
                    k_shift = w_rb // 2 * 5
                    if bs == 144:
                        k_shift = (w_rb % 8) // 2 * 4
                    elif bs == 192:
                        k_shift = w_rb // 2 * 2
                    elif bs == 240:
                        k_shift = (w_rb % 14) // 2 * 2
                    k_order = (k_shift + k_direction) % 16
                d0 = k_order * WEIGHTS_PROJECTION_K_TILE
                x_tile = pl.slice(
                    x_flat,
                    [MM_ROW_TILE, WEIGHTS_PROJECTION_K_TILE],
                    [w_r0, d0],
                    valid_shape=[w_rows, WEIGHTS_PROJECTION_K_TILE],
                )
                weights_proj_tile = weights_proj[
                    d0 : d0 + WEIGHTS_PROJECTION_K_TILE, w_n0 : w_n0 + WEIGHTS_PROJECTION_N_TILE
                ]
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile, init_cond=(db == 0))
            weights_partial[w_r0 : w_r0 + MM_ROW_TILE, w_n0 : w_n0 + WEIGHTS_PROJECTION_N_TILE] = weights_acc

    with pl.spmd(
        row_blocks,
        name_hint="weights_proj_reduce",
        allow_early_resolve=True,
    ) as weights_tid:
        w_rb = pl.tile.get_block_idx()
        w_r0 = w_rb * MM_ROW_TILE
        w_sum = weights_partial[w_r0 : w_r0 + MM_ROW_TILE, :]
        # Native weights_proj and its scale multiplication each materialize
        # BF16, then A3 QLI consumes FP16 weights.
        w_sum = pl.cast(pl.cast(w_sum, pl.BF16, mode="rint"), pl.FP32)
        w_scaled = pl.cast(pl.mul(w_sum, WEIGHTS_SCALE), pl.BF16, mode="rint")
        weights[w_r0 : w_r0 + MM_ROW_TILE, :] = pl.cast(pl.cast(w_scaled, pl.FP16), pl.FP32)

    return weights, weights_tid


@pl.jit.inline(auto_scope=False)
def indexer_weights_score(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    qr_hadamard_i8: pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8],
    qr_hadamard_scale_dq: pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    cache_write_dep: pl.Scalar[pl.TASK_ID],
    weights_gate_dep: pl.Scalar[pl.TASK_ID],
    qh_quant_tid: pl.Scalar[pl.TASK_ID],
    weights_workers: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32], pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32], pl.Scalar[pl.TASK_ID]]:
    """Weights projection and the score/top-k forest over an already-quantized query."""
    weights, weights_tid = indexer_weights_project(x, weights_proj, weights_gate_dep, weights_workers)

    topk_scores, topk_idxs, leaf_tid = indexer_score_topk_forest(
        qr_hadamard_i8,
        qr_hadamard_scale_dq,
        weights,
        idx_kv_cache,
        idx_block_table,
        position_ids,
        kv_seq_lens,
        topk_scores,
        topk_idxs,
        qh_quant_tid,
        weights_tid,
        cache_write_dep,
    )
    return topk_scores, topk_idxs, leaf_tid


@pl.jit.inline
def indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    late_dep: pl.Scalar[pl.TASK_ID],
    cache_write_dep: pl.Scalar[pl.TASK_ID],
):
    qr_hadamard_i8 = pl.create_tensor([T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T_PAD * IDX_N_HEADS, 1], dtype=pl.FP32)
    _qh_mm_tid, qh_quant_tid, _idx_qr_mm_tid = indexer_qr_hadamard(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        cos,
        sin,
        hadamard,
        qr_hadamard_i8,
        qr_hadamard_scale_dq,
    )
    weights_gate_dep = pl.system.task_dummy(deps=[])
    topk_scores, topk_idxs, _leaf_tid = indexer_weights_score(
        x,
        weights_proj,
        qr_hadamard_i8,
        qr_hadamard_scale_dq,
        idx_kv_cache,
        idx_block_table,
        topk_scores,
        topk_idxs,
        position_ids,
        kv_seq_lens,
        cache_write_dep,
        weights_gate_dep,
        qh_quant_tid,
        TP1_WEIGHTS_WORKERS,
    )
    return topk_scores, topk_idxs
