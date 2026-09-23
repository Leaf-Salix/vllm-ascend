# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TP1 attention-only CSA adapted from the pinned pypto-lib reference."""

import pypto.language as pl

from .compact_metadata import build_compact_row_offsets
from .config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    KV_ORI_BLOCK_NUM,
)
from .config import (
    FLASH as M,
)
from .config import (
    TP as TP_SIZE,
)
from .decode_compressor_ratio4 import compressor_ratio4
from .decode_indexer import indexer
from .decode_indexer_compressor import indexer_compressor
from .decode_o_proj import LOCAL_T, LOCAL_T_PAD, decode_o_proj_tp1
from .decode_sparse_attn_csa import T_PAD, sparse_attn_csa_tp1
from .layout import (
    COMPRESSED_ROWS_DYN,
    COMPRESSED_TABLE_COLUMNS_DYN,
    INDEXER_PAGE_BYTES_DYN,
    INDEXER_ROWS_DYN,
    INDEXER_TABLE_COLUMNS_DYN,
    INNER_STATE_PAGE_ELEMENTS_DYN,
    INNER_STATE_TABLE_COLUMNS_DYN,
    ORIGINAL_TABLE_COLUMNS_DYN,
    QUERY_BOUNDS_DYN,
    STATE_PAGE_ELEMENTS_DYN,
    STATE_TABLE_COLUMNS_DYN,
)
from .qkv_proj_rope import qkv_proj_rope

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # per-request axis
T_DYN = pl.dynamic("T_DYN")  # T = B * S
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic("CSA_STATE_BLOCK_NUM_DYN")
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")

# model config
B = DECODE_BATCH // TP_SIZE
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_TOPK = M.index_topk
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel constants
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
MAIN_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_DIM = 2 * INNER_OUT_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_LEN = COFF * COMPRESS_RATIO
ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
CMP_MAX_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO
CMP_MAX_BLOCKS = (CMP_MAX_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE
IDX_MAX_BLOCKS = CMP_MAX_BLOCKS

# Indexer dependency slots across conditional scopes.
IDX_QUERY_DEP_SLOT = 0
IDX_HADAMARD_DEP_SLOT = 1
IDX_QUANT_DEP_SLOT = 2
IDX_LEAF_DEP_SLOT = 3
IDX_DEP_COUNT = IDX_LEAF_DEP_SLOT + 1

# tiling
CSA_PROJECTION_PACK_ROW_TILE = 8
CSA_PROJECTION_PACK_WORKERS = 16
CSA_ALL_VISIBLE_WORKERS = 16
CSA_WB_TOKEN_TILE = 8
CSA_WB_WORKERS = 48  # CSA cache-write workers
TP1_CSA_WB_WORKERS = 8  # TP1 CSA cache-write workers

if T != LOCAL_T:
    raise ValueError(f"CSA token capacity {T} must equal TP local token capacity {LOCAL_T}")
if T_PAD != LOCAL_T_PAD:
    raise ValueError(f"CSA token capacity {T_PAD} must equal TP local token capacity {LOCAL_T_PAD}")


def _decode_csa_tp1_attention(
    x_normed_t: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_freqs_cos: pl.Tensor[[COMPRESSED_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_freqs_sin: pl.Tensor[[COMPRESSED_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    inner_freqs_cos: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    inner_freqs_sin: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32]],
    state_block_table: pl.Tensor[[B_DYN, STATE_TABLE_COLUMNS_DYN], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32]],
    inner_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_TABLE_COLUMNS_DYN], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]],
    idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
    ori_block_table: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[COMPRESSED_ROWS_DYN, 2], pl.INT32],
    idx_slot_mapping: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32],
    state_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
    inner_state_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    cmp_query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    cmp_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    idx_query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    idx_topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    idx_topk: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """TP1 attention with Native interleaved FP32 frequencies and no outer HC."""
    x_normed_t.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    cmp_freqs_cos.bind_dynamic(0, COMPRESSED_ROWS_DYN)
    cmp_freqs_sin.bind_dynamic(0, COMPRESSED_ROWS_DYN)
    inner_freqs_cos.bind_dynamic(0, INDEXER_ROWS_DYN)
    inner_freqs_sin.bind_dynamic(0, INDEXER_ROWS_DYN)
    compress_state.bind_dynamic(0, MAIN_STATE_BLOCK_NUM_DYN)
    compress_state.bind_dynamic(1, STATE_PAGE_ELEMENTS_DYN)
    state_block_table.bind_dynamic(0, B_DYN)
    state_block_table.bind_dynamic(1, STATE_TABLE_COLUMNS_DYN)
    inner_compress_state.bind_dynamic(0, INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state.bind_dynamic(1, INNER_STATE_PAGE_ELEMENTS_DYN)
    inner_state_block_table.bind_dynamic(0, B_DYN)
    inner_state_block_table.bind_dynamic(1, INNER_STATE_TABLE_COLUMNS_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    idx_kv_cache.bind_dynamic(0, IDX_CACHE_BLOCK_NUM_DYN)
    idx_kv_cache.bind_dynamic(1, INDEXER_PAGE_BYTES_DYN)
    ori_slot_mapping.bind_dynamic(0, T_DYN)
    ori_block_table.bind_dynamic(0, B_DYN)
    ori_block_table.bind_dynamic(1, ORIGINAL_TABLE_COLUMNS_DYN)
    cmp_slot_mapping.bind_dynamic(0, COMPRESSED_ROWS_DYN)
    idx_slot_mapping.bind_dynamic(0, INDEXER_ROWS_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    inner_state_slot_mapping.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)
    cmp_query_start_loc.bind_dynamic(0, QUERY_BOUNDS_DYN)
    cmp_seq_lens.bind_dynamic(0, B_DYN)
    idx_query_start_loc.bind_dynamic(0, QUERY_BOUNDS_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    cmp_block_table.bind_dynamic(1, COMPRESSED_TABLE_COLUMNS_DYN)
    idx_block_table.bind_dynamic(0, B_DYN)
    idx_block_table.bind_dynamic(1, INDEXER_TABLE_COLUMNS_DYN)
    attn_out.bind_dynamic(0, T_DYN)
    idx_topk.bind_dynamic(0, T_DYN)
    idx_topk_scores.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x_normed_t, 0)
    wb_blocks = (t_dim + CSA_WB_TOKEN_TILE - 1) // CSA_WB_TOKEN_TILE

    idx_sin_signed = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_row_offsets = pl.create_tensor([pl.tensor.dim(cmp_seq_lens, 0)], dtype=pl.INT32)
    idx_row_offsets = pl.create_tensor([pl.tensor.dim(kv_seq_lens, 0)], dtype=pl.INT32)
    # Reuse this task for per-request compact offsets and normal-RoPE signs.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_rope_sign") as rope_tid:
        build_compact_row_offsets(cmp_query_start_loc, cmp_seq_lens, cmp_row_offsets)
        build_compact_row_offsets(idx_query_start_loc, kv_seq_lens, idx_row_offsets)
        il_ones = pl.tile.full([4, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        il_lane_ids = pl.cast(pl.tile.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32)
        il_col = pl.col_expand_mul(il_ones, il_lane_ids)
        il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)
        for rope_t0 in pl.range(0, t_dim, 4):
            rope_rows = pl.min(4, t_dim - rope_t0)
            rope_sin_rows = pl.load(freqs_sin, [rope_t0, 0], [4, ROPE_HEAD_DIM], valid_shape=[rope_rows, ROPE_HEAD_DIM])
            rope_sign_rows = pl.set_validshape(il_sign, rope_rows, ROPE_HEAD_DIM)
            pl.store(pl.mul(rope_sin_rows, rope_sign_rows), [rope_t0, 0], idx_sin_signed)

    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    position_ids_t1 = pl.reshape(position_ids, [t_dim, 1])
    with pl.scope():
        # Projection-chain dependency marker.
        late_dep = pl.system.task_dummy(deps=[rope_tid])
        qkv_proj_rope(
            x_normed_t,
            wq_a,
            wq_b,
            wq_b_scale,
            wkv,
            freqs_cos,
            freqs_sin,
            gamma_cq,
            gamma_ckv,
            q,
            kv,
            qr,
            qr_scale,
            late_dep,
        )

        ori_block_num = pl.tensor.dim(kv_cache, 0)
        kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
        with pl.spmd(TP1_CSA_WB_WORKERS, name_hint="csa_cache_writeback"):
            wb_worker = pl.tile.get_block_idx()
            for wb_blk in pl.range(wb_worker, wb_blocks, TP1_CSA_WB_WORKERS):
                wb_t0 = wb_blk * CSA_WB_TOKEN_TILE
                for write_dt in pl.range(pl.min(CSA_WB_TOKEN_TILE, t_dim - wb_t0)):
                    write_t = wb_t0 + write_dt
                    write_page = pl.read(ori_slot_mapping, [write_t, 0])
                    write_offset = pl.read(ori_slot_mapping, [write_t, 1])
                    if write_page >= 0 and write_offset >= 0:
                        write_row = pl.cast(write_page, pl.INDEX) * BLOCK_SIZE + write_offset
                        kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]

        cmp_out = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
        cmp_out, cmp_cache_write_tid, cmp_kv_score_tid = compressor_ratio4(
            x_normed_t,
            cmp_out,
            compress_state,
            state_block_table,
            cmp_wkv,
            cmp_wgate,
            cmp_ape,
            cmp_norm_w,
            cmp_freqs_cos,
            cmp_freqs_sin,
            cmp_row_offsets,
            cmp_kv,
            position_ids,
            cmp_slot_mapping,
            state_slot_mapping,
            late_dep,
        )
        idx_kv_unused = pl.create_tensor([t_dim, IDX_HEAD_DIM], dtype=pl.FP32)
        idx_cache_write_tid = indexer_compressor(
            x_normed_t,
            idx_kv_unused,
            inner_compress_state,
            inner_state_block_table,
            inner_wkv,
            inner_wgate,
            inner_ape,
            inner_norm_w,
            inner_freqs_cos,
            inner_freqs_sin,
            idx_row_offsets,
            hadamard_idx,
            idx_kv_cache,
            position_ids,
            idx_slot_mapping,
            inner_state_slot_mapping,
            late_dep,
            cmp_kv_score_tid,
        )
        # Bound indexer scratch to its own runtime scope.
        with pl.scope():
            idx_topk_scores, idx_topk = indexer(
                x_normed_t,
                qr,
                qr_scale,
                idx_wq_b,
                idx_wq_b_scale,
                weights_proj,
                freqs_cos,
                idx_sin_signed,
                hadamard_idx,
                idx_kv_cache,
                idx_block_table,
                idx_topk_scores,
                idx_topk,
                position_ids,
                kv_seq_lens,
                late_dep,
                idx_cache_write_tid,
            )

        # sparse_attn_csa folds the compressed-slot masking + valid-block flags in from the
        # raw indexer topk + position.
        o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16)
        o_packed_heads, heads_dep = sparse_attn_csa_tp1(
            q,
            kv_cache,
            ori_block_table,
            cmp_kv,
            cmp_block_table,
            idx_topk,
            position_ids_t1,
            attn_sink,
            freqs_cos,
            freqs_sin,
            o_packed_heads,
        )
        attn_out = decode_o_proj_tp1(o_packed_heads, wo_a, wo_b, wo_b_scale, attn_out, heads_dep)
    return attn_out


decode_csa_tp1_attention = pl.jit.inline(auto_scope=False)(_decode_csa_tp1_attention)
decode_csa_tp1_attention_test = pl.jit(auto_scope=False)(_decode_csa_tp1_attention)
