# SPDX-License-Identifier: Apache-2.0
"""Remove only dead forwarded weight arguments; retain the Native-order math."""

import pypto.language as pl

from ..deepseek_v4_flash_csa_tnd_precision.decode_indexer import (
    B_DYN,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    IDX_NATIVE_CACHE_BLOCK_NUM_DYN,
    IDX_TOPK,
    INDEXER_PAGE_BYTES_DYN,
    INDEXER_TABLE_COLUMNS_DYN,
    Q_LORA,
    QUERY_BOUNDS_DYN,
    ROPE_HEAD_DIM,
    T_DYN,
    T_PAD,
    TP1_WEIGHTS_WORKERS,
    D,
    indexer_native_weights,
    indexer_qr_hadamard,
    indexer_score_topk_forest,
)


@pl.jit.inline(auto_scope=False)
def indexer_weights_score(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    native_indexer_weights: pl.Tensor[[T_DYN, IDX_N_HEADS], pl.FP16],
    qr_hadamard_i8: pl.Tensor[[T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8],
    qr_hadamard_scale_dq: pl.Tensor[[T_PAD * IDX_N_HEADS, 1], pl.FP32],
    idx_native_kv_cache: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    token_request: pl.Tensor[[T_DYN], pl.INT32],
    query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    cache_write_dep: pl.Scalar[pl.TASK_ID],
    weights_gate_dep: pl.Scalar[pl.TASK_ID],
    qh_quant_tid: pl.Scalar[pl.TASK_ID],
    weights_workers: pl.Scalar[pl.INDEX],
    host_max_seq_len: pl.Scalar[pl.INT32],
) -> tuple[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32], pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32], pl.Scalar[pl.TASK_ID]]:
    """Native weight publication and the score/top-k forest over an already-quantized query."""
    (weights, weights_tid) = indexer_native_weights(native_indexer_weights, weights_gate_dep)
    (topk_scores, topk_idxs, leaf_tid) = indexer_score_topk_forest(
        qr_hadamard_i8,
        qr_hadamard_scale_dq,
        weights,
        idx_native_kv_cache,
        idx_block_table,
        position_ids,
        kv_seq_lens,
        token_request,
        query_start_loc,
        topk_scores,
        topk_idxs,
        qh_quant_tid,
        weights_tid,
        cache_write_dep,
        host_max_seq_len,
    )
    return (topk_scores, topk_idxs, leaf_tid)


@pl.jit.inline
def indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    native_indexer_weights: pl.Tensor[[T_DYN, IDX_N_HEADS], pl.FP16],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_native_kv_cache: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    token_request: pl.Tensor[[T_DYN], pl.INT32],
    query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    late_dep: pl.Scalar[pl.TASK_ID],
    cache_write_dep: pl.Scalar[pl.TASK_ID],
    host_max_seq_len: pl.Scalar[pl.INT32],
):
    qr_hadamard_i8 = pl.create_tensor([T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T_PAD * IDX_N_HEADS, 1], dtype=pl.FP32)
    (_qh_mm_tid, qh_quant_tid, _idx_qr_mm_tid) = indexer_qr_hadamard(
        x, qr, qr_scale, wq_b, wq_b_scale, cos, sin, hadamard, qr_hadamard_i8, qr_hadamard_scale_dq
    )
    weights_gate_dep = pl.system.task_dummy(deps=[])
    (topk_scores, topk_idxs, _leaf_tid) = indexer_weights_score(
        x,
        native_indexer_weights,
        qr_hadamard_i8,
        qr_hadamard_scale_dq,
        idx_native_kv_cache,
        idx_block_table,
        topk_scores,
        topk_idxs,
        position_ids,
        kv_seq_lens,
        token_request,
        query_start_loc,
        cache_write_dep,
        weights_gate_dep,
        qh_quant_tid,
        TP1_WEIGHTS_WORKERS,
        host_max_seq_len,
    )
    return (topk_scores, topk_idxs)
