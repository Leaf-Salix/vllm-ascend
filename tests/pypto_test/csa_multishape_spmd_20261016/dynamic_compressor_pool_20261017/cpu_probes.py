# SPDX-License-Identifier: Apache-2.0
"""动态请求驱动 pool/state 的四个独立编译入口，不是完整 CSA。

每个包装只分配八个测试 worker，把动态描述符交给正式 helper；不提供样本
形状，不模拟阶段同步，不运行设备。pool 输出和 state 都是部分更新，使用
InOut 明确保护未写区；完整组合必须另行安排 pool→commit 的核间依赖。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.compressor_pool_stages import (
    COMPRESS_RATIO,
    HEAD_DIM,
    IDX_HEAD_DIM,
    IDX_OUT_DIM,
    OUT_DIM,
    POOL_ROWS_DYN,
    PROJECTION_ROWS_DYN,
    QUERY_BOUNDS_DYN,
    REQUESTS_DYN,
    STATE_ELEMENTS_DYN,
    STATE_PAGES_DYN,
    T_DYN,
    TABLE_COLUMNS_DYN,
    compressor_pool_stage,
    compressor_state_commit_stage,
    indexer_compressor_pool_stage,
    indexer_compressor_state_commit_stage,
)


@pl.jit
def compressor_pool_probe(
    kv_proj: pl.Tensor[[PROJECTION_ROWS_DYN, OUT_DIM], pl.FP32],
    score_proj: pl.Tensor[[PROJECTION_ROWS_DYN, OUT_DIM], pl.FP32],
    state: pl.Tensor[[STATE_PAGES_DYN, STATE_ELEMENTS_DYN], pl.FP32],
    state_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    pooled: pl.InOut[pl.Tensor[[POOL_ROWS_DYN, HEAD_DIM], pl.FP32]],
):
    with pl.spmd(8):
        worker = pl.tile.get_block_idx()
        pooled = compressor_pool_stage(
            kv_proj, score_proj, state, state_table, ape, positions, bounds, lengths, pooled, worker, 8
        )
    return pooled


@pl.jit
def indexer_compressor_pool_probe(
    kv_proj: pl.Tensor[[PROJECTION_ROWS_DYN, IDX_OUT_DIM], pl.FP32],
    score_proj: pl.Tensor[[PROJECTION_ROWS_DYN, IDX_OUT_DIM], pl.FP32],
    state: pl.Tensor[[STATE_PAGES_DYN, STATE_ELEMENTS_DYN], pl.FP32],
    state_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    ape: pl.Tensor[[COMPRESS_RATIO, IDX_OUT_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    pooled: pl.InOut[pl.Tensor[[POOL_ROWS_DYN, IDX_HEAD_DIM], pl.FP32]],
):
    with pl.spmd(8):
        worker = pl.tile.get_block_idx()
        pooled = indexer_compressor_pool_stage(
            kv_proj, score_proj, state, state_table, ape, positions, bounds, lengths, pooled, worker, 8
        )
    return pooled


@pl.jit
def compressor_state_commit_probe(
    kv_proj: pl.Tensor[[PROJECTION_ROWS_DYN, OUT_DIM], pl.FP32],
    score_proj: pl.Tensor[[PROJECTION_ROWS_DYN, OUT_DIM], pl.FP32],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    state_slots: pl.Tensor[[T_DYN, 2], pl.INT32],
    state: pl.InOut[pl.Tensor[[STATE_PAGES_DYN, STATE_ELEMENTS_DYN], pl.FP32]],
):
    with pl.spmd(8):
        worker = pl.tile.get_block_idx()
        state = compressor_state_commit_stage(
            kv_proj, score_proj, ape, positions, bounds, lengths, state_slots, state, worker, 8
        )
    return state


@pl.jit
def indexer_compressor_state_commit_probe(
    kv_proj: pl.Tensor[[PROJECTION_ROWS_DYN, IDX_OUT_DIM], pl.FP32],
    score_proj: pl.Tensor[[PROJECTION_ROWS_DYN, IDX_OUT_DIM], pl.FP32],
    ape: pl.Tensor[[COMPRESS_RATIO, IDX_OUT_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    state_slots: pl.Tensor[[T_DYN, 2], pl.INT32],
    state: pl.InOut[pl.Tensor[[STATE_PAGES_DYN, STATE_ELEMENTS_DYN], pl.FP32]],
):
    with pl.spmd(8):
        worker = pl.tile.get_block_idx()
        state = indexer_compressor_state_commit_stage(
            kv_proj, score_proj, ape, positions, bounds, lengths, state_slots, state, worker, 8
        )
    return state
