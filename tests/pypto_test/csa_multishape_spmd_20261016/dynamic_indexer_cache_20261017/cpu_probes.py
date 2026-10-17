# SPDX-License-Identifier: Apache-2.0
"""Indexer key量化/scale提交的两个动态签名CPU编译壳，不执行设备。

只为分别检查Tile尾行和64B读改写的lowering，未组合其同步依赖。
所有工作区由参数显式给出；不注入样本shape，部分写入保留InOut合同。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_cache_stages import (
    CACHE_BYTES_DYN,
    CACHE_PAGES_DYN,
    HEAD_DIM,
    QUERY_BOUNDS_DYN,
    REQUESTS_DYN,
    SCALE_ROWS_DYN,
    SLOT_ROWS_DYN,
    T_DYN,
    VALUE_ROWS_DYN,
    indexer_compact_quant_cache_stage,
    indexer_compact_scale_commit_stage,
)


@pl.jit
def indexer_compact_quant_cache_probe(
    values: pl.Tensor[[VALUE_ROWS_DYN, HEAD_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    slots: pl.Tensor[[SLOT_ROWS_DYN, 2], pl.INT32],
    scale_values: pl.InOut[pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32]],
    kv_debug: pl.InOut[pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32]],
    cache: pl.InOut[pl.Tensor[[CACHE_PAGES_DYN, CACHE_BYTES_DYN], pl.INT8]],
):
    with pl.spmd(8):
        worker = pl.tile.get_block_idx()
        (scale_values, kv_debug, cache) = indexer_compact_quant_cache_stage(
            values, positions, bounds, lengths, offsets, slots, scale_values, kv_debug, cache, worker, 8
        )
    return (scale_values, kv_debug, cache)


@pl.jit
def indexer_compact_scale_commit_probe(
    scale_values: pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    slots: pl.Tensor[[SLOT_ROWS_DYN, 2], pl.INT32],
    cache: pl.InOut[pl.Tensor[[CACHE_PAGES_DYN, CACHE_BYTES_DYN], pl.INT8]],
):
    with pl.spmd(1):
        worker = pl.tile.get_block_idx()
        cache = indexer_compact_scale_commit_stage(
            scale_values, positions, bounds, lengths, offsets, slots, cache, worker
        )
    return cache
