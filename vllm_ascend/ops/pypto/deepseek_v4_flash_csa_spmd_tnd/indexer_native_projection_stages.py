# SPDX-License-Identifier: Apache-2.0
"""Indexer Compressor的Native顺序投影；保留原M16/N32逻辑生产者，内部拆N16。

K128与列组循环偏移参考Native Compressor及旧精度分支，不改state/池化或信号。
参数保留用于单因素对拍；生产固定K128与Native列组循环顺序。
"""

import pypto.language as pl

from .compressor_projection_stages import (
    INDEXER_M_TILE,
    INDEXER_N_TILE,
    INDEXER_OUT_DIM,
    INDEXER_PAD_ROWS_DYN,
    T_DYN,
    D,
)
from .request_geometry import QUERY_BOUNDS_DYN


@pl.jit.inline(auto_scope=False)
def indexer_native_projection_stage(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[INDEXER_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[INDEXER_OUT_DIM, D], pl.BF16],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    kv_proj: pl.InOut[pl.Tensor[[INDEXER_PAD_ROWS_DYN, INDEXER_OUT_DIM], pl.FP32]],
    score_proj: pl.InOut[pl.Tensor[[INDEXER_PAD_ROWS_DYN, INDEXER_OUT_DIM], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    K_STEP: pl.constexpr,
    ROTATE: pl.constexpr,
):
    tokens = pl.tensor.dim(x, 0)
    padded = (tokens + INDEXER_M_TILE - 1) // INDEXER_M_TILE * INDEXER_M_TILE
    column_group = pl.cast(64, pl.INDEX)
    if ROTATE:
        # Native arch32 SetBaseSize: uniform requests select 8 groups/head
        # at this capacity; ragged requests retain 2. Read live TND bounds.
        first = pl.read(bounds, [1]) - pl.read(bounds, [0])
        deviation = pl.cast(0, pl.INT32)
        for request in pl.range(1, pl.tensor.dim(bounds, 0) - 1):
            delta = pl.read(bounds, [request + 1]) - pl.read(bounds, [request]) - first
            deviation = deviation + delta * delta
        if deviation == 0 and tokens <= 1536:
            column_group = pl.cast(16, pl.INDEX)
    for job in pl.range(worker, padded * INDEXER_OUT_DIM // (INDEXER_M_TILE * INDEXER_N_TILE), WORKERS):
        row = job // (INDEXER_OUT_DIM // INDEXER_N_TILE) * INDEXER_M_TILE
        parent_column = job % (INDEXER_OUT_DIM // INDEXER_N_TILE) * INDEXER_N_TILE
        rows = pl.min(INDEXER_M_TILE, tokens - row)
        for sub in pl.range(2):
            column = parent_column + sub * 16
            value = pl.create_tensor([INDEXER_M_TILE, 16], dtype=pl.FP32)
            score = pl.create_tensor([INDEXER_M_TILE, 16], dtype=pl.FP32)
            for step in pl.pipeline(0, D // K_STEP, stage=2):
                offset = step * K_STEP
                if ROTATE:
                    offset = (step * K_STEP + (column % 128 // column_group) * 256) % D
                a = pl.slice(x, [INDEXER_M_TILE, K_STEP], [row, offset], valid_shape=[rows, K_STEP])
                b_value = pl.slice(wkv, [16, K_STEP], [column, offset])
                b_score = pl.slice(wgate, [16, K_STEP], [column, offset])
                # M16 spans one fractal; preserve the original init_cond path.
                value = pl.matmul_acc(value, a, b_value, b_trans=True, init_cond=(step == 0))
                score = pl.matmul_acc(score, a, b_score, b_trans=True, init_cond=(step == 0))
            kv_proj[row : row + INDEXER_M_TILE, column : column + 16] = value
            score_proj[row : row + INDEXER_M_TILE, column : column + 16] = score
    return kv_proj, score_proj
