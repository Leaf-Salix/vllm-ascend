# SPDX-License-Identifier: Apache-2.0
"""Source template for a joint grid over five existing Native BF16 projections."""

SOURCE = '''# SPDX-License-Identifier: Apache-2.0
"""Experimental non-QA projection grid; Native roots and destinations are unchanged.

One row block has 24 N128 jobs: two Indexer value, two Indexer gate,
eight main value, eight main gate, four original KV. No weights are packed
or cached, and no request membership is inferred from T. First matmul
establishes dynamic compact-M accumulation; ascending K256 chunks cover K4096.
This changes the old K512 grouping and therefore requires strict device checks.
"""

import pypto.language as pl

from .runtime_abi import IPROJ, KV_ROWS, MPROJ, T

K_TILE = 256
N_TILE = 128
JOBS_PER_ROW_BLOCK = 24


@pl.jit.inline(auto_scope=False)
def _project_indexer(
    x: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[256, 4096], pl.BF16],
    output: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    ROW_TILE: pl.constexpr,
):
    first_x = pl.slice(x, [ROW_TILE, K_TILE], [row, 0], valid_shape=[rows, K_TILE])
    first_w = weight[column : column + N_TILE, 0:K_TILE]
    acc = pl.matmul(first_x, first_w, b_trans=True, out_dtype=pl.FP32)
    for block in pl.pipeline(1, 4096 // K_TILE, stage=2):
        k0 = block * K_TILE
        current_x = pl.slice(x, [ROW_TILE, K_TILE], [row, k0], valid_shape=[rows, K_TILE])
        current_w = weight[column : column + N_TILE, k0 : k0 + K_TILE]
        acc = pl.matmul_acc(acc, current_x, current_w, b_trans=True)
    output = pl.assemble(output, acc, [row, column])
    return output


@pl.jit.inline(auto_scope=False)
def _project_main(
    x: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[1024, 4096], pl.BF16],
    output: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    ROW_TILE: pl.constexpr,
):
    first_x = pl.slice(x, [ROW_TILE, K_TILE], [row, 0], valid_shape=[rows, K_TILE])
    first_w = weight[column : column + N_TILE, 0:K_TILE]
    acc = pl.matmul(first_x, first_w, b_trans=True, out_dtype=pl.FP32)
    for block in pl.pipeline(1, 4096 // K_TILE, stage=2):
        k0 = block * K_TILE
        current_x = pl.slice(x, [ROW_TILE, K_TILE], [row, k0], valid_shape=[rows, K_TILE])
        current_w = weight[column : column + N_TILE, k0 : k0 + K_TILE]
        acc = pl.matmul_acc(acc, current_x, current_w, b_trans=True)
    output = pl.assemble(output, acc, [row, column])
    return output


@pl.jit.inline(auto_scope=False)
def _project_forward(
    x: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[4096, 512], pl.BF16],
    output: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    ROW_TILE: pl.constexpr,
):
    first_x = pl.slice(x, [ROW_TILE, K_TILE], [row, 0], valid_shape=[rows, K_TILE])
    first_w = weight[0:K_TILE, column : column + N_TILE]
    acc = pl.matmul(first_x, first_w, out_dtype=pl.FP32)
    for block in pl.pipeline(1, 4096 // K_TILE, stage=2):
        k0 = block * K_TILE
        current_x = pl.slice(x, [ROW_TILE, K_TILE], [row, k0], valid_shape=[rows, K_TILE])
        current_w = weight[k0 : k0 + K_TILE, column : column + N_TILE]
        acc = pl.matmul_acc(acc, current_x, current_w)
    output = pl.assemble(output, acc, [row, column])
    return output


@pl.jit.inline(auto_scope=False)
def joint_bf16_prefix_stage(
    normed: pl.Tensor[[T, 4096], pl.BF16],
    idx_wkv: pl.Tensor[[256, 4096], pl.BF16],
    idx_wgate: pl.Tensor[[256, 4096], pl.BF16],
    idx_value: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],
    idx_score: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],
    main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
    main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
    main_value: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    main_score: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    wkv: pl.Tensor[[4096, 512], pl.BF16],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ROW_TILE: pl.constexpr,
):
    tokens = pl.tensor.dim(normed, 0)
    row_blocks = (tokens + ROW_TILE - 1) // ROW_TILE
    for job in pl.range(worker, row_blocks * JOBS_PER_ROW_BLOCK, WORKERS):
        row = (job // JOBS_PER_ROW_BLOCK) * ROW_TILE
        rows = pl.min(ROW_TILE, tokens - row)
        projection_column = job % JOBS_PER_ROW_BLOCK
        if projection_column < 2:
            column = projection_column * N_TILE
            idx_value = _project_indexer(normed, idx_wkv, idx_value, row, column, rows, ROW_TILE)
        elif projection_column < 4:
            column = (projection_column - 2) * N_TILE
            idx_score = _project_indexer(normed, idx_wgate, idx_score, row, column, rows, ROW_TILE)
        elif projection_column < 12:
            column = (projection_column - 4) * N_TILE
            main_value = _project_main(normed, main_wkv, main_value, row, column, rows, ROW_TILE)
        elif projection_column < 20:
            column = (projection_column - 12) * N_TILE
            main_score = _project_main(normed, main_wgate, main_score, row, column, rows, ROW_TILE)
        else:
            column = (projection_column - 20) * N_TILE
            kv_fp32 = _project_forward(normed, wkv, kv_fp32, row, column, rows, ROW_TILE)
    return idx_value, idx_score, main_value, main_score, kv_fp32
'''
