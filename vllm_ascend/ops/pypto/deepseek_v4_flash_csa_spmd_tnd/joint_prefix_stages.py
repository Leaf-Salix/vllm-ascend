# SPDX-License-Identifier: Apache-2.0
"""Joint main value/gate and KV projection; Native roots and destinations are unchanged.

One row block has 20 N128 jobs: eight main value, eight main gate,
four original KV, on physical cores 4..23. No weights are packed
or cached, and no request membership is inferred from T. First matmul
establishes dynamic compact-M accumulation; ascending K256 chunks cover K4096.
Exactness is validated for the adopted workload cases; see the central implementation log.
"""

import pypto.language as pl

from .qkv_stages import BF16_WEIGHT_LAYOUT
from .runtime_abi import KV_ROWS, MPROJ, T

K_TILE = 256
N_TILE = 128
JOBS_PER_ROW_BLOCK = 20
INDEXER_BRANCH_CORES = 4


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
    weight: pl.Tensor[[512, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    output: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    ROW_TILE: pl.constexpr,
):
    # column在KV分支属于[0,512)，绑定标量让NZ切片可追踪非负定义。
    nz_column = pl.max(0, column)
    first_x = pl.slice(x, [ROW_TILE, K_TILE], [row, 0], valid_shape=[rows, K_TILE])
    first_w = weight[nz_column : nz_column + N_TILE, 0:K_TILE]
    acc = pl.matmul(first_x, first_w, b_trans=True, out_dtype=pl.FP32)
    for block in pl.pipeline(1, 4096 // K_TILE, stage=2):
        k0 = block * K_TILE
        current_x = pl.slice(x, [ROW_TILE, K_TILE], [row, k0], valid_shape=[rows, K_TILE])
        current_w = weight[nz_column : nz_column + N_TILE, k0 : k0 + K_TILE]
        acc = pl.matmul_acc(acc, current_x, current_w, b_trans=True)
    output = pl.assemble(output, acc, [row, column])
    return output


@pl.jit.inline(auto_scope=False)
def joint_bf16_prefix_stage(
    normed: pl.Tensor[[T, 4096], pl.BF16],
    main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
    main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
    main_value: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    main_score: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    wkv: pl.Tensor[[512, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ROW_TILE: pl.constexpr,
):
    tokens = pl.tensor.dim(normed, 0)
    row_blocks = (tokens + ROW_TILE - 1) // ROW_TILE
    projection_worker = (worker + WORKERS - INDEXER_BRANCH_CORES) % WORKERS
    for job in pl.range(projection_worker, row_blocks * JOBS_PER_ROW_BLOCK, WORKERS):
        row = (job // JOBS_PER_ROW_BLOCK) * ROW_TILE
        rows = pl.min(ROW_TILE, tokens - row)
        projection_column = job % JOBS_PER_ROW_BLOCK
        if projection_column < 8:
            column = projection_column * N_TILE
            main_value = _project_main(normed, main_wkv, main_value, row, column, rows, ROW_TILE)
        elif projection_column < 16:
            column = (projection_column - 8) * N_TILE
            main_score = _project_main(normed, main_wgate, main_score, row, column, rows, ROW_TILE)
        else:
            # 此分支projection_column为16..19；取模保留同一列号，并让NZ证明
            # 不依赖前面if的范围约束（减法本身不能证明非负）。
            column = (projection_column % 4) * N_TILE
            kv_fp32 = _project_forward(normed, wkv, kv_fp32, row, column, rows, ROW_TILE)
    return main_value, main_score, kv_fp32
