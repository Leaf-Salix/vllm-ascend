# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""原四split权重投影与同一左折叠，先导出原始sum再隔离BF16边界。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.indexer_query_stages import (
    BF16_WEIGHT_LAYOUT,
    IDX_N_HEADS,
    MM_ROW_TILE,
    T_DYN,
    WEIGHTS_OK,
    WEIGHTS_PARTIAL_ROWS_DYN,
    WEIGHTS_ROWS_DYN,
    WEIGHTS_SCALE,
    D,
    indexer_weights_project_stage,
)


@pl.jit.inline(auto_scope=False)
def reduce_debug(
    partials: pl.Tensor[[WEIGHTS_PARTIAL_ROWS_DYN, IDX_N_HEADS], pl.FP32],
    weights: pl.Tensor[[WEIGHTS_ROWS_DYN, IDX_N_HEADS], pl.FP32],
    projected: pl.Tensor[[WEIGHTS_ROWS_DYN, IDX_N_HEADS], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    MODE: pl.constexpr,
):
    row_blocks = (tokens + MM_ROW_TILE - 1) // MM_ROW_TILE
    t_pad = row_blocks * MM_ROW_TILE
    for rb in pl.range(worker, row_blocks, 48):
        row = rb * MM_ROW_TILE
        raw = partials[row : row + MM_ROW_TILE, :]
        for kb in pl.unroll(1, WEIGHTS_OK):
            offset = kb * t_pad + row
            raw = pl.add(raw, partials[offset : offset + MM_ROW_TILE, :])
        projected[row : row + MM_ROW_TILE, :] = raw
        if MODE == 1:
            linear = pl.cast(pl.cast(raw, pl.BF16, mode="rint"), pl.FP32)
            scaled = pl.cast(pl.cast(pl.mul(linear, WEIGHTS_SCALE), pl.BF16, mode="rint"), pl.FP32)
        else:
            scaled = pl.mul(raw, WEIGHTS_SCALE)
        weights[row : row + MM_ROW_TILE, :] = scaled
    return weights, projected


@pl.jit
def weightproj_arm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[IDX_N_HEADS, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    partials: pl.InOut[pl.Tensor[[WEIGHTS_PARTIAL_ROWS_DYN, IDX_N_HEADS], pl.FP32]],
    weights: pl.InOut[pl.Tensor[[WEIGHTS_ROWS_DYN, IDX_N_HEADS], pl.FP32]],
    projected: pl.InOut[pl.Tensor[[WEIGHTS_ROWS_DYN, IDX_N_HEADS], pl.FP32]],
    MODE: pl.constexpr,
):
    tokens = pl.tensor.dim(x, 0)
    for worker in pl.spmd(24, name_hint="original_weights_project", allow_early_resolve=True):
        partials = indexer_weights_project_stage(x, weight, partials, worker, 24)
    for writer in pl.spmd(48, name_hint="weights_rounding_probe", allow_early_resolve=True):
        weights, projected = reduce_debug(partials, weights, projected, tokens, writer, MODE)
    return partials, weights, projected
