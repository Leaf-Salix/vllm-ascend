# SPDX-License-Identifier: Apache-2.0
"""固定权重/舍入，只隔离四份partial左折叠与连续Cube K累加。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.indexer_query_stages import (
    BF16_WEIGHT_LAYOUT,
    T_DYN,
    WEIGHTS_PARTIAL_ROWS_DYN,
    WEIGHTS_ROWS_DYN,
    WEIGHTS_SCALE,
    indexer_weights_project_stage,
)
from weightproj_kernels import reduce_debug


@pl.jit
def weight240_arm(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[64, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    partials: pl.InOut[pl.Tensor[[WEIGHTS_PARTIAL_ROWS_DYN, 64], pl.FP32]],
    scaled: pl.InOut[pl.Tensor[[WEIGHTS_ROWS_DYN, 64], pl.FP32]],
    projected: pl.InOut[pl.Tensor[[WEIGHTS_ROWS_DYN, 64], pl.FP32]],
    MODE: pl.constexpr,
):
    tokens = pl.tensor.dim(x, 0)
    if MODE == 0:
        for worker in pl.spmd(24, name_hint="production_weight_partials", allow_early_resolve=True):
            partials = indexer_weights_project_stage(x, weight, partials, worker, 24)
        for writer in pl.spmd(48, name_hint="production_weight_rounding", allow_early_resolve=True):
            scaled, projected = reduce_debug(partials, scaled, projected, tokens, writer, 1)
    else:
        for worker in pl.spmd(24, name_hint="continuous_weight_k", allow_early_resolve=True):
            for row_block in pl.range(worker, (tokens + 15) // 16, 24):
                row = row_block * 16
                rows = pl.min(16, tokens - row)
                acc = pl.create_tensor([16, 64], dtype=pl.FP32)
                if MODE == 1:
                    for step in pl.range(8):
                        k = step * 512
                        a = pl.slice(x, [16, 512], [row, k], valid_shape=[rows, 512])
                        b = weight[:, k : k + 512]
                        acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=step == 0)
                else:
                    for step in pl.range(16):
                        block = step
                        if MODE >= 3:
                            direction = step
                            if row_block % 2 == 1:
                                direction = 15 - step
                            block = ((row_block % 14) // 2 * 2 + direction) % 16
                        k = pl.max(block, 0) * 256
                        a256 = pl.slice(x, [16, 256], [row, k], valid_shape=[rows, 256])
                        b256 = weight[:, k : k + 256]
                        acc = pl.matmul_acc(acc, a256, b256, b_trans=True, init_cond=step == 0)
                projected = pl.assemble(projected, acc, [row, 0])
        for writer in pl.spmd(48, name_hint="same_weight_bf16_boundaries", allow_early_resolve=True):
            for row_block in pl.range(writer, (tokens + 15) // 16, 48):
                row = row_block * 16
                raw = projected[row : row + 16, :]
                linear = pl.cast(pl.cast(raw, pl.BF16, mode="rint"), pl.FP32)
                value = pl.cast(pl.cast(pl.mul(linear, WEIGHTS_SCALE), pl.BF16, mode="rint"), pl.FP32)
                scaled[row : row + 16, :] = value
    return partials, scaled, projected
