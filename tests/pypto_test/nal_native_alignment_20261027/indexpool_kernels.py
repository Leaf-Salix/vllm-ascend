# SPDX-License-Identifier: Apache-2.0
"""固定实际八行窗口，仅改变Indexer pool；Norm/RoPE保持原算术。"""

import pypto.language as pl

WINDOW_ROWS = pl.dynamic("INDEX_POOL_WINDOW_ROWS")
ROWS = pl.dynamic("INDEX_POOL_RESULT_ROWS")


@pl.jit
def indexpool_arm(
    values: pl.Tensor[[WINDOW_ROWS, 128], pl.FP32],
    scores: pl.Tensor[[WINDOW_ROWS, 128], pl.FP32],
    gamma: pl.Tensor[[128], pl.BF16],
    cos: pl.Tensor[[ROWS, 64], pl.FP32],
    sin: pl.Tensor[[ROWS, 64], pl.FP32],
    pooled: pl.InOut[pl.Tensor[[ROWS, 128], pl.FP32]],
    normed: pl.InOut[pl.Tensor[[ROWS, 128], pl.BF16]],
    MODE: pl.constexpr,
):
    for worker in pl.spmd(48, name_hint="index_pool_same_window", allow_early_resolve=True):
        for row in pl.range(worker, pl.tensor.dim(pooled, 0), 48):
            for h0 in pl.range(0, 128, 64):
                if MODE == 0:
                    maximum = pl.load(scores, [row * 8 + 7, h0], [1, 64])
                    total = pl.exp(pl.sub(maximum, maximum))
                    weighted = pl.load(values, [row * 8 + 7, h0], [1, 64])
                    for index in pl.range(7):
                        value = pl.load(values, [row * 8 + index, h0], [1, 64])
                        score = pl.load(scores, [row * 8 + index, h0], [1, 64])
                        next_maximum = pl.maximum(maximum, score)
                        alpha = pl.exp(pl.sub(maximum, next_maximum))
                        beta = pl.exp(pl.sub(score, next_maximum))
                        total = pl.add(pl.mul(alpha, total), beta)
                        weighted = pl.add(pl.mul(weighted, alpha), pl.mul(value, beta))
                        maximum = next_maximum
                    result = pl.div(weighted, total)
                else:
                    window_values = pl.tile.full([8, 64], dtype=pl.FP32, value=0.0)
                    window_scores = pl.tile.full([8, 64], dtype=pl.FP32, value=0.0)
                    for index in pl.range(8):
                        native_row = index % 4 * 2 + index // 4
                        value = pl.load(values, [row * 8 + index, h0], [1, 64])
                        score = pl.load(scores, [row * 8 + index, h0], [1, 64])
                        window_values = pl.tile.assemble(window_values, value, [native_row, 0])
                        window_scores = pl.tile.assemble(window_scores, score, [native_row, 0])
                    max4 = pl.maximum(window_scores[:4, :], window_scores[4:8, :])
                    max2 = pl.maximum(max4[:2, :], max4[2:4, :])
                    maximum = pl.maximum(max2[:1, :], max2[1:2, :])
                    probability = pl.exp(pl.col_expand_sub(window_scores, maximum))
                    sum4 = pl.add(probability[:4, :], probability[4:8, :])
                    sum2 = pl.add(sum4[:2, :], sum4[2:4, :])
                    total = pl.add(sum2[:1, :], sum2[1:2, :])
                    probability = pl.col_expand_div(probability, total)
                    tree_weighted = pl.mul(window_values, probability)
                    weighted4 = pl.add(tree_weighted[:4, :], tree_weighted[4:8, :])
                    weighted2 = pl.add(weighted4[:2, :], weighted4[2:4, :])
                    result = pl.add(weighted2[:1, :], weighted2[1:2, :])
                pl.store(result, [row, h0], pooled)
    for worker in pl.spmd(48, name_hint="unchanged_index_norm_rope", allow_early_resolve=True):
        for row in pl.range(worker * 8, pl.tensor.dim(pooled, 0), 48 * 8):
            rows = pl.min(8, pl.tensor.dim(pooled, 0) - row)
            low = pl.load(pooled, [row, 0], [8, 64], valid_shape=[rows, 64])
            high = pl.load(pooled, [row, 64], [8, 64], valid_shape=[rows, 64])
            square = pl.add(pl.mul(low, low), pl.mul(high, high))
            square_sum = pl.row_sum(square, pl.create_tile([8, 64], dtype=pl.FP32))
            inverse = pl.recip(pl.sqrt(pl.add(pl.mul(square_sum, 1.0 / 128.0), 1e-6)))
            low_weight = pl.cast(pl.reshape(pl.load(gamma, [0], [64]), [1, 64]), pl.FP32)
            high_weight = pl.cast(pl.reshape(pl.load(gamma, [64], [64]), [1, 64]), pl.FP32)
            low = pl.col_expand_mul(pl.row_expand_mul(low, inverse), low_weight)
            high = pl.col_expand_mul(pl.row_expand_mul(high, inverse), high_weight)
            ones = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
            column = pl.col_expand_mul(ones, pl.cast(pl.tile.arange(0, [1, 64], dtype=pl.INT32), pl.FP32))
            pair = pl.cast(pl.cast(pl.mul(column, 0.5), pl.INT32, mode="trunc"), pl.FP32)
            lane = pl.sub(column, pl.mul(pair, 2.0))
            swap = pl.sub(pl.add(column, 1.0), pl.mul(lane, 2.0))
            row_offsets = pl.reshape(pl.tile.muls(pl.tile.ci(0, [1, 8], dtype=pl.INT32), 64), [8, 1])
            indices = pl.set_validshape(pl.row_expand_add(pl.cast(swap, pl.INT32), row_offsets), rows, 64)
            swapped = pl.tile.gather(high, indices, pl.create_tile([8, 64], dtype=pl.INT32))
            cos_tile = pl.load(cos, [row, 0], [8, 64], valid_shape=[rows, 64])
            sin_tile = pl.load(sin, [row, 0], [8, 64], valid_shape=[rows, 64])
            signed = pl.mul(sin_tile, pl.sub(pl.mul(lane, 2.0), 1.0))
            rope = pl.add(pl.mul(high, cos_tile), pl.mul(swapped, signed))
            pl.store(pl.cast(low, pl.BF16, mode="rint"), [row, 0], normed)
            pl.store(pl.cast(rope, pl.BF16, mode="rint"), [row, 64], normed)
    return pooled, normed
