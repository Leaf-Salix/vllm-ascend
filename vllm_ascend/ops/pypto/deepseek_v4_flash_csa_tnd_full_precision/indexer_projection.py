# SPDX-License-Identifier: Apache-2.0
"""独立Indexer投影候选：FP32点积、BF16发布、BF16缩放发布、FP16输入。"""

import pypto.language as pl

T = pl.dynamic("INDEXER_PROJ_T")


@pl.jit
def indexer_projection(
    x: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[64, 4096], pl.BF16],
    order: pl.Tensor[[16, 4, 16], pl.INT32],
    raw: pl.Out[pl.Tensor[[T, 64], pl.BF16]],
    scaled: pl.Out[pl.Tensor[[T, 64], pl.BF16]],
    output: pl.Out[pl.Tensor[[T, 64], pl.FP16]],
):
    x.bind_dynamic(0, T)
    raw.bind_dynamic(0, T)
    scaled.bind_dynamic(0, T)
    output.bind_dynamic(0, T)
    rows = pl.tensor.dim(x, 0)
    row_tiles = (rows + 15) // 16
    accum = pl.create_tensor([row_tiles * 16, 64], dtype=pl.FP32)
    for worker in pl.spmd(row_tiles * 4, name_hint="indexer_projection_matmul"):
        rb = worker // 4
        group = worker % 4
        row = rb * 16
        col = group * 16
        valid = pl.min(16, rows - row)
        tile = pl.create_tensor([16, 16], dtype=pl.FP32)
        for step in pl.range(16):
            k_block = pl.read(order, [rb, group, step])
            offset = pl.max(k_block, 0) * 256
            a = pl.slice(x, [16, 256], [row, offset], valid_shape=[valid, 256])
            b = weight[col : col + 16, offset : offset + 256]
            tile = pl.matmul_acc(tile, a, b, b_trans=True, init_cond=(step == 0))
        accum[row : row + 16, col : col + 16] = tile
    for block in pl.spmd(row_tiles, name_hint="indexer_projection_round"):
        row = block * 16
        valid = pl.min(16, rows - row)
        value = pl.load(accum, [row, 0], [16, 64], valid_shape=[valid, 64])
        projected = pl.cast(value, pl.BF16, mode="rint")
        pl.store(projected, [row, 0], raw)
        value = pl.cast(projected, pl.FP32)
        value = pl.mul(value, 0.011048543456039806)
        weighted = pl.cast(value, pl.BF16, mode="rint")
        pl.store(weighted, [row, 0], scaled)
        result = pl.cast(weighted, pl.FP16, mode="rint")
        pl.store(result, [row, 0], output)
    return raw, scaled, output
