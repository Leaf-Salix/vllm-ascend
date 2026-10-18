# SPDX-License-Identifier: Apache-2.0
"""使用与正式 CSA 一样的模块级 pl 绑定，保留动态 Tensor 轴。"""

import pypto.language as pl

DYNAMIC_ROWS = pl.dynamic('TILE_PARAM_DYNAMIC_ROWS')


@pl.jit.inline(auto_scope=False)
def keep_tile(tile: pl.Tile[[8, 64], pl.FP32, pl.MemorySpace.Vec]) -> pl.Tile:
    return tile


@pl.jit.incore
def tile_kernel(
    source: pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32],
    output: pl.Out[pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32]],
):
    rows = pl.min(pl.tensor.dim(source, 0), 8)
    tile = pl.load(source, [0, 0], [8, 64], valid_shape=[rows, 64])
    same_tile = keep_tile(tile)
    pl.store(same_tile, [0, 0], output)
    return output


@pl.jit
def tile_entry(
    source: pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32],
    output: pl.Out[pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32]],
):
    return tile_kernel(source, output)
