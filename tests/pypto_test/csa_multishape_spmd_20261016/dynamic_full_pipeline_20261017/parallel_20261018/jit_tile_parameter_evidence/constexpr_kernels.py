# SPDX-License-Identifier: Apache-2.0
"""延迟注解仅用于暴露同一 helper constexpr 名字的明确未支持边界。"""

from __future__ import annotations

import pypto.language as pl

DYNAMIC_ROWS = pl.dynamic('TILE_CONSTEXPR_DYNAMIC_ROWS')


@pl.jit.inline(auto_scope=False)
def keep_tile(
    tile: pl.Tile[[ROWS, 64], pl.FP32, pl.MemorySpace.Vec],
    ROWS: pl.constexpr,
) -> pl.Tile:
    return tile


@pl.jit.incore
def tile_kernel(
    source: pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32],
    output: pl.Out[pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32]],
):
    tile = pl.load(source, [0, 0], [8, 64], valid_shape=[pl.min(pl.tensor.dim(source, 0), 8), 64])
    same_tile = keep_tile(tile, 8)
    pl.store(same_tile, [0, 0], output)
    return output


@pl.jit
def tile_entry(
    source: pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32],
    output: pl.Out[pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32]],
):
    return tile_kernel(source, output)
