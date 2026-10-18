# SPDX-License-Identifier: Apache-2.0
"""覆盖模块数学常量、两个 Tile 经 assemble 返回及外层循环携带。"""

import pypto.language as pl

STATE_ROWS = pl.dynamic('TILE_STATE_DYNAMIC_ROWS')
GROUP_SIZE = 6
PAIR_WIDTH = 1024


@pl.jit.inline(auto_scope=False)
def update_state(
    roots: pl.Tile[[GROUP_SIZE // 2, PAIR_WIDTH], pl.FP32, pl.MemorySpace.Vec],
    segments: pl.Tile[[GROUP_SIZE, PAIR_WIDTH], pl.FP32, pl.MemorySpace.Vec],
    incoming: pl.Tile[[1, PAIR_WIDTH], pl.FP32, pl.MemorySpace.Vec],
) -> tuple[
    pl.Tile[[GROUP_SIZE // 2, PAIR_WIDTH], pl.FP32, pl.MemorySpace.Vec],
    pl.Tile[[GROUP_SIZE, PAIR_WIDTH], pl.FP32, pl.MemorySpace.Vec],
]:
    next_roots = pl.tile.assemble(roots, incoming, [0, 0])
    next_segments = pl.tile.assemble(segments, incoming, [0, 0])
    return next_roots, next_segments


@pl.jit.incore
def state_kernel(
    source: pl.Tensor[[STATE_ROWS, PAIR_WIDTH], pl.FP32],
    roots_out: pl.Out[pl.Tensor[[STATE_ROWS, PAIR_WIDTH], pl.FP32]],
    segments_out: pl.Out[pl.Tensor[[STATE_ROWS, PAIR_WIDTH], pl.FP32]],
):
    rows = pl.tensor.dim(source, 0)
    roots = pl.load(source, [0, 0], [GROUP_SIZE // 2, PAIR_WIDTH], valid_shape=[pl.min(rows, GROUP_SIZE // 2), PAIR_WIDTH])
    segments = pl.load(source, [0, 0], [GROUP_SIZE, PAIR_WIDTH], valid_shape=[pl.min(rows, GROUP_SIZE), PAIR_WIDTH])
    incoming = pl.load(source, [0, 0], [1, PAIR_WIDTH], valid_shape=[pl.min(rows, 1), PAIR_WIDTH])
    for step, (root_state, segment_state) in pl.range(2, init_values=(roots, segments)):
        next_roots, next_segments = update_state(root_state, segment_state, incoming)
        final_roots, final_segments = pl.yield_(next_roots, next_segments)
    pl.store(final_roots, [0, 0], roots_out)
    pl.store(final_segments, [0, 0], segments_out)
    return roots_out, segments_out


@pl.jit
def tile_entry(
    source: pl.Tensor[[STATE_ROWS, PAIR_WIDTH], pl.FP32],
    roots_out: pl.Out[pl.Tensor[[STATE_ROWS, PAIR_WIDTH], pl.FP32]],
    segments_out: pl.Out[pl.Tensor[[STATE_ROWS, PAIR_WIDTH], pl.FP32]],
):
    return state_kernel(source, roots_out, segments_out)
