# SPDX-License-Identifier: Apache-2.0
"""实际 full-valid 状态：动态 T 只决定访问范围，不作为片上数学 Tile 形状。"""

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
    roots = pl.tile.full([GROUP_SIZE // 2, PAIR_WIDTH], dtype=pl.FP32, value=0.0)
    segments = pl.tile.full([GROUP_SIZE, PAIR_WIDTH], dtype=pl.FP32, value=0.0)
    for step, (root_state, segment_state) in pl.range(pl.min(rows, 2), init_values=(roots, segments)):
        incoming = pl.load(source, [step, 0], [1, PAIR_WIDTH])
        next_roots, next_segments = update_state(root_state, segment_state, incoming)
        final_roots, final_segments = pl.yield_(next_roots, next_segments)
    pl.store(pl.set_validshape(final_roots, pl.min(rows, GROUP_SIZE // 2), PAIR_WIDTH), [0, 0], roots_out)
    pl.store(pl.set_validshape(final_segments, pl.min(rows, GROUP_SIZE), PAIR_WIDTH), [0, 0], segments_out)
    return roots_out, segments_out


@pl.jit
def tile_entry(
    source: pl.Tensor[[STATE_ROWS, PAIR_WIDTH], pl.FP32],
    roots_out: pl.Out[pl.Tensor[[STATE_ROWS, PAIR_WIDTH], pl.FP32]],
    segments_out: pl.Out[pl.Tensor[[STATE_ROWS, PAIR_WIDTH], pl.FP32]],
):
    return state_kernel(source, roots_out, segments_out)
