# SPDX-License-Identifier: Apache-2.0
"""固定G6数学Mat/Left状态的最小编译检查；动态T只决定安全GM访问和循环。"""

import pypto.language as pl

ROWS = pl.dynamic("TILE_MAT_LEFT_DYNAMIC_ROWS")
HEAD_ROWS = 384
PANEL_COLS = 64
WEIGHT_ROWS = 16


@pl.jit.inline(auto_scope=False)
def one_step(
    source: pl.Tensor[[ROWS, PANEL_COLS], pl.FP16],
    output: pl.InOut[pl.Tensor[[ROWS, PANEL_COLS], pl.FP32]],
    step: pl.Scalar[pl.INDEX],
    panel: pl.Tile[[HEAD_ROWS, PANEL_COLS], pl.FP16, pl.MemorySpace.Mat],
    coefficient: pl.Tile[[WEIGHT_ROWS, HEAD_ROWS], pl.FP16, pl.MemorySpace.Left],
) -> tuple[
    pl.Tile[[HEAD_ROWS, PANEL_COLS], pl.FP16, pl.MemorySpace.Mat],
    pl.Tile[[WEIGHT_ROWS, HEAD_ROWS], pl.FP16, pl.MemorySpace.Left],
    pl.Tensor[[ROWS, PANEL_COLS], pl.FP32],
]:
    panel = pl.gather_row(panel, source, [0, 0], [step * HEAD_ROWS, 0], [HEAD_ROWS, PANEL_COLS])
    right = pl.tile.move(panel, target_memory=pl.MemorySpace.Right)
    acc = pl.tile.matmul(coefficient, right)
    pl.store(acc, [step * WEIGHT_ROWS, 0], output)
    return panel, coefficient, output


@pl.jit.incore
def mat_left_kernel(
    source: pl.Tensor[[ROWS, PANEL_COLS], pl.FP16],
    weights: pl.Tensor[[WEIGHT_ROWS, HEAD_ROWS], pl.FP16],
    output: pl.InOut[pl.Tensor[[ROWS, PANEL_COLS], pl.FP32]],
):
    rows = pl.tensor.dim(source, 0)
    panel = pl.create_tile([HEAD_ROWS, PANEL_COLS], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
    weight_mat = pl.load(weights, [0, 0], [WEIGHT_ROWS, HEAD_ROWS], target_memory=pl.MemorySpace.Mat)
    coefficient = pl.tile.move(weight_mat, target_memory=pl.MemorySpace.Left)
    for step, (panel_state, coefficient_state, out_state) in pl.range(
        pl.min(rows // HEAD_ROWS, 2), init_values=(panel, coefficient, output)
    ):
        next_panel, next_coefficient, next_output = one_step(source, out_state, step, panel_state, coefficient_state)
        final_panel, final_coefficient, final_output = pl.yield_(next_panel, next_coefficient, next_output)
    return final_output


@pl.jit
def tile_entry(
    source: pl.Tensor[[ROWS, PANEL_COLS], pl.FP16],
    weights: pl.Tensor[[WEIGHT_ROWS, HEAD_ROWS], pl.FP16],
    output: pl.InOut[pl.Tensor[[ROWS, PANEL_COLS], pl.FP32]],
):
    return mat_left_kernel(source, weights, output)
