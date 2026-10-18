# SPDX-License-Identifier: Apache-2.0
"""只隔离动态range的外层IterArg经过inline/unroll后的作用域，不调用设备。"""

import pypto.language as pl

ROWS = pl.dynamic("LOOP_SCOPE_DYNAMIC_ROWS")


@pl.jit.inline(auto_scope=False)
def seed(rows: pl.Scalar[pl.INDEX]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]]:
    return pl.cast(0, pl.INDEX), pl.min(rows, 1)


@pl.jit.inline(auto_scope=False)
def update(
    state: pl.Tile[[2, 64], pl.FP32, pl.MemorySpace.Vec],
    row: pl.Tile[[1, 64], pl.FP32, pl.MemorySpace.Vec],
    step: pl.Scalar[pl.INDEX],
    live: pl.Scalar[pl.INDEX],
) -> pl.Tile[[2, 64], pl.FP32, pl.MemorySpace.Vec]:
    # 避免已独立定位的Tile形参重绑定限制；两侧完全相同。
    state_local = state
    for lane in pl.unroll(2):
        state_local = pl.tile.assemble(state_local, row, [(step + lane + live) % 2, 0])
    return state_local


@pl.jit.incore
def direct_kernel(output: pl.InOut[pl.Tensor[[ROWS, 64], pl.FP32]]):
    rows = pl.tensor.dim(output, 0)
    state = pl.tile.full([2, 64], dtype=pl.FP32, value=0.0)
    row = pl.tile.full([1, 64], dtype=pl.FP32, value=1.0)
    initial_step, initial_live = seed(rows)
    for tick, (state_iter, step_iter, live_iter) in pl.range(
        pl.min(rows, 2), init_values=(state, initial_step, initial_live)
    ):
        next_state = update(state_iter, row, step_iter, live_iter)
        final_state, final_step, final_live = pl.yield_(next_state, step_iter + 1, live_iter)
    valid_state = pl.tile.set_validshape(final_state, [pl.min(rows, 2), 64])
    output = pl.store(valid_state, [0, 0], output)
    return output


@pl.jit.incore
def alias_kernel(output: pl.InOut[pl.Tensor[[ROWS, 64], pl.FP32]]):
    rows = pl.tensor.dim(output, 0)
    state = pl.tile.full([2, 64], dtype=pl.FP32, value=0.0)
    row = pl.tile.full([1, 64], dtype=pl.FP32, value=1.0)
    initial_step, initial_live = seed(rows)
    for tick, (state_iter, step_iter, live_iter) in pl.range(
        pl.min(rows, 2), init_values=(state, initial_step, initial_live)
    ):
        # 普通局部Var桥必须在含unroll的helper调用之前，不能放到unroll内部。
        step_arg: pl.Scalar[pl.INDEX] = step_iter
        live_arg: pl.Scalar[pl.INDEX] = live_iter
        next_state = update(state_iter, row, step_arg, live_arg)
        final_state, final_step, final_live = pl.yield_(next_state, step_iter + 1, live_iter)
    valid_state = pl.tile.set_validshape(final_state, [pl.min(rows, 2), 64])
    output = pl.store(valid_state, [0, 0], output)
    return output


@pl.jit
def direct_entry(output: pl.InOut[pl.Tensor[[ROWS, 64], pl.FP32]]):
    return direct_kernel(output)


@pl.jit
def alias_entry(output: pl.InOut[pl.Tensor[[ROWS, 64], pl.FP32]]):
    return alias_kernel(output)
