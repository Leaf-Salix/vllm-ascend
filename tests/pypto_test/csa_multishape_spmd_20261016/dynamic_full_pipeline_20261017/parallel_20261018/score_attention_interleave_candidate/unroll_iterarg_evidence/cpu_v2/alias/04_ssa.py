# pypto.program: _jit_alias_entry
import pypto.language as pl

LOOP_SCOPE_DYNAMIC_ROWS = pl.dynamic("LOOP_SCOPE_DYNAMIC_ROWS")


@pl.program
class _jit_alias_entry:
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def alias_kernel(output__ssa_v0: pl.InOut[pl.Tensor[[LOOP_SCOPE_DYNAMIC_ROWS, 64], pl.FP32]]) -> pl.Tensor[[LOOP_SCOPE_DYNAMIC_ROWS, 64], pl.FP32]:
        rows__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(output__ssa_v0, 0)
        state__ssa_v0: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([2, 64], dtype=pl.FP32, value=0.0)
        row__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=1.0)
        initial_step__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX)
        initial_live__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(rows__ssa_v0, 1)
        for tick__idx_v0, (state_iter, step_iter, live_iter) in pl.range(pl.min(rows__ssa_v0, 2), init_values=(state__ssa_v0, initial_step__ssa_v0, initial_live__ssa_v0)):
            step_arg__ssa_v0: pl.Scalar[pl.INDEX] = step_iter
            live_arg__ssa_v0: pl.Scalar[pl.INDEX] = live_iter
            state_local_inline3__ssa_v0: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = state_iter
            state_local_inline3__ssa_v1: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(state_local_inline3__ssa_v0, row__ssa_v0, [(step_arg__ssa_v0 + 0 + live_arg__ssa_v0) % 2, 0])
            state_local_inline3__ssa_v2: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(state_local_inline3__ssa_v1, row__ssa_v0, [(step_arg__ssa_v0 + 1 + live_arg__ssa_v0) % 2, 0])
            next_state__ssa_v0: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = state_local_inline3__ssa_v2
            final_state, final_step, final_live = pl.yield_(next_state__ssa_v0, step_iter + 1, live_iter)
        valid_state__ssa_v0: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(rows__ssa_v0, pl.const(2, pl.INDEX)), 64])] = pl.tile.set_validshape(
            final_state, pl.min(rows__ssa_v0, 2), 64
        )
        output_v1__ssa_v0: pl.Tensor[[LOOP_SCOPE_DYNAMIC_ROWS, 64], pl.FP32] = pl.tile.store(valid_state__ssa_v0, [0, 0], output__ssa_v0)
        return output_v1__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def alias_entry(self, output__ssa_v0: pl.InOut[pl.Tensor[[LOOP_SCOPE_DYNAMIC_ROWS, 64], pl.FP32]]):
        return self.alias_kernel(output__ssa_v0)