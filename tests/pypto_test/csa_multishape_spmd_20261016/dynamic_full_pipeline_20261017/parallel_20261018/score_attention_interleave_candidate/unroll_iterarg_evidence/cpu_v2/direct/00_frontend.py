# pypto.program: _jit_direct_entry
import pypto.language as pl

LOOP_SCOPE_DYNAMIC_ROWS = pl.dynamic("LOOP_SCOPE_DYNAMIC_ROWS")


@pl.program
class _jit_direct_entry:
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def seed(self, rows: pl.Scalar[pl.INDEX]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]]:
        return pl.cast(0, pl.INDEX), pl.min(rows, 1)

    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def update(self, state: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec], row: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec], step: pl.Scalar[pl.INDEX], live: pl.Scalar[pl.INDEX]):
        state_local: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = state
        for lane in pl.unroll(2):
            state_local: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(state_local, row, [(step + lane + live) % 2, 0])
        return state_local

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def direct_kernel(output: pl.InOut[pl.Tensor[[LOOP_SCOPE_DYNAMIC_ROWS, 64], pl.FP32]]) -> pl.Tensor[[LOOP_SCOPE_DYNAMIC_ROWS, 64], pl.FP32]:
        rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(output, 0)
        state: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([2, 64], dtype=pl.FP32, value=0.0)
        row: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=1.0)
        _tuple_tmp: pl.Tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]] = self.seed(rows)
        initial_step: pl.Scalar[pl.INDEX] = _tuple_tmp[0]
        initial_live: pl.Scalar[pl.INDEX] = _tuple_tmp[1]
        for tick, (state_iter, step_iter, live_iter) in pl.range(pl.min(rows, 2), init_values=(state, initial_step, initial_live)):
            next_state: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec] = self.update(state_iter, row, step_iter, live_iter)
            final_state, final_step, final_live = pl.yield_(next_state, step_iter + 1, live_iter)
        valid_state: pl.Tile[[2, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[pl.min(rows, pl.const(2, pl.INDEX)), 64])] = pl.tile.set_validshape(final_state, pl.min(rows, 2), 64)
        output_v1: pl.Tensor[[LOOP_SCOPE_DYNAMIC_ROWS, 64], pl.FP32] = pl.tile.store(valid_state, [0, 0], output)
        return output_v1

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def direct_entry(self, output: pl.InOut[pl.Tensor[[LOOP_SCOPE_DYNAMIC_ROWS, 64], pl.FP32]]):
        return self.direct_kernel(output)