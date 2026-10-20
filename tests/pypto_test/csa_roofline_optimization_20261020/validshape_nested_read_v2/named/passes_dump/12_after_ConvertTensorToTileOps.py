# pypto.program: _jit_entry_named
import pypto.language as pl


@pl.program
class _jit_entry_named:
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def calculate_named(src__ssa_v0: pl.Tensor[[1, 2048], pl.FP32], valid__ssa_v0: pl.Tensor[[1], pl.INT32], out__ssa_v0: pl.Out[pl.Tensor[[1, 2048], pl.FP32]]) -> pl.Tensor[[1, 2048], pl.FP32]:
        tile__ssa_v0: pl.Tile[[1, 2048], pl.FP32] = pl.tile.load(src__ssa_v0, [0, 0], [1, 2048], [1, 2048])
        n__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid__ssa_v0, [0])
        n_index__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(n__tile, pl.INDEX)
        narrowed__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.TileView(valid_shape=[1, n_index__ssa_v0])] = pl.tile.set_validshape(tile__ssa_v0, 1, n_index__ssa_v0)
        result__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, n_index__ssa_v0])] = pl.tile.adds(narrowed__ssa_v0, 1.0)
        pl.tile.store(result__ssa_v0, [0, 0], out__ssa_v0)
        return out__ssa_v0

    @pl.function(type=pl.FunctionType.Spmd)
    def entry_named_spmd_0(
        self, src__ssa_v0: pl.Tensor[[1, 2048], pl.FP32], valid__ssa_v0: pl.Tensor[[1], pl.INT32], out__ssa_v0: pl.Out[pl.Tensor[[1, 2048], pl.FP32]]
    ) -> pl.Tensor[[1, 2048], pl.FP32]:
        out__ssa_v1: pl.Tensor[[1, 2048], pl.FP32] = self.calculate_named(src__ssa_v0, valid__ssa_v0, out__ssa_v0)
        return out__ssa_v0

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def entry_named(self, src__ssa_v0: pl.Tensor[[1, 2048], pl.FP32], valid__ssa_v0: pl.Tensor[[1], pl.INT32], out__ssa_v0: pl.Out[pl.Tensor[[1, 2048], pl.FP32]]) -> pl.Tensor[[1, 2048], pl.FP32]:
        out__ssa_v1: pl.Tensor[[1, 2048], pl.FP32] = self.entry_named_spmd_0(src__ssa_v0, valid__ssa_v0, out__ssa_v0, attrs={"core_num": 1})
        return out__ssa_v1
