# pypto.program: _jit_tile_entry
import pypto.language as pl

TILE_INOUT_DYNAMIC_ROWS = pl.dynamic("TILE_INOUT_DYNAMIC_ROWS")


@pl.program
class _jit_tile_entry:
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def keep_tile(self, tile: pl.InOut[pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec]]):
        return tile

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def tile_kernel(source: pl.Tensor[[TILE_INOUT_DYNAMIC_ROWS, 64], pl.FP32], output: pl.Out[pl.Tensor[[TILE_INOUT_DYNAMIC_ROWS, 64], pl.FP32]]) -> pl.Tensor[[TILE_INOUT_DYNAMIC_ROWS, 64], pl.FP32]:
        tile: pl.Tile[[8, 64], pl.FP32, pl.TileView(valid_shape=[pl.min(pl.tensor.dim(source, pl.const(0, pl.INDEX)), pl.const(8, pl.INDEX)), 64])] = pl.tile.load(
            source, [0, 0], [8, 64], [pl.min(pl.tensor.dim(source, 0), 8), 64]
        )
        same_tile: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = self.keep_tile(tile)
        pl.tile.store(same_tile, [0, 0], output)
        return output

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def tile_entry(self, source: pl.Tensor[[TILE_INOUT_DYNAMIC_ROWS, 64], pl.FP32], output: pl.Out[pl.Tensor[[TILE_INOUT_DYNAMIC_ROWS, 64], pl.FP32]]):
        return self.tile_kernel(source, output)