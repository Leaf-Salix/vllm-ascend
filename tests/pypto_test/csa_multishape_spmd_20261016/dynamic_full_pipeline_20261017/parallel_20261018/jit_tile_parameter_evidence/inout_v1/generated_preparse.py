import pypto.language as pl

TILE_INOUT_DYNAMIC_ROWS = pl.dynamic("TILE_INOUT_DYNAMIC_ROWS")

@pl.program
class _jit_tile_entry:
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def keep_tile(self, tile: pl.InOut[pl.Tile[[8, 64], pl.FP32, pl.MemorySpace.Vec]]):
        return tile

    @pl.function(type=pl.FunctionType.InCore)
    def tile_kernel(self, source: pl.Tensor[[TILE_INOUT_DYNAMIC_ROWS, 64], pl.FP32], output: pl.Out[pl.Tensor[[TILE_INOUT_DYNAMIC_ROWS, 64], pl.FP32]]) -> pl.Tensor[[TILE_INOUT_DYNAMIC_ROWS, 64], pl.FP32]:
        tile = pl.load(source, [0, 0], [8, 64], valid_shape=[pl.min(pl.tensor.dim(source, 0), 8), 64])
        same_tile = self.keep_tile(tile)
        pl.store(same_tile, [0, 0], output)
        return output

    @pl.function(type=pl.FunctionType.Orchestration)
    def tile_entry(self, source: pl.Tensor[[TILE_INOUT_DYNAMIC_ROWS, 64], pl.FP32], output: pl.Out[pl.Tensor[[TILE_INOUT_DYNAMIC_ROWS, 64], pl.FP32]]):
        return self.tile_kernel(source, output)
