# pypto.program: _jit_tile_entry
import pypto.language as pl

TILE_STATE_DYNAMIC_ROWS = pl.dynamic("TILE_STATE_DYNAMIC_ROWS")


@pl.program
class _jit_tile_entry:
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def update_state(
        self, roots: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec], segments: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec], incoming: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec]
    ) -> tuple[pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec], pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec]]:
        next_roots: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(roots, incoming, [0, 0])
        next_segments: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(segments, incoming, [0, 0])
        return next_roots, next_segments

    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def state_kernel(
        source: pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32],
        roots_out: pl.Out[pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32]],
        segments_out: pl.Out[pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32]],
    ) -> tuple[pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32], pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32]]:
        rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(source, 0)
        roots: pl.Tile[[3, 1024], pl.FP32, pl.TileView(valid_shape=[pl.min(rows, pl.const(3, pl.INDEX)), 1024])] = pl.tile.load(source, [0, 0], [3, 1024], [pl.min(rows, 3), 1024])
        segments: pl.Tile[[6, 1024], pl.FP32, pl.TileView(valid_shape=[pl.min(rows, pl.const(6, pl.INDEX)), 1024])] = pl.tile.load(source, [0, 0], [6, 1024], [pl.min(rows, 6), 1024])
        incoming: pl.Tile[[1, 1024], pl.FP32, pl.TileView(valid_shape=[pl.min(rows, pl.const(1, pl.INDEX)), 1024])] = pl.tile.load(source, [0, 0], [1, 1024], [pl.min(rows, 1), 1024])
        for step, (root_state, segment_state) in pl.range(2, init_values=(roots, segments)):
            _tuple_tmp: pl.Tuple[pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec], pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec]] = self.update_state(root_state, segment_state, incoming)
            next_roots: pl.Tile[[3, 1024], pl.FP32, pl.Mem.Vec] = _tuple_tmp[0]
            next_segments: pl.Tile[[6, 1024], pl.FP32, pl.Mem.Vec] = _tuple_tmp[1]
            final_roots, final_segments = pl.yield_(next_roots, next_segments)
        pl.tile.store(final_roots, [0, 0], roots_out)
        pl.tile.store(final_segments, [0, 0], segments_out)
        return roots_out, segments_out

    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def tile_entry(
        self,
        source: pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32],
        roots_out: pl.Out[pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32]],
        segments_out: pl.Out[pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32]],
    ):
        return self.state_kernel(source, roots_out, segments_out)
