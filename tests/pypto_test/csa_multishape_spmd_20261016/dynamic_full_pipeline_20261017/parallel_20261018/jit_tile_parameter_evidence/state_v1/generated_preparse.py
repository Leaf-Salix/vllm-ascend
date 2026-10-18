import pypto.language as pl

TILE_STATE_DYNAMIC_ROWS = pl.dynamic("TILE_STATE_DYNAMIC_ROWS")

@pl.program
class _jit_tile_entry:
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def update_state(self, roots: pl.Tile[[6 // 2, 1024], pl.FP32, pl.MemorySpace.Vec], segments: pl.Tile[[6, 1024], pl.FP32, pl.MemorySpace.Vec], incoming: pl.Tile[[1, 1024], pl.FP32, pl.MemorySpace.Vec]) -> tuple[pl.Tile[[6 // 2, 1024], pl.FP32, pl.MemorySpace.Vec], pl.Tile[[6, 1024], pl.FP32, pl.MemorySpace.Vec]]:
        next_roots = pl.tile.assemble(roots, incoming, [0, 0])
        next_segments = pl.tile.assemble(segments, incoming, [0, 0])
        return (next_roots, next_segments)

    @pl.function(type=pl.FunctionType.InCore)
    def state_kernel(self, source: pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32], roots_out: pl.Out[pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32]], segments_out: pl.Out[pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32]]) -> tuple[pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32], pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32]]:
        rows = pl.tensor.dim(source, 0)
        roots = pl.load(source, [0, 0], [6 // 2, 1024], valid_shape=[pl.min(rows, 6 // 2), 1024])
        segments = pl.load(source, [0, 0], [6, 1024], valid_shape=[pl.min(rows, 6), 1024])
        incoming = pl.load(source, [0, 0], [1, 1024], valid_shape=[pl.min(rows, 1), 1024])
        for (step, (root_state, segment_state)) in pl.range(2, init_values=(roots, segments)):
            (next_roots, next_segments) = self.update_state(root_state, segment_state, incoming)
            (final_roots, final_segments) = pl.yield_(next_roots, next_segments)
        pl.store(final_roots, [0, 0], roots_out)
        pl.store(final_segments, [0, 0], segments_out)
        return (roots_out, segments_out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def tile_entry(self, source: pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32], roots_out: pl.Out[pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32]], segments_out: pl.Out[pl.Tensor[[TILE_STATE_DYNAMIC_ROWS, 1024], pl.FP32]]):
        return self.state_kernel(source, roots_out, segments_out)
