import pypto.language as pl

TILE_MAT_LEFT_DYNAMIC_ROWS = pl.dynamic("TILE_MAT_LEFT_DYNAMIC_ROWS")

@pl.program
class _jit_tile_entry:
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def one_step(self, source: pl.Tensor[[TILE_MAT_LEFT_DYNAMIC_ROWS, 64], pl.FP16], output: pl.Tensor[[TILE_MAT_LEFT_DYNAMIC_ROWS, 64], pl.FP32], step: pl.Scalar[pl.INDEX], panel: pl.Tile[[384, 64], pl.FP16, pl.MemorySpace.Mat], coefficient: pl.Tile[[16, 384], pl.FP16, pl.MemorySpace.Left]) -> tuple[pl.Tile[[384, 64], pl.FP16, pl.MemorySpace.Mat], pl.Tile[[16, 384], pl.FP16, pl.MemorySpace.Left], pl.Tensor[[TILE_MAT_LEFT_DYNAMIC_ROWS, 64], pl.FP32]]:
        panel_v1 = pl.gather_row(panel, source, [0, 0], [step * 384, 0], [384, 64])
        right = pl.tile.move(panel_v1, target_memory=pl.MemorySpace.Right)
        acc = pl.tile.matmul(coefficient, right)
        pl.store(acc, [step * 16, 0], output)
        return (panel_v1, coefficient, output)

    @pl.function(type=pl.FunctionType.InCore)
    def mat_left_kernel(self, source: pl.Tensor[[TILE_MAT_LEFT_DYNAMIC_ROWS, 64], pl.FP16], weights: pl.Tensor[[16, 384], pl.FP16], output: pl.InOut[pl.Tensor[[TILE_MAT_LEFT_DYNAMIC_ROWS, 64], pl.FP32]]) -> pl.Tensor[[TILE_MAT_LEFT_DYNAMIC_ROWS, 64], pl.FP32]:
        rows = pl.tensor.dim(source, 0)
        panel = pl.create_tile([384, 64], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
        weight_mat = pl.load(weights, [0, 0], [16, 384], target_memory=pl.MemorySpace.Mat)
        coefficient = pl.tile.move(weight_mat, target_memory=pl.MemorySpace.Left)
        for (step, (panel_state, coefficient_state)) in pl.range(pl.min(rows // 384, 2), init_values=(panel, coefficient)):
            (next_panel, next_coefficient, output) = self.one_step(source, output, step, panel_state, coefficient_state)
            (final_panel, final_coefficient) = pl.yield_(next_panel, next_coefficient)
        return output

    @pl.function(type=pl.FunctionType.Orchestration)
    def tile_entry(self, source: pl.Tensor[[TILE_MAT_LEFT_DYNAMIC_ROWS, 64], pl.FP16], weights: pl.Tensor[[16, 384], pl.FP16], output: pl.InOut[pl.Tensor[[TILE_MAT_LEFT_DYNAMIC_ROWS, 64], pl.FP32]]):
        return self.mat_left_kernel(source, weights, output)
