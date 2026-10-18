import pypto.language as pl
from pypto import backend, passes
from pypto.backend import BackendType
backend.reset_for_testing()
backend.set_backend_type(BackendType.Ascend910B)

first_rows = 192
total_rows = first_rows + 48 + 256

@pl.program
class PhaseArena:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(
        self,
        x: pl.Tensor[[total_rows, 128], pl.FP16],
        w: pl.Tensor[[128, 32], pl.FP16],
        out: pl.Out[pl.Tensor[[total_rows, 32], pl.FP32]],
    ) -> pl.Tensor[[total_rows, 32], pl.FP32]:
        weight = pl.tile.load(w, [0, 0], [128, 32], target_memory=pl.Mem.Mat)
        right = pl.tile.move(weight, target_memory=pl.Mem.Right)
        a_mat = pl.tile.load(x, [0, 0], [first_rows, 128], target_memory=pl.Mem.Mat)
        b_mat = pl.tile.load(x, [first_rows, 0], [48, 128], target_memory=pl.Mem.Mat)
        a = pl.tile.move(a_mat, target_memory=pl.Mem.Left)
        b = pl.tile.move(b_mat, target_memory=pl.Mem.Left)
        a_out = pl.tile.matmul(a, right)
        out_a = pl.tile.store(a_out, [0, 0], out)
        b_out = pl.tile.matmul(b, right)
        out_b = pl.tile.store(b_out, [first_rows, 0], out_a)
        pl.system.fence()
        c_mat = pl.tile.load(x, [first_rows + 48, 0], [128, 128], target_memory=pl.Mem.Mat)
        d_mat = pl.tile.load(x, [first_rows + 176, 0], [128, 128], target_memory=pl.Mem.Mat)
        c = pl.tile.move(c_mat, target_memory=pl.Mem.Left)
        d = pl.tile.move(d_mat, target_memory=pl.Mem.Left)
        c_out = pl.tile.matmul(c, right)
        out_c = pl.tile.store(c_out, [first_rows + 48, 0], out_b)
        d_out = pl.tile.matmul(d, right)
        out_d = pl.tile.store(d_out, [first_rows + 176, 0], out_c)
        return out_d

p = passes.init_mem_ref()(PhaseArena)
p = passes.materialize_semantic_aliases()(p)
p = passes.memory_reuse()(p)
passes.allocate_memory_addr()(p)
