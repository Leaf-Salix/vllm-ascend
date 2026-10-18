import pypto.language as pl
from pypto import backend, passes
from pypto.backend import BackendType
backend.reset_for_testing()
backend.set_backend_type(BackendType.Ascend910B)

@pl.program
class BranchGather:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self,
        pool: pl.Tensor[[64, 512], pl.BF16],
        flags: pl.Tensor[[1], pl.INT32],
        out: pl.Out[pl.Tensor[[64, 512], pl.BF16]],
        state: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
    ) -> tuple[pl.Tensor[[64, 512], pl.BF16], pl.Tensor[[64, 256], pl.FP32]]:
        first = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        second = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        flag = pl.tensor.read(flags, [0])
        for i, (s0, s1) in pl.range(2, init_values=(first, second)):
            seed = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
            if flag != 0:
                filled = pl.tile.gather_row(seed, pool, [0, 0], [0, 0], [64, 512])
                selected = filled
            else:
                fresh = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                for j, (carry,) in pl.range(64, init_values=(fresh,)):
                    if j < flag:
                        next_value = pl.tile.gather_row(
                            carry, pool, [j, 0], [j, 0], [1, 512]
                        )
                    else:
                        next_value = carry
                    gathered = pl.yield_(next_value)
                selected = gathered
            saved = pl.tile.store(selected, [0, 0], out)  # noqa: F841 - observe the selected tile
            next0 = pl.tile.add(s0, 1.0)
            next1 = pl.tile.add(s1, 2.0)
            result0, result1 = pl.yield_(next0, next1)
        state0 = pl.tile.store(result0, [0, 0], state)
        state1 = pl.tile.store(result1, [32, 0], state0)
        return out, state1

@pl.program
class ConditionalReset:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self,
        pool: pl.Tensor[[128, 256], pl.FP32],
        flags: pl.Tensor[[1], pl.INT32],
        out: pl.Out[pl.Tensor[[128, 256], pl.FP32]],
    ) -> pl.Tensor[[128, 256], pl.FP32]:
        held = pl.tile.load(pool, [64, 0], [64, 256], target_memory=pl.Mem.Vec)
        current = pl.tile.load(pool, [0, 0], [64, 256], target_memory=pl.Mem.Vec)
        flag = pl.tensor.read(flags, [0])
        if flag != 0:
            reset = pl.tile.full([64, 256], dtype=pl.FP32, value=0.0)
            selected = reset
        else:
            selected = current
        saved = pl.tile.store(selected, [0, 0], out)
        result = pl.tile.store(held, [64, 0], saved)
        return result

if __name__ == "__main__":
    import sys
    p = BranchGather if sys.argv[1] == "gather" else ConditionalReset
    p = passes.convert_to_ssa()(p)
    p = passes.init_mem_ref()(p)
    p = passes.materialize_semantic_aliases()(p)
    p = passes.memory_reuse()(p)
    passes.allocate_memory_addr()(p)
