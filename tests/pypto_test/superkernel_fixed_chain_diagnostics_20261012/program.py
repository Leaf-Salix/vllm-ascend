import pypto.language as pl

@pl.program
class TraceProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def task(self, data: pl.InOut[pl.Tensor[[27664,512],pl.FP32]],
             trace: pl.Out[pl.Tensor[[48,8],pl.INT64]], stage: pl.Scalar[pl.INT64],
             work: pl.Scalar[pl.INT64], scalar_steps: pl.Scalar[pl.INT64], seed: pl.Scalar[pl.INT64]) -> tuple[pl.Tensor[[27664,512],pl.FP32],pl.Tensor[[48,8],pl.INT64]]:
        pl.func_attr({"async_return_trace": trace})
        block = pl.tile.get_block_idx()
        state = seed + stage + block
        for _ in pl.range(scalar_steps):
            state = (state * 17 + 3) % 65521
        pl.system.wait_pre_task_end()
        value = pl.load(data, [(stage * 48 + block) * 32, 0], [32,512])
        value = pl.add(value, pl.cast(state % 17, pl.FP32))
        for _ in pl.range(work):
            value = pl.add(value, 1.0)
        data = pl.store(value, [((stage + 1) * 48 + block) * 32,0],data)
        pl.system.set_next_task_start()
        return data, trace

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, data: pl.InOut[pl.Tensor[[27664,512],pl.FP32]],
             trace: pl.Out[pl.Tensor[[817,8],pl.INT64]], length: pl.Scalar[pl.INT64],
             work: pl.Scalar[pl.INT64], scalar_steps: pl.Scalar[pl.INT64], seed: pl.Scalar[pl.INT64]) -> tuple[pl.Tensor[[27664,512],pl.FP32],pl.Tensor[[817,8],pl.INT64]]:
        for stage in pl.range(length):
            stage_trace = pl.slice(trace,[48,8],[stage * 48,0])
            with pl.spmd(48, allow_early_resolve=True):
                data, stage_trace = self.task(data,stage_trace,stage,work,scalar_steps,seed)
        return data, trace
