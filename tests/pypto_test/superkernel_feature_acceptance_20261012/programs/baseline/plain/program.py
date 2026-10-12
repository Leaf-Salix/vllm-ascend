import pypto.language as pl

@pl.jit
def acceptance_chain(
    data: pl.InOut[pl.Tensor[[27664, 512], pl.FP32]],
    trace: pl.Out[pl.Tensor[[817, 8], pl.INT64]],
    length: pl.Scalar[pl.INT64],
    work: pl.Scalar[pl.INT64],
    scalar_steps: pl.Scalar[pl.INT64],
    seed: pl.Scalar[pl.INT64],
):
    for stage in pl.range(length):
        with pl.spmd(48, allow_early_resolve=True, name_hint="acceptance_task"):
            block = pl.tile.get_block_idx()
            # DIAG_BEGIN
            state = seed + stage + block
            for _ in pl.range(scalar_steps):
                state = (state * 17 + 3) % 65521
            # WAIT
            value = pl.load(data, [(stage * 48 + block) * 32, 0], [32, 512])
            value = pl.add(value, pl.cast(state % 17, pl.FP32))
            for _ in pl.range(work):
                value = pl.add(value, 1.0)
            pl.store(value, [((stage + 1) * 48 + block) * 32, 0], data)
            # SET
    return data, trace
