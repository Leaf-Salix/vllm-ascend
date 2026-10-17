# SPDX-License-Identifier: Apache-2.0
"""CPU探针：不使用gather的RoPE相邻交换，尚无精度或性能结论。"""

import pypto.language as pl


@pl.jit
def entry(x: pl.Tensor[[1, 512], pl.FP32], out: pl.Out[pl.Tensor[[1, 512], pl.FP32]]):
    for worker in pl.spmd(1, name_hint="rope_pair_swap"):
        values = pl.tile.load(x, [worker, 0], [1, 512], target_memory=pl.Mem.Vec)
        pairs = pl.tile.reshape(values, [256, 2])
        even = pl.tile.extract(pairs, 0, 0, [256, 1], target_memory=pl.Mem.Vec)
        odd = pl.tile.extract(pairs, 0, 1, [256, 1], target_memory=pl.Mem.Vec)
        swapped = pl.tile.concat(odd, even)
        flat = pl.tile.reshape(swapped, [1, 512])
        out = pl.tile.store(flat, [worker, 0], out)
    return out
