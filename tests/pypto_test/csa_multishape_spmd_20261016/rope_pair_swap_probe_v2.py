# SPDX-License-Identifier: Apache-2.0
"""CPU探针：先转置到对齐行，再交换行对，最后转回；尚无设备结论。"""

import pypto.language as pl


@pl.jit
def entry(x: pl.Tensor[[1, 512], pl.FP32], out: pl.Out[pl.Tensor[[1, 512], pl.FP32]]):
    for worker in pl.spmd(1, name_hint="rope_pair_swap"):
        values = pl.tile.load(x, [worker, 0], [1, 512], target_memory=pl.Mem.Vec)
        matrix = pl.tile.reshape(values, [64, 8])
        columns = pl.tile.transpose(matrix, 0, 1)
        swapped = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for pair in pl.range(4):
            even = pl.tile.extract(columns, pair * 2, 0, [1, 64], target_memory=pl.Mem.Vec)
            odd = pl.tile.extract(columns, pair * 2 + 1, 0, [1, 64], target_memory=pl.Mem.Vec)
            swapped = pl.tile.assemble(swapped, odd, [pair * 2, 0])
            swapped = pl.tile.assemble(swapped, even, [pair * 2 + 1, 0])
        restored = pl.tile.transpose(swapped, 0, 1)
        flat = pl.tile.reshape(restored, [1, 512])
        out = pl.tile.store(flat, [worker, 0], out)
    return out
