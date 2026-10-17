# SPDX-License-Identifier: Apache-2.0
"""CPU编译探针：未重新赋值的DPS assemble能否跨循环保留L1更新。

没有设备调用，不能当数值/性能证据。用于判断未来N64循环流水是否能避免
Mat/Right作为显式循环携带值所引入的不支持搬运；不是当前CSA候选。
"""

import pypto.language as pl


@pl.jit(auto_scope=False)
def entry(a: pl.Tensor[[16, 16], pl.FP16], b: pl.Tensor[[16, 16], pl.FP16],
          out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
    with pl.spmd(1, name_hint="tile_inplace_loop_probe"):
        worker = pl.tile.get_block_idx()
        a_mat = pl.tile.load(a, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        b_mat = pl.tile.load(b, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        a_left = pl.tile.move(a_mat, target_memory=pl.Mem.Left)
        b_right = pl.tile.move(b_mat, target_memory=pl.Mem.Right)
        pool = pl.tile.create([16, 32], dtype=pl.FP16, target_memory=pl.Mem.Mat)
        for panel in pl.range(2):
            product = pl.tile.matmul(a_left, b_right)
            pl.tile.assemble(pool, product, [0, panel * 16])
        retained = pl.tile.extract(pool, 0, 16, [16, 16], target_memory=pl.Mem.Right)
        result = pl.tile.matmul(a_left, retained)
        out = pl.tile.store(result, [worker * 16, 0], out)
    return out
