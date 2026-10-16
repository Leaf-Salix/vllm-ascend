# SPDX-License-Identifier: Apache-2.0
"""两个顺序阶段的 L0A 分段复用；仅供 CPU 编译定位，不是 CSA 功能测试。"""
import pypto.language as pl


@pl.program
class PhasePartition:
    @pl.function(type=pl.FunctionType.AIC)
    def body(self, x: pl.Tensor[[496, 128], pl.FP16], w: pl.Tensor[[128, 32], pl.FP16],
             out: pl.Out[pl.Tensor[[496, 32], pl.FP32]]) -> pl.Tensor[[496, 32], pl.FP32]:
        weight = pl.tile.load(w, [0, 0], [128, 32], target_memory=pl.Mem.Mat)
        right = pl.tile.move(weight, target_memory=pl.Mem.Right)
        # 第一阶段的两个 Left Tile 在第一个 matmul 前均已产生：48 + 12 KiB。
        a_mat = pl.tile.load(x, [0, 0], [192, 128], target_memory=pl.Mem.Mat)
        b_mat = pl.tile.load(x, [192, 0], [48, 128], target_memory=pl.Mem.Mat)
        a = pl.tile.move(a_mat, target_memory=pl.Mem.Left)
        b = pl.tile.move(b_mat, target_memory=pl.Mem.Left)
        a_out = pl.tile.matmul(a, right)
        out = pl.tile.store(a_out, [0, 0], out)
        b_out = pl.tile.matmul(b, right)
        out = pl.tile.store(b_out, [192, 0], out)
        pl.system.fence()
        # 第一阶段完成后，第二阶段需要同时容纳 32 + 32 KiB。
        c_mat = pl.tile.load(x, [240, 0], [128, 128], target_memory=pl.Mem.Mat)
        d_mat = pl.tile.load(x, [368, 0], [128, 128], target_memory=pl.Mem.Mat)
        c = pl.tile.move(c_mat, target_memory=pl.Mem.Left)
        d = pl.tile.move(d_mat, target_memory=pl.Mem.Left)
        c_out = pl.tile.matmul(c, right)
        out = pl.tile.store(c_out, [240, 0], out)
        d_out = pl.tile.matmul(d, right)
        out = pl.tile.store(d_out, [368, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, x: pl.Tensor[[496, 128], pl.FP16], w: pl.Tensor[[128, 32], pl.FP16],
              out: pl.Out[pl.Tensor[[496, 32], pl.FP32]]) -> pl.Tensor[[496, 32], pl.FP32]:
        with pl.spmd(1):
            out = self.body(x, w, out)
        return out
