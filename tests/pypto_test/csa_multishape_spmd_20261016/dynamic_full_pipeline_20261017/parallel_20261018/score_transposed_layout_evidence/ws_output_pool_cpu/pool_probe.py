# SPDX-License-Identifier: Apache-2.0
"""仅验证WS多面板结果在Acc中集合后一次写GM的编译可行性。

单个64×16 FP32结果4KiB，八个结果共32KiB；与QK的96KiB Acc恰合128KiB。
每个子片单独完成原K384矩阵，pool只汇集互不重叠的输出，不重排浮点规约。
本探针尚不包含QK/Key预取或完整CSA，只检查Slice/Assemble是否可保持Acc别名。
"""
import pypto.language as pl

@pl.jit.incore
def pool_incore(x: pl.Tensor[[512,384],pl.FP16], w: pl.Tensor[[384,16],pl.FP16], out: pl.Out[pl.Tensor[[512,16],pl.FP32]]):
    weight = pl.load(w,[0,0],[384,16],target_memory=pl.MemorySpace.Mat)
    right = pl.tile.move(weight,target_memory=pl.MemorySpace.Right)
    pool = pl.tile.create([512,16],dtype=pl.FP32,target_memory=pl.MemorySpace.Acc)
    for panel in pl.unroll(8):
        values = pl.load(x,[panel*64,0],[64,384],target_memory=pl.MemorySpace.Mat)
        left = pl.tile.move(values,target_memory=pl.MemorySpace.Left)
        part = pl.tile.slice(pool,[64,16],[panel*64,0])
        part = pl.tile.matmul_acc(part,left,right,init_cond=True)
        pool = pl.tile.assemble(pool,part,[panel*64,0])
    pl.store(pool,[0,0],out)
    return out

@pl.jit
def pool_entry(x: pl.Tensor[[512,384],pl.FP16], w: pl.Tensor[[384,16],pl.FP16], out: pl.Out[pl.Tensor[[512,16],pl.FP32]]):
    with pl.spmd(1):
        out = pool_incore(x,w,out)
    return out
