# SPDX-License-Identifier: Apache-2.0
"""验证候选Score key-major交接布局可在AIV恢复原query-major；仅数学tile探针。

MTE1 WS中间值搬运仍占大头，下一结构方向是把QK/WS同时转置，令WS使用
Left非转置读取。AIC结果会变成[key,query]，本探针只核对AIV端32KiB
FP32[1024,8]能否转为原[query,1024]。没有实现完整Score或CSA，不作为性能证据。
"""
import pypto.language as pl

@pl.jit.incore
def transpose_incore(x: pl.Tensor[[1024,8],pl.FP32], out: pl.Out[pl.Tensor[[3,1024],pl.FP32]]):
    source = pl.load(x,[0,0],[1024,8])
    transposed = pl.transpose(source,0,1)
    picked = pl.tile.extract(transposed,0,0,[3,1024],target_memory=pl.MemorySpace.Vec)
    pl.store(picked,[0,0],out)
    return out

@pl.jit
def transpose_entry(x: pl.Tensor[[1024,8],pl.FP32], out: pl.Out[pl.Tensor[[3,1024],pl.FP32]]):
    with pl.spmd(1):
        out = transpose_incore(x,out)
    return out
