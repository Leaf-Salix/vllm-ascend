# SPDX-License-Identifier: Apache-2.0
"""仅 CPU 复现：带有效尺寸的 Mat 子视图 move 到 Right 编译失败。"""

import sys

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions


@pl.jit.incore
def kernel(
    a: pl.Tensor[[16, 256], pl.FP16], b: pl.Tensor[[384, 64], pl.FP16], out: pl.InOut[pl.Tensor[[16, 64], pl.FP32]]
):
    lhs = pl.load(a, [0, 0], [16, 256], valid_shape=[16, 64], target_memory=pl.MemorySpace.Mat)
    parent = pl.load(b, [0, 0], [384, 64], target_memory=pl.MemorySpace.Mat)
    view = pl.tile.slice(parent, [256, 64], [64, 0], valid_shape=[64, 64])
    left = pl.move(lhs, target_memory=pl.MemorySpace.Left)
    right = pl.move(view, target_memory=pl.MemorySpace.Right)
    return pl.store(pl.tile.matmul(left, right), [0, 0], out)


@pl.jit
def entry(
    a: pl.Tensor[[16, 256], pl.FP16], b: pl.Tensor[[384, 64], pl.FP16], out: pl.InOut[pl.Tensor[[16, 64], pl.FP32]]
):
    with pl.spmd(1):
        out = kernel(a, b, out)
    return out


if __name__ == "__main__":
    config = CompileOptions(
        platform="a2a3", output_dir=sys.argv[1], dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO
    )
    entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
