# SPDX-License-Identifier: Apache-2.0
"""CPU-only reproducer: GEMV Acc column aliases become illegal Acc TMOVs."""

import sys

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions


@pl.jit.incore
def kernel(
    a: pl.Tensor[[1, 64], pl.FP16],
    b: pl.Tensor[[64, 64], pl.FP16],
    out: pl.InOut[pl.Tensor[[1, 128], pl.FP32]],
):
    lhs = pl.load(a, [0, 0], [1, 256], valid_shape=[1, 64], target_memory=pl.MemorySpace.Mat)
    rhs = pl.load(b, [0, 0], [256, 64], valid_shape=[64, 64], target_memory=pl.MemorySpace.Mat)
    left = pl.move(lhs, target_memory=pl.MemorySpace.Left)
    right = pl.move(rhs, target_memory=pl.MemorySpace.Right)
    pool = pl.tile.create([16, 128], pl.FP32, target_memory=pl.MemorySpace.Acc)
    for i in pl.unroll(2):
        slot = pl.tile.slice(pool, [16, 64], [0, i * 64], valid_shape=[1, 64])
        result = pl.tile.gemv_acc(slot, left, right, init_cond=True)
        pool = pl.tile.assemble(pool, result, [0, i * 64])
    return pl.store(pl.set_validshape(pool, 1, 128), [0, 0], out)


@pl.jit
def entry(
    a: pl.Tensor[[1, 64], pl.FP16],
    b: pl.Tensor[[64, 64], pl.FP16],
    out: pl.InOut[pl.Tensor[[1, 128], pl.FP32]],
):
    with pl.spmd(1):
        out = kernel(a, b, out)
    return out


if __name__ == "__main__":
    config = CompileOptions(
        platform="a2a3", output_dir=sys.argv[1], dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO
    )
    entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
