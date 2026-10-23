# SPDX-License-Identifier: Apache-2.0
"""CPU核对混合组参数改形后，成员函数内保存的尺寸是否仍可绑定。"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions

T = pl.dynamic("MIX_CALLER_T")
N = pl.dynamic("MIX_MEMBER_N")
M = pl.dynamic("MIX_GROUP_M")


@pl.jit.incore
def cube(
    left: pl.Tensor[[16, 16], pl.BF16], right: pl.Tensor[[16, 16], pl.BF16], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
):
    a = pl.load(left, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    b = pl.load(right, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    value = pl.matmul(a, b, out_dtype=pl.FP32)
    pl.store(value, [0, 0], out)
    return out


@pl.jit.incore
def vector(work: pl.InOut[pl.Tensor[[8, N], pl.FP32]], out: pl.Out[pl.Tensor[[2, 16], pl.FP32]]):
    pl.func_attr({"split_aiv": True})
    lane = pl.tile.get_subblock_idx()
    saved = work
    for part in pl.range(2):
        value = pl.full([1, 16], dtype=pl.FP32, value=1.0)
        work[lane : lane + 1, part * 16 : part * 16 + 16] = value
    first = pl.read(work, [lane, 0])
    pl.write(out, [lane, 0], first)
    return saved, out


@pl.jit.inline(auto_scope=False)
def mixed_group(
    work: pl.InOut[pl.Tensor[[8, M], pl.FP32]],
    left: pl.Tensor[[16, 16], pl.BF16],
    right: pl.Tensor[[16, 16], pl.BF16],
    cube_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    vec_out: pl.Out[pl.Tensor[[2, 16], pl.FP32]],
):
    with pl.cluster(name_hint="mixed_extent"):
        cube_out = cube(left, right, cube_out)
        work, vec_out = vector(work, vec_out)
    return work, cube_out, vec_out


@pl.jit
def clustered(
    size: pl.Tensor[[T], pl.INT32],
    left: pl.Tensor[[16, 16], pl.BF16],
    right: pl.Tensor[[16, 16], pl.BF16],
    cube_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    vec_out: pl.Out[pl.Tensor[[2, 16], pl.FP32]],
):
    t = pl.tensor.dim(size, 0)
    work = pl.create_tensor([8, t + 32], dtype=pl.FP32)
    with pl.spmd(1):
        work, cube_out, vec_out = mixed_group(work, left, right, cube_out, vec_out)


@pl.jit
def separate(
    size: pl.Tensor[[T], pl.INT32],
    left: pl.Tensor[[16, 16], pl.BF16],
    right: pl.Tensor[[16, 16], pl.BF16],
    cube_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    vec_out: pl.Out[pl.Tensor[[2, 16], pl.FP32]],
):
    t = pl.tensor.dim(size, 0)
    work = pl.create_tensor([8, t + 32], dtype=pl.FP32)
    with pl.spmd(1):
        cube_out = cube(left, right, cube_out)
    with pl.spmd(1):
        work, vec_out = vector(work, vec_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = []
    for entry in (clustered, separate):
        try:
            entry.compile(
                config=CompileOptions(
                    platform="a2a3",
                    output_dir=str(args.output / entry.__name__),
                    memory_planner=passes.MemoryPlanner.PYPTO,
                )
            )
        except Exception as error:
            report.append({"entry": entry.__name__, "status": "FAIL", "error": str(error)})
        else:
            report.append({"entry": entry.__name__, "status": "PASS"})
        print(json.dumps(report[-1], ensure_ascii=False), flush=True)
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
