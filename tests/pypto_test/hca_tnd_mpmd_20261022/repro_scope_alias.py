# SPDX-License-Identifier: Apache-2.0
"""纯CPU检查：scope内回传的manual_dep别名在外部消费时是否生成未声明SSA。"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions
from pypto.runtime.kernel_compiler import KernelCompiler


@pl.jit.inline(auto_scope=False)
def produce(x: pl.Tensor[[16, 32], pl.FP32], q: pl.Tensor[[16, 32], pl.FP32]):
    with pl.spmd(1) as ready:
        row = pl.tile.get_block_idx() * 16
        q[row:row + 16, :] = pl.add(x[row:row + 16, :], 1.0)
    return q, ready


@pl.jit(auto_scope=False)
def outside(x: pl.Tensor[[16, 32], pl.FP32], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            captured: pl.Out[pl.Tensor[[16, 32], pl.FP32]]):
    q = pl.create_tensor([16, 32], dtype=pl.FP32, manual_dep=True)
    signals = pl.array.create(1, pl.TASK_ID)
    signals[0] = pl.system.task_invalid()
    with pl.scope():
        q, ready = produce(x, q)
        signals[0] = ready
        with pl.spmd(1, deps=[signals]):
            row = pl.tile.get_block_idx() * 16
            out[row:row + 16, :] = pl.mul(q[row:row + 16, :], 2.0)
    with pl.spmd(1, deps=[signals]):
        row = pl.tile.get_block_idx() * 16
        captured[row:row + 16, :] = q[row:row + 16, :]
    return out


@pl.jit(auto_scope=False)
def inside(x: pl.Tensor[[16, 32], pl.FP32], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
           captured: pl.Out[pl.Tensor[[16, 32], pl.FP32]]):
    q = pl.create_tensor([16, 32], dtype=pl.FP32, manual_dep=True)
    signals = pl.array.create(1, pl.TASK_ID)
    signals[0] = pl.system.task_invalid()
    with pl.scope():
        q, ready = produce(x, q)
        signals[0] = ready
        with pl.spmd(1, deps=[signals]):
            row = pl.tile.get_block_idx() * 16
            out[row:row + 16, :] = pl.mul(q[row:row + 16, :], 2.0)
        with pl.spmd(1, deps=[signals]):
            row = pl.tile.get_block_idx() * 16
            captured[row:row + 16, :] = q[row:row + 16, :]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    compiler = KernelCompiler(platform="a2a3")
    result = {"device_execution": False}
    for name, entry in (("outside", outside), ("inside", inside)):
        directory = args.output / name
        row = {"lowering": False, "orchestration": False}
        try:
            entry.compile(config=CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                                                output_dir=str(directory), dump_passes=True))
            row["lowering"] = True
            sources = list(directory.glob("orchestration/*.cpp"))
            if not sources:
                raise AssertionError("未生成编排C++")
            for source in sources:
                compiler.compile_orchestration("tensormap_and_ringbuffer", str(source))
            row["orchestration"] = True
        except Exception as error:
            row["error"] = str(error)
        result[name] = row
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
