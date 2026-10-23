# SPDX-License-Identifier: Apache-2.0
"""CPU比较复合尺寸内部分配：直接SPMD作用域与显式incore参数边界。"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions

B = pl.dynamic("COMPOUND_B")
T = pl.dynamic("COMPOUND_T")
ROWS = pl.dynamic("COMPOUND_WORK_ROWS")


@pl.jit.incore
def fill_and_copy(work: pl.Out[pl.Tensor[[ROWS, 16], pl.FP32]], out: pl.Out[pl.Tensor[[1, 16], pl.FP32]]):
    value = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
    pl.store(value, [0, 0], work)
    copied = pl.load(work, [0, 0], [1, 16])
    pl.store(copied, [0, 0], out)
    return work, out


@pl.jit
def direct_scope(src: pl.Tensor[[B, T], pl.FP32], out: pl.Out[pl.Tensor[[1, 16], pl.FP32]]):
    b = pl.tensor.dim(src, 0)
    t = pl.tensor.dim(src, 1)
    work = pl.create_tensor([(b + 15) // 16 * 16 + t * 4, 16], dtype=pl.FP32)
    with pl.spmd(1):
        worker = pl.tile.get_block_idx()
        value = pl.tile.full([1, 16], dtype=pl.FP32, value=0)
        pl.store(value, [worker, 0], work)
        copied = pl.load(work, [worker, 0], [1, 16])
        pl.store(copied, [worker, 0], out)


@pl.jit
def explicit_task(src: pl.Tensor[[B, T], pl.FP32], out: pl.Out[pl.Tensor[[1, 16], pl.FP32]]):
    b = pl.tensor.dim(src, 0)
    t = pl.tensor.dim(src, 1)
    work = pl.create_tensor([(b + 15) // 16 * 16 + t * 4, 16], dtype=pl.FP32)
    with pl.spmd(1):
        work, out = fill_and_copy(work, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = []
    for entry in (direct_scope, explicit_task):
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
