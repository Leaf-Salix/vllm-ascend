# SPDX-License-Identifier: Apache-2.0
"""纯CPU：constexpr分支内manual_scope/parallel的Tensor合流缺少yield。"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions

T = pl.dynamic("T")


@pl.jit.inline(auto_scope=False)
def producer(x: pl.Tensor[[T, 32], pl.FP32], tmp: pl.Tensor[[T, 32], pl.FP32], group: pl.Scalar[pl.INDEX]):
    with pl.spmd(1) as ready:
        row = group * 32 + pl.tile.get_block_idx() * 32
        count = pl.min(32, pl.tensor.dim(x, 0) - row)
        value = pl.load(x, [row, 0], [32, 32], valid_shape=[count, 32])
        pl.store(pl.add(value, 1.0), [row, 0], tmp)
    return tmp, ready


@pl.jit.inline(auto_scope=False)
def branch(x: pl.Tensor[[T, 32], pl.FP32], out: pl.Tensor[[T, 32], pl.FP32], COUNT: pl.constexpr):
    tokens = pl.tensor.dim(x, 0)
    tmp = pl.create_tensor([tokens, 32], dtype=pl.FP32)
    deps = pl.array.create(COUNT, pl.TASK_ID)
    if COUNT == 1:
        tmp, ready = producer(x, tmp, 0)
        deps[0] = ready
    else:
        with pl.manual_scope():
            for group in pl.parallel(COUNT):
                tmp, ready = producer(x, tmp, group)
                deps[group] = ready
    with pl.spmd(COUNT, deps=[deps]):
        row = pl.tile.get_block_idx() * 32
        count = pl.min(32, tokens - row)
        value = pl.load(tmp, [row, 0], [32, 32], valid_shape=[count, 32])
        pl.store(value, [row, 0], out)
    return out


@pl.jit(auto_scope=False)
def entry(x: pl.Tensor[[T, 32], pl.FP32], out: pl.Out[pl.Tensor[[T, 32], pl.FP32]]):
    if pl.tensor.dim(x, 0) <= 32:
        out = branch(x, out, 1)
    else:
        out = branch(x, out, 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"device_execution": False}
    try:
        entry.compile(config=CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                                            dump_passes=True, output_dir=str(args.output / "build")))
        report["compiled"] = True
    except Exception as exc:
        report.update(compiled=False, error=str(exc), exception_type=type(exc).__name__)
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
