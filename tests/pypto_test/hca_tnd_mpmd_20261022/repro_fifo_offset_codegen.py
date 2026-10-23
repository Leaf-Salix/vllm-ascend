# SPDX-License-Identifier: Apache-2.0
"""纯CPU：条件yield经过两层Scalar FIFO后用于GM load偏移。"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions


@pl.jit.inline(auto_scope=False)
def conditional_offset(step: pl.Scalar[pl.INDEX]):
    if step % 2 == 0:
        offset = pl.yield_(pl.cast(0, pl.INDEX))
    else:
        offset = pl.yield_(pl.cast(16, pl.INDEX))
    return offset


@pl.jit.inline(auto_scope=False)
def direct_offset(step: pl.Scalar[pl.INDEX]):
    return step % 2 * 16


def make_entry(offset_for_step):
    @pl.jit(auto_scope=False)
    def entry(x: pl.Tensor[[32, 128], pl.BF16], weight: pl.Tensor[[128, 32], pl.BF16],
              out: pl.Out[pl.Tensor[[16, 32], pl.FP32]], marks: pl.Out[pl.Tensor[[2, 16], pl.INT32]]):
        with pl.spmd(1):
            worker = pl.tile.get_block_idx()
            for query, (previous, recent) in pl.range(
                worker, 2, init_values=(pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX)),
            ):
                for step, (old, new) in pl.range(3, init_values=(previous, recent)):
                    source = offset_for_step(step)
                    if step < 2:
                        saved = pl.yield_(source)
                    else:
                        saved = pl.yield_(new)
                    if step >= 1:
                        for chunk in pl.pipeline(2, stage=2):
                            lhs = pl.load(x, [old, chunk * 64], [16, 64], target_memory=pl.MemorySpace.Mat)
                            rhs = pl.load(weight, [chunk * 64, 0], [64, 32], target_memory=pl.MemorySpace.Mat)
                            product = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
                            pl.store(product, [0, 0], out)
                    old_done, new_done = pl.yield_(new, saved)
                previous_done, recent_done = pl.yield_(old_done, new_done)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                marker = pl.tile.full([1, 16], dtype=pl.INT32, value=1)
                pl.store(marker, [lane, 0], marks)
        return out, marks
    return entry


def make_entry_no_pipeline(offset_for_step):

    @pl.jit(auto_scope=False)
    def entry(x: pl.Tensor[[32, 128], pl.BF16], weight: pl.Tensor[[128, 32], pl.BF16], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]], marks: pl.Out[pl.Tensor[[2, 16], pl.INT32]]):
        with pl.spmd(1):
            worker = pl.tile.get_block_idx()
            for (query, (previous, recent)) in pl.range(worker, 2, init_values=(pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX))):
                for (step, (old, new)) in pl.range(3, init_values=(previous, recent)):
                    source = offset_for_step(step)
                    if step < 2:
                        saved = pl.yield_(source)
                    else:
                        saved = pl.yield_(new)
                    if step >= 1:
                        for chunk in pl.range(2):
                            lhs = pl.load(x, [old, chunk * 64], [16, 64], target_memory=pl.MemorySpace.Mat)
                            rhs = pl.load(weight, [chunk * 64, 0], [64, 32], target_memory=pl.MemorySpace.Mat)
                            product = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
                            pl.store(product, [0, 0], out)
                    (old_done, new_done) = pl.yield_(new, saved)
                (previous_done, recent_done) = pl.yield_(old_done, new_done)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                marker = pl.tile.full([1, 16], dtype=pl.INT32, value=1)
                pl.store(marker, [lane, 0], marks)
        return (out, marks)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"device_execution": False}
    for name, offset, factory in (("conditional", conditional_offset, make_entry), ("direct", direct_offset, make_entry),
                                  ("range_conditional", conditional_offset, make_entry_no_pipeline),
                                  ("range_direct", direct_offset, make_entry_no_pipeline)):
        try:
            factory(offset).compile(config=CompileOptions(
                platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                dump_passes=True, output_dir=str(args.output / name),
            ))
            report[name] = {"compiled": True}
        except Exception as error:
            report[name] = {"compiled": False, "type": type(error).__name__, "error": str(error)}
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
