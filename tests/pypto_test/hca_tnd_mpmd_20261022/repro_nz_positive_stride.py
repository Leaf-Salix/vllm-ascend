# SPDX-License-Identifier: Apache-2.0
"""纯CPU复现：SPMD outlining丢失编排侧正步长，合法NZ列地址被拒绝。

两版只有range的步长写法不同：编排局部变量与直接编译常量。
没有设备执行、权重或外部文件依赖；输出IR用于核查Simplify和outlining的顺序。
"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions


TOKENS = pl.dynamic("NZ_STRIDE_T")


@pl.jit.inline(auto_scope=False)
def captured_step(
    a: pl.Tensor[[TOKENS, 256], pl.INT8],
    w: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ],
    out: pl.Out[pl.Tensor[[1024, 128], pl.INT32]],
    ROWS: pl.constexpr,
):
    workers = 24
    with pl.spmd(workers):
        worker = pl.tile.get_block_idx()
        for unit in pl.range(worker, 32, workers):
            col = pl.max(0, unit % 32 * 128)
            lhs = pl.load(a, [0, 0], [ROWS, 256], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(w, [0, col], [256, 128], target_memory=pl.MemorySpace.Mat)
            value = pl.matmul(lhs, rhs, out_dtype=pl.INT32)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                shard = pl.aiv_shard(value)
                pl.store(shard, [unit * ROWS, lane * 64], out)
    return out


@pl.jit.inline(auto_scope=False)
def literal_step(
    a: pl.Tensor[[TOKENS, 256], pl.INT8],
    w: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ],
    out: pl.Out[pl.Tensor[[1024, 128], pl.INT32]],
    ROWS: pl.constexpr,
):
    with pl.spmd(24):
        worker = pl.tile.get_block_idx()
        for unit in pl.range(worker, 32, 24):
            col = pl.max(0, unit % 32 * 128)
            lhs = pl.load(a, [0, 0], [ROWS, 256], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(w, [0, col], [256, 128], target_memory=pl.MemorySpace.Mat)
            value = pl.matmul(lhs, rhs, out_dtype=pl.INT32)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                shard = pl.aiv_shard(value)
                pl.store(shard, [unit * ROWS, lane * 64], out)
    return out



@pl.jit(auto_scope=False)
def captured_entry(
    a: pl.Tensor[[TOKENS, 256], pl.INT8],
    w: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ],
    out: pl.Out[pl.Tensor[[1024, 128], pl.INT32]],
):
    tokens = pl.tensor.dim(a, 0)
    if tokens <= 16:
        out = captured_step(a, w, out, 16)
    else:
        out = captured_step(a, w, out, 32)
    return out


@pl.jit(auto_scope=False)
def literal_entry(
    a: pl.Tensor[[TOKENS, 256], pl.INT8],
    w: pl.Tensor[[8192, 4096], pl.INT8, pl.NZ],
    out: pl.Out[pl.Tensor[[1024, 128], pl.INT32]],
):
    tokens = pl.tensor.dim(a, 0)
    if tokens <= 16:
        out = literal_step(a, w, out, 16)
    else:
        out = literal_step(a, w, out, 32)
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {}
    for name, entry in (("captured_step", captured_entry), ("literal_step", literal_entry)):
        try:
            entry.compile(config=CompileOptions(
                platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                dump_passes=True, output_dir=str(args.output / name),
            ))
            result[name] = {"compiled": True}
        except Exception as exc:
            result[name] = {"compiled": False, "error": str(exc)}
    result["device_execution"] = False
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
