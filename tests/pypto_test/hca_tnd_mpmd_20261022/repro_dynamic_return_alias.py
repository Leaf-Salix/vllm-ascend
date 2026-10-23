# SPDX-License-Identifier: Apache-2.0
"""纯CPU核对嵌套inline返回值的动态维度别名，不运行设备。"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions

T = pl.dynamic("T_INNER")
ROOT_T = pl.dynamic("T_ROOT")


@pl.jit.inline(auto_scope=False)
def leaf(x: pl.Tensor[[T, 32], pl.FP32], q: pl.Tensor[[T, 32], pl.FP32]):
    tokens = pl.tensor.dim(x, 0)
    with pl.spmd(1) as ready:
        for row in pl.range(pl.tile.get_block_idx(), tokens):
            q[row:row + 1, :] = pl.add(x[row:row + 1, :], 1.0)
    return q, ready


@pl.jit.inline(auto_scope=False)
def wrapper(x: pl.Tensor[[T, 32], pl.FP32], q: pl.Tensor[[T, 32], pl.FP32]):
    q, ready = leaf(x, q)
    return q, ready


@pl.jit.inline(auto_scope=False)
def alias_writer(x: pl.Tensor[[T, 32], pl.FP32], q: pl.Tensor[[T, 32], pl.FP32]):
    flat = q
    with pl.spmd(1) as ready:
        for row in pl.range(pl.tile.get_block_idx(), pl.tensor.dim(x, 0)):
            flat[row:row + 1, :] = pl.add(x[row:row + 1, :], 1.0)
    return ready


@pl.jit.inline(auto_scope=False)
def tensor_leaf(x: pl.Tensor[[T, 32], pl.FP32], q: pl.Tensor[[T, 32], pl.FP32]):
    tokens = pl.tensor.dim(x, 0)
    for base in pl.range(0, tokens, 384):
        with pl.scope():
            alias_writer(x, q)
    return q


@pl.jit.inline(auto_scope=False)
def nested_alias(x: pl.Tensor[[T, 32], pl.FP32], q: pl.Tensor[[T, 32], pl.FP32]):
    q = tensor_leaf(x, q)
    ready = pl.system.task_dummy(deps=[])
    return q, ready


def make_entry(producer):
    @pl.jit(auto_scope=False)
    def entry(x: pl.Tensor[[ROOT_T, 32], pl.FP32], out: pl.Out[pl.Tensor[[ROOT_T, 32], pl.FP32]]):
        tokens = pl.tensor.dim(x, 0)
        temporary = pl.create_tensor([tokens, 32], dtype=pl.FP32)
        with pl.spmd(1):
            for row in pl.range(pl.tile.get_block_idx(), tokens):
                temporary[row:row + 1, :] = x[row:row + 1, :]
        q = pl.create_tensor([tokens, 32], dtype=pl.FP32, manual_dep=True)
        with pl.scope():
            q, ready = producer(temporary, q)
            with pl.spmd(1, deps=[ready]):
                for row in pl.range(pl.tile.get_block_idx(), tokens):
                    out[row:row + 1, :] = q[row:row + 1, :]
        return out
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"device_execution": False}
    for name, producer in (("direct", leaf), ("nested", wrapper), ("nested_alias", nested_alias)):
        try:
            make_entry(producer).compile(config=CompileOptions(
                platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                dump_passes=True, output_dir=str(args.output / name),
            ))
            result[name] = {"compiled": True}
        except Exception as error:
            result[name] = {"compiled": False, "type": type(error).__name__, "error": str(error)}
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
