# SPDX-License-Identifier: Apache-2.0
"""仅CPU编译：自然while的链表节点回边与显式pl.while_对照。

只保存源码/生成IR，不在设备运行可能不终止的自然while。节点只在循环内部
使用，函数返回写入的Tensor；合法测试链表应最终到达-1。期望两种写法都保留
loop-carried index，而非始终用循环前的head比较及访问。
"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto.runtime import CompileOptions

ROWS = pl.dynamic("WHILE_REPRO_ROWS")


@pl.jit.inline(auto_scope=False)
def natural_walk(links: pl.Tensor[[ROWS], pl.INT32], output: pl.InOut[pl.Tensor[[ROWS], pl.INT32]]):
    node = pl.cast(pl.read(links, [0]), pl.INDEX)
    while node >= 0:
        pl.write(output, [node], pl.cast(node, pl.INT32))
        node = pl.cast(pl.read(links, [node]), pl.INDEX)
    return output


@pl.jit.inline(auto_scope=False)
def explicit_walk(links: pl.Tensor[[ROWS], pl.INT32], output: pl.InOut[pl.Tensor[[ROWS], pl.INT32]]):
    initial = pl.cast(pl.read(links, [0]), pl.INDEX)
    for (node,) in pl.while_(init_values=(initial,)):
        pl.cond(node >= 0)
        pl.write(output, [node], pl.cast(node, pl.INT32))
        next_node = pl.cast(pl.read(links, [node]), pl.INDEX)
        completed_node = pl.yield_(next_node)  # noqa: F841 — DSL回边的结果绑定
    return output


@pl.jit.incore
def natural_core(links: pl.Tensor[[ROWS], pl.INT32], output: pl.InOut[pl.Tensor[[ROWS], pl.INT32]]):
    output = natural_walk(links, output)
    return output


@pl.jit.incore
def explicit_core(links: pl.Tensor[[ROWS], pl.INT32], output: pl.InOut[pl.Tensor[[ROWS], pl.INT32]]):
    output = explicit_walk(links, output)
    return output


@pl.jit
def natural_entry(links: pl.Tensor[[ROWS], pl.INT32], output: pl.InOut[pl.Tensor[[ROWS], pl.INT32]]):
    with pl.spmd(1):
        output = natural_core(links, output)
    return output


@pl.jit
def explicit_entry(links: pl.Tensor[[ROWS], pl.INT32], output: pl.InOut[pl.Tensor[[ROWS], pl.INT32]]):
    with pl.spmd(1):
        output = explicit_core(links, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, entry in (("natural", natural_entry), ("explicit", explicit_entry)):
        directory = args.output / name
        entry._resolve_kernel_artifact(
            (), {"config": CompileOptions(platform="a2a3", output_dir=str(directory))}, allow_signature_mode=True
        )
        source = next((directory / "ptoas").glob("*.pto"))
        loops = [line.strip() for line in source.read_text().splitlines() if "scf.while" in line]
        assert len(loops) == 1, loops
        reports[name] = {"while_header": loops[0], "has_carried_index": "(index) -> (index)" in loops[0]}
    reports["device_execution"] = False
    (args.output / "result.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(reports, ensure_ascii=False))


if __name__ == "__main__":
    main()
