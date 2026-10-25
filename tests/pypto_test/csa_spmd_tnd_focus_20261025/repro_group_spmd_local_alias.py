# SPDX-License-Identifier: Apache-2.0
"""CPU复现：SPMD内部reshape别名交给Group，与SPMD外建立同一别名对照。"""

import argparse
import json
import subprocess
from pathlib import Path

import pypto
import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions


@pl.jit.incore
def cube(
    left: pl.Tensor[[16, 16], pl.BF16],
    right: pl.Tensor[[16, 16], pl.BF16],
    output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
):
    lhs = pl.load(left, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    rhs = pl.load(right, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    value = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
    pl.store(value, [0, 0], output)
    return output


@pl.jit.incore
def vector(
    work: pl.InOut[pl.Tensor[[128], pl.INT32]],
    matrix: pl.InOut[pl.Tensor[[2, 64], pl.INT32]],
):
    pl.func_attr({"split_aiv": True})
    lane = pl.tile.get_subblock_idx()
    value = pl.tile.full([1, 64], dtype=pl.INT32, value=7)
    pl.store(value, [lane, 0], matrix)
    return work


@pl.jit.inline(auto_scope=False)
def group_with_local_alias(
    work: pl.InOut[pl.Tensor[[128], pl.INT32]],
    left: pl.Tensor[[16, 16], pl.BF16],
    right: pl.Tensor[[16, 16], pl.BF16],
    output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
):
    matrix = pl.reshape(work, [2, 64])
    with pl.cluster(name_hint="local_alias_group"):
        output = cube(left, right, output)
        work = vector(work, matrix)
    return work, output


@pl.jit.inline(auto_scope=False)
def group_with_ready_alias(
    work: pl.InOut[pl.Tensor[[128], pl.INT32]],
    matrix: pl.InOut[pl.Tensor[[2, 64], pl.INT32]],
    left: pl.Tensor[[16, 16], pl.BF16],
    right: pl.Tensor[[16, 16], pl.BF16],
    output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
):
    with pl.cluster(name_hint="ready_alias_group"):
        output = cube(left, right, output)
        work = vector(work, matrix)
    return work, output


@pl.jit
def alias_inside_spmd(
    work: pl.InOut[pl.Tensor[[128], pl.INT32]],
    left: pl.Tensor[[16, 16], pl.BF16],
    right: pl.Tensor[[16, 16], pl.BF16],
    output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
):
    with pl.spmd(1):
        work, output = group_with_local_alias(work, left, right, output)
    return work, output


@pl.jit
def alias_before_spmd(
    work: pl.InOut[pl.Tensor[[128], pl.INT32]],
    left: pl.Tensor[[16, 16], pl.BF16],
    right: pl.Tensor[[16, 16], pl.BF16],
    output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
):
    matrix = pl.reshape(work, [2, 64])
    with pl.spmd(1):
        work, output = group_with_ready_alias(work, matrix, left, right, output)
    return work, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    workspace = next(p for p in Path(__file__).resolve().parents if (p / "env-dsv4-0251rc1.sh").is_file())
    manifest_path = workspace / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    revisions = {}
    for name in ("pypto", "ptoas", "simpler"):
        root = workspace / manifest[name]["source"]
        revisions[name] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        assert revisions[name] == manifest[name]["commit"], name
    assert Path(pypto.__file__).resolve().is_relative_to(workspace / manifest["pypto"]["source"])
    report = {
        "shared_toolchain": manifest,
        "actual_sdk_revisions": revisions,
        "pypto_import": pypto.__file__,
        "source": str(Path(__file__).resolve()),
        "device_execution": False,
        "sdk_modified": False,
        "fixed_shapes": True,
        "element_count": "128 == 2 * 64",
        "hardware_barriers": False,
        "runs": [],
    }
    for entry in (alias_inside_spmd, alias_before_spmd):
        name = entry.__name__
        try:
            entry.compile(
                config=CompileOptions(
                    platform="a2a3",
                    output_dir=str(args.output / name),
                    memory_planner=passes.MemoryPlanner.PYPTO,
                    dump_passes=True,
                )
            )
        except Exception as error:
            row = {"name": name, "status": "FAIL", "error": str(error)}
        else:
            row = {"name": name, "status": "PASS"}
        report["runs"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
