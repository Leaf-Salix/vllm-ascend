# SPDX-License-Identifier: Apache-2.0
"""仅 CPU 检查 Matmul 是否可表达原生 Cube/FIX 的成对 unit-flag。"""

import argparse
import json
import runpy
import textwrap
from pathlib import Path

from pypto import passes
from pypto.runtime import CompileOptions


def source(kind):
    # JIT parses all constexpr branches before folding; use separate programs.
    common = """\
import pypto.language as pl

@pl.jit.incore
def kernel(a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
           out: pl.InOut[pl.Tensor[[64, 128], pl.FP32]]):
    lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
    rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
"""
    bodies = {
        "control": "result = pl.tile.matmul(lhs, rhs)\nreturn pl.store(result, [0, 0], out)\n",
        "matmul_final": "result = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)\n"
        "return pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)\n",
        "matmul_acc_final": "seed = pl.tile.create([64, 128], pl.FP32, target_memory=pl.MemorySpace.Acc)\n"
        "result = pl.tile.matmul_acc(seed, lhs, rhs, init_cond=True, acc_phase=pl.AccPhase.Final)\n"
        "return pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)\n",
    }
    return (
        common
        + textwrap.indent(bodies[kind], "    ")
        + """
@pl.jit
def entry(a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
          out: pl.InOut[pl.Tensor[[64, 128], pl.FP32]]):
    with pl.spmd(1):
        out = kernel(a, b, out)
    return out
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    report = dict(device_execution=False, cases={})
    for name in ("control", "matmul_final", "matmul_acc_final"):
        script = args.output / f"{name}.py"
        script.write_text(source(name))
        config = CompileOptions(
            platform="a2a3",
            output_dir=str(args.output / name),
            dump_passes=True,
            memory_planner=passes.MemoryPlanner.PYPTO,
        )
        try:
            entry = runpy.run_path(str(script))["entry"]
            artifact = entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
            report["cases"][name] = dict(compiled=True, artifact=str(artifact.output_dir))
        except Exception as exc:
            report["cases"][name] = dict(compiled=False, error=str(exc))
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
