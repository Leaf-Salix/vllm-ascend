# SPDX-License-Identifier: Apache-2.0
"""原有tensor.view直接使用内联tensor.dim的完整CPU复现，无新增stride接口。"""
import argparse
import json
from pathlib import Path
import pypto
import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions
P = pl.dynamic("INLINE_DIM_P")
W = pl.dynamic("INLINE_DIM_W")

@pl.jit.incore
def core(source: pl.Tensor[[P, W], pl.INT8], output: pl.Out[pl.Tensor[[1, 32], pl.INT8]]):
    alias = pl.tensor.view(source, [pl.tensor.dim(source, 0), pl.tensor.dim(source, 1)])
    value = pl.load(alias, [0, 0], [1, 32])
    pl.store(value, [0, 0], output)
    return output

@pl.jit
def entry(source: pl.Tensor[[P, W], pl.INT8], output: pl.Out[pl.Tensor[[1, 32], pl.INT8]]):
    with pl.spmd(1):
        output = core(source, output)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"pypto": pypto.__file__, "passed": False, "new_stride_api_used": False}
    config = CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"), dump_passes=True,
                            memory_planner=passes.MemoryPlanner.PYPTO)
    try:
        entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)
