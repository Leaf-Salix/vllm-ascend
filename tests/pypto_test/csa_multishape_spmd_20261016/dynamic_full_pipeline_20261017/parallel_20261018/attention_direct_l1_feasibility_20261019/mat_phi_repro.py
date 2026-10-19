# SPDX-License-Identifier: Apache-2.0
"""两个仍存活的 Mat 缓冲之间做条件选择；不依赖 gm_view、MGATHER 或 CSA。"""
import json
import sys
import traceback
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.ir.printer import python_print
from pypto.runtime import CompileOptions


@pl.jit.incore
def mat_phi_core(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    a = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    b = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Mat)
    rhs = pl.load(y, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    if pl.read(flag, [0]) > 0:
        selected = pl.yield_(a)
    else:
        selected = pl.yield_(b)
    product = pl.matmul(selected, rhs, out_dtype=pl.FP32)
    pl.store(product, [0, 0], out)
    # a/b 在选择后仍分别使用，不能通过把它们强制合并成同一物理地址来修复。
    keep_a = pl.matmul(a, rhs, out_dtype=pl.FP32)
    pl.store(keep_a, [16, 0], out)
    keep_b = pl.matmul(b, rhs, out_dtype=pl.FP32)
    pl.store(keep_b, [32, 0], out)
    return out


@pl.jit
def mat_phi(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    with pl.spmd(1):
        out = mat_phi_core(x, y, flag, out)
    return out


if __name__ == "__main__":
    folder = Path(sys.argv[1]).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(compiled=False, device_execution=False)
    try:
        (folder / "symbolic.py").write_text(python_print(mat_phi.specialize()))
        artifact = mat_phi._resolve_kernel_artifact(
            (), {"config": CompileOptions(platform="a2a3", output_dir=str(folder / "kernel"),
                                          dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)},
            allow_signature_mode=True,
        )
        result.update(compiled=True, artifact=str(artifact.output_dir))
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
    (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False))
    raise SystemExit(0 if result["compiled"] else 1)
