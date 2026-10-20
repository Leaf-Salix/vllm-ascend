# SPDX-License-Identifier: Apache-2.0
"""只在 CPU 编译 A3 gather_compare，用于核对标量阈值到 ISA 的 ABI。"""

import argparse
import json
import traceback
from pathlib import Path

import pypto.language as pl


@pl.jit.incore
def compare_kernel(
    src: pl.Tensor[[1, 64], pl.FP32],
    selected: pl.Out[pl.Tensor[[1, 64], pl.INT32]],
    counts: pl.Out[pl.Tensor[[1, 8], pl.INT32]],
):
    values = pl.load(src, [0, 0], [1, 64])
    tmp = pl.tile.create([1, 512], dtype=pl.UINT8)
    ids, count = pl.tile.gather_compare(values, pl.cast(0.5, pl.FP32), tmp,
                                       cmp_mode="gt", out_cols=64)
    pl.store(ids, [0, 0], selected)
    pl.store(count, [0, 0], counts)
    return selected, counts


@pl.jit
def entry(src: pl.Tensor[[1, 64], pl.FP32],
          selected: pl.Out[pl.Tensor[[1, 64], pl.INT32]],
          counts: pl.Out[pl.Tensor[[1, 8], pl.INT32]]):
    with pl.spmd(1):
        selected, counts = compare_kernel(src, selected, counts)
    return selected, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from pypto import passes
    from pypto.runtime import CompileOptions, ensure_pto_isa_root
    from simpler_setup.kernel_compiler import KernelCompiler

    report = dict(lowering=False, native_compile=False, device_run=False)
    try:
        config = CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"),
                                dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)
        artifact = entry._resolve_kernel_artifact((), {"config": config}, allow_signature_mode=True)
        report["lowering"] = True
        report["isa"] = ensure_pto_isa_root()
        cpp = Path(artifact.output_dir) / "kernels/aiv/compare_kernel.cpp"
        report["cpp"] = str(cpp)
        raw = KernelCompiler("a2a3").compile_incore(str(cpp), core_type="aiv",
                                                  pto_isa_root=report["isa"])
        report.update(native_compile=True, local_binary_bytes=len(raw))
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, ensure_ascii=False))
    if not report["native_compile"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
