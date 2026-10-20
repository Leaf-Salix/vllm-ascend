# SPDX-License-Identifier: Apache-2.0
"""不使用任何新增压缩原语，独立复现tile.full整数值/FP32目标的编译错误。"""
import argparse
import json
from pathlib import Path
import traceback

import pypto
import pypto.language as pl


@pl.jit.incore
def fill_kernel(out: pl.Out[pl.Tensor[[1, 64], pl.FP32]]):
    value = pl.tile.full([1, 64], dtype=pl.FP32, value=0)
    pl.store(value, [0, 0], out)
    return out


@pl.jit
def entry(out: pl.Out[pl.Tensor[[1, 64], pl.FP32]]):
    with pl.spmd(1):
        out = fill_kernel(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pypto.configure_cache(pypto.CacheConfig(enabled=False))
    report = {"passed": False, "device_run": False, "new_compact_op_used": False}
    try:
        from pypto.runtime import CompileOptions
        from pypto import pypto_core
        report["core"] = pypto_core.__file__
        config = CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"),
                                dump_passes=True)
        artifact = entry._resolve_kernel_artifact(
            (), {"config": config}, allow_signature_mode=True
        )
        report.update(passed=True, artifact=str(artifact.output_dir))
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("error", "traceback")},
                     ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
