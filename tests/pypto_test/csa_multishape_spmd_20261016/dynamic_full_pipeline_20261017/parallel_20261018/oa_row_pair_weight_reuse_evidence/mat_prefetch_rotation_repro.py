# SPDX-License-Identifier: Apache-2.0
"""CPU最小复现：Mat current/next跨if与loop时不应产生目标不支持的Mat复制。

该脚本不使用权重或设备，不作为CSA数值/性能测试。保存全编译结果和失败
日志，用于区分#3016的同窗口phi与这里不同存储的预取轮转需求。
"""

import argparse
import json
import traceback
from pathlib import Path

import pypto.language as pl


@pl.jit.incore
def rotate(a: pl.Tensor[[512, 64], pl.FP16], b: pl.Tensor[[64, 16], pl.FP16],
           out: pl.Out[pl.Tensor[[512, 16], pl.FP32]]):
    current = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    weight = pl.load(b, [0, 0], [64, 16], target_memory=pl.MemorySpace.Mat)
    right = pl.tile.move(weight, target_memory=pl.MemorySpace.Right)
    for block in pl.range(8):
        if block + 1 < 8:
            following = pl.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        else:
            following = current
        left = pl.tile.move(current, target_memory=pl.MemorySpace.Left)
        result = pl.tile.matmul(left, right)
        pl.store(result, [block * 64, 0], out)
        current = following
    return out


@pl.jit
def entry(a: pl.Tensor[[512, 64], pl.FP16], b: pl.Tensor[[64, 16], pl.FP16],
          out: pl.Out[pl.Tensor[[512, 16], pl.FP32]]):
    with pl.spmd(1):
        out = rotate(a, b, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from pypto import passes
    from pypto.runtime import CompileOptions

    report = dict(passed=False, device_execution=False)
    try:
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                                 output_dir=str(args.output / "kernel"), dump_passes=True)
        artifact = entry._resolve_kernel_artifact((), {"config": options}, allow_signature_mode=True)
        report.update(passed=True, artifact=str(artifact.output_dir))
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
