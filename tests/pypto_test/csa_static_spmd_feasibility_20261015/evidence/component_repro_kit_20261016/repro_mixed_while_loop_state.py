# SPDX-License-Identifier: Apache-2.0
"""CPU 复现 MIX 展开丢失 while 条件携带值；不分配或运行 NPU Tensor。"""

import argparse
import json
import platform
import sys
import traceback
from pathlib import Path

import pypto
import pypto.language as pl
from pypto.runtime import RunConfig


@pl.jit
def mixed_while_repro(
    a: pl.Tensor[[384, 16], pl.FP16],
    b: pl.Tensor[[16, 16], pl.FP16],
    scratch: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
    flags: pl.InOut[pl.Tensor[[16], pl.INT32]],
    cube_out: pl.Out[pl.Tensor[[384, 16], pl.FP32]],
    vector_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    TILE_POLLING: pl.constexpr = False,
    KEEP_RESULT: pl.constexpr = False,
):
    # 矩阵运算只用于使这一段成为 MIX 核；flags 协议只作为编译输入。
    for worker in pl.spmd(24, sync_start=True, name_hint="mixed_while_repro"):
        ac = pl.load(a, [worker * 16, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        bc = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        cc = pl.matmul(ac, bc, out_dtype=pl.FP32)
        pl.store(cc, [worker * 16, 0], cube_out)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            if lane == 0:
                if worker == 0:
                    av = pl.load(a, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
                    pl.store(pl.add(pl.cast(av, pl.FP32), 1.0), [0, 0], scratch)
                    pl.system.cacheinvalid()
                    pl.system.fence()
                    pl.write(flags, [0], pl.cast(1, pl.INT32))
                    pl.system.cacheinvalid()
                    pl.system.fence()
                elif worker == 1:
                    if TILE_POLLING:
                        initial_tile = pl.load(flags, [0], [16], target_memory=pl.MemorySpace.Vec)
                        initial_ready = pl.tile.read(initial_tile, [0])
                        for (ready,) in pl.while_(init_values=(initial_ready,)):
                            pl.cond(ready != 1)
                            pl.system.cacheinvalid()
                            flag_tile = pl.load(flags, [0], [16], target_memory=pl.MemorySpace.Vec)
                            final_ready = pl.yield_(pl.tile.read(flag_tile, [0]))
                        if KEEP_RESULT:
                            # 仅供定位活跃性分析的对照；不作为生产代码绕过方案。
                            pl.write(flags, [1], final_ready)
                    else:
                        initial_ready = pl.read(flags, [0])
                        for (ready,) in pl.while_(init_values=(initial_ready,)):
                            pl.cond(ready != 1)
                            pl.system.cacheinvalid()
                            final_ready = pl.yield_(pl.read(flags, [0]))
                        if KEEP_RESULT:
                            pl.write(flags, [1], final_ready)
                    pl.system.cacheinvalid()
                    received = pl.load(scratch, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
                    pl.store(pl.mul(received, 2.0), [0, 0], vector_out)
    return cube_out, vector_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("scalar", "tile"), default="scalar")
    parser.add_argument("--keep-result", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "variant": args.variant,
        "keep_result": args.keep_result,
        "python": sys.version,
        "host_architecture": platform.machine(),
        "pypto_module": pypto.__file__,
        "pypto_core_module": getattr(sys.modules.get("pypto.pypto_core"), "__file__", None),
        "platform": "a2a3",
        "device_execution": False,
    }
    try:
        compiled = mixed_while_repro.compile(
            TILE_POLLING=args.variant == "tile",
            KEEP_RESULT=args.keep_result,
            config=RunConfig(
                platform="a2a3",
                save_kernels=True,
                save_kernels_dir=str(output / "kernel"),
                dump_passes=True,
            ),
        )
        result["code_generation_completed"] = True
        result["output_dir"] = str(compiled.output_dir)
    except Exception as exc:
        result["code_generation_completed"] = False
        result["error"] = str(exc)
        (output / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")

    # 记录 pass 边界，便于直接将生成文件与编译错误关联。
    boundary_files = sorted((output / "kernel/passes_dump").glob("*ExpandMixedKernel.py"))
    result["expand_mixed_kernel_dumps"] = [str(path.relative_to(output)) for path in boundary_files]
    result["free_ready_after_expand"] = any(
        "ready__FREE_VAR" in path.read_text(encoding="utf-8") for path in boundary_files
    )
    result["pypto_core_module"] = getattr(sys.modules.get("pypto.pypto_core"), "__file__", None)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "variant", "keep_result", "code_generation_completed", "free_ready_after_expand"
    )}, ensure_ascii=False))
    return 0 if result["code_generation_completed"] and not result["free_ready_after_expand"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
