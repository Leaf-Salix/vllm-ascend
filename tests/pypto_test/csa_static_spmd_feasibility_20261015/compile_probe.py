"""仅做 CPU 编译：核组分支、组间标志及已有 InCore helper 的可表达性。"""

import argparse
import json
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import pypto.language as pl
from pypto.runtime import RunConfig


@pl.jit
def static_roles(
    a: pl.Tensor[[384, 16], pl.FP16],
    b: pl.Tensor[[16, 16], pl.FP16],
    cube_out: pl.Out[pl.Tensor[[384, 16], pl.FP32]],
    vector_out: pl.Out[pl.Tensor[[768, 16], pl.FP32]],
):
    # 一次 MIX SPMD；Cube 和双 Vector 各自按核编号执行不同的工作序列。
    for worker in pl.spmd(24, sync_start=True, name_hint="static_roles"):
        if worker < 12:
            ac = pl.load(a, [worker * 16, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            bc = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            cc = pl.matmul(ac, bc, out_dtype=pl.FP32)
            pl.store(cc, [worker * 16, 0], cube_out)
        else:
            for cb in pl.range(2):
                ac2 = pl.load(a, [worker * 16, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
                bc2 = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
                cc2 = pl.matmul(ac2, bc2, out_dtype=pl.FP32)
                pl.store(cc2, [worker * 16, 0], cube_out)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            vi = worker * 2 + lane
            av = pl.load(a, [worker * 16, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
            fv = pl.cast(av, pl.FP32)
            if vi < 24:
                fv = pl.add(fv, 1.0)
            else:
                fv = pl.mul(fv, 2.0)
            pl.store(fv, [vi * 16, 0], vector_out)
    return cube_out, vector_out


@pl.jit
def cross_group_signal(
    a: pl.Tensor[[384, 16], pl.FP16],
    b: pl.Tensor[[16, 16], pl.FP16],
    scratch: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
    flags: pl.InOut[pl.Tensor[[16], pl.INT32]],
    cube_out: pl.Out[pl.Tensor[[384, 16], pl.FP32]],
    vector_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    TILE_POLLING: pl.constexpr = False,
):
    # 探针假定 flags[0] 初始为零；只编译，不作为重放协议或真机正确性的证据。
    for worker in pl.spmd(24, sync_start=True, name_hint="cross_group_signal"):
        ac = pl.load(a, [worker * 16, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        bc = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        cc = pl.matmul(ac, bc, out_dtype=pl.FP32)
        pl.store(cc, [worker * 16, 0], cube_out)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            if lane == 0:
                if worker == 0:
                    av = pl.load(a, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
                    fv = pl.add(pl.cast(av, pl.FP32), 1.0)
                    pl.store(fv, [0, 0], scratch)
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
                    else:
                        initial_ready = pl.read(flags, [0])
                        for (ready,) in pl.while_(init_values=(initial_ready,)):
                            pl.cond(ready != 1)
                            pl.system.cacheinvalid()
                            final_ready = pl.yield_(pl.read(flags, [0]))
                    pl.system.cacheinvalid()
                    received = pl.load(scratch, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
                    pl.store(pl.mul(received, 2.0), [0, 0], vector_out)
    return cube_out, vector_out


@pl.jit.incore
def cube_piece(
    a: pl.Tensor[[384, 16], pl.FP16],
    b: pl.Tensor[[16, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[384, 16], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    ac = pl.load(a, [worker * 16, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    bc = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    pl.store(pl.matmul(ac, bc, out_dtype=pl.FP32), [worker * 16, 0], out)
    return out


@pl.jit
def incore_helper_call(
    a: pl.Tensor[[384, 16], pl.FP16],
    b: pl.Tensor[[16, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[384, 16], pl.FP32]],
):
    for worker in pl.spmd(24, sync_start=True, name_hint="incore_helper_call"):
        out = cube_piece(a, b, out, worker)
    return out


@pl.jit.inline
def cube_piece_inline(
    a: pl.Tensor[[384, 16], pl.FP16],
    b: pl.Tensor[[16, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[384, 16], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    ac = pl.load(a, [worker * 16, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    bc = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    pl.store(pl.matmul(ac, bc, out_dtype=pl.FP32), [worker * 16, 0], out)
    return out


@pl.jit
def inline_helper_call(
    a: pl.Tensor[[384, 16], pl.FP16],
    b: pl.Tensor[[16, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[384, 16], pl.FP32]],
):
    for worker in pl.spmd(24, sync_start=True, name_hint="inline_helper_call"):
        out = cube_piece_inline(a, b, out, worker)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    names = ("static_roles", "cross_group_signal", "incore_helper_call", "inline_helper_call")
    parser.add_argument("--case", action="append", choices=names)
    parser.add_argument("--tile-polling", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    kernels = (static_roles, cross_group_signal, incore_helper_call, inline_helper_call)
    for name, kernel in zip(names, kernels):
        if args.case and name not in args.case:
            continue
        directory = args.output / name
        try:
            specialization = {"TILE_POLLING": args.tile_polling} if name == "cross_group_signal" else {}
            compiled = kernel.compile(**specialization, config=RunConfig(
                platform="a2a3", save_kernels=True, save_kernels_dir=str(directory), dump_passes=True,
            ))
            report[name] = {"compiled": True, "output_dir": str(compiled.output_dir)}
        except Exception as exc:
            report[name] = {"compiled": False, "error": str(exc), "traceback": traceback.format_exc()}
        (args.output / "compile_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
