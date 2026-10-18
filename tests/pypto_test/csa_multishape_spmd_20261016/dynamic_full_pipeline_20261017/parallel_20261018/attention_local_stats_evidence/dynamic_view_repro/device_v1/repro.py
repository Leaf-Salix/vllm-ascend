# SPDX-License-Identifier: Apache-2.0
"""动态slice后reshape与显式extract对照；无需模型，CPU编译/队列真机共用。"""
import argparse
import json
import os
from pathlib import Path

import pypto.language as pl


@pl.jit.incore
def slice_reshape_core(
    src: pl.Tensor[[6, 32], pl.FP32],
    row: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
):
    pool = pl.load(src, [0, 0], [6, 32])
    index = pl.cast(pl.read(row, [0]), pl.INDEX)
    selected = pl.tile.slice(pool, [1, 32], [index, 0])
    column = pl.reshape(selected, [32, 1])
    pl.store(column, [0, 0], out)
    return out


@pl.jit.incore
def extract_reshape_core(
    src: pl.Tensor[[6, 32], pl.FP32],
    row: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
):
    pool = pl.load(src, [0, 0], [6, 32])
    index = pl.cast(pl.read(row, [0]), pl.INDEX)
    selected = pl.tile.extract(pool, index, 0, [1, 32], target_memory=pl.MemorySpace.Vec)
    column = pl.reshape(selected, [32, 1])
    pl.store(column, [0, 0], out)
    return out


@pl.jit
def slice_entry(src: pl.Tensor[[6, 32], pl.FP32], row: pl.Tensor[[1], pl.INT32],
                out: pl.Out[pl.Tensor[[32, 1], pl.FP32]]):
    with pl.spmd(1):
        out = slice_reshape_core(src, row, out)
    return out


@pl.jit
def extract_entry(src: pl.Tensor[[6, 32], pl.FP32], row: pl.Tensor[[1], pl.INT32],
                  out: pl.Out[pl.Tensor[[32, 1], pl.FP32]]):
    with pl.spmd(1):
        out = extract_reshape_core(src, row, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import pypto
    from pypto import passes, pypto_core
    from pypto.runtime import CompileOptions
    report = dict(pypto=pypto.__file__,core=pypto_core.__file__,device=args.device,results=[])
    if not args.device:
        for name, entry in (("slice", slice_entry),("extract",extract_entry)):
            artifact = entry._resolve_kernel_artifact((), {"config":CompileOptions(
                platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                output_dir=str(args.output/name),dump_passes=True)}, allow_signature_mode=True)
            report["results"].append(dict(name=name,compiled=True,artifact=str(artifact.output_dir)))
    else:
        import pypto.torch
        import torch
        import torch_npu
        device_id = int(os.environ["TASK_DEVICE"])
        torch.npu.set_device(device_id)
        pypto.torch.init(device=device_id,platform="a2a3",runtime="tensormap_and_ringbuffer")
        device = torch.device(f"npu:{device_id}")
        src = torch.arange(192,dtype=torch.float32,device=device).reshape(6,32)
        row = torch.zeros((1,),dtype=torch.int32,device=device)
        config=CompileOptions(platform="a2a3",memory_planner=passes.MemoryPlanner.PYPTO)
        for name, entry in (("slice",slice_entry),("extract",extract_entry)):
            out = torch.full((32,1),-777.,dtype=torch.float32,device=device)
            for r in (0,1,4,5,1):
                row.fill_(r);out.fill_(-777.)
                entry(src,row,out,config=config);torch.npu.synchronize()
                actual=out.cpu().flatten().tolist();expected=list(range(r*32,(r+1)*32))
                report["results"].append(dict(name=name,row=r,correct=actual==expected,actual=actual,expected=expected))
        report["defect_reproduced"] = any(not x["correct"] for x in report["results"] if x["name"]=="slice")
        report["extract_control_passed"] = all(x["correct"] for x in report["results"] if x["name"]=="extract")
        assert report["extract_control_passed"]
    (args.output/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="results"},ensure_ascii=False))


if __name__ == "__main__":
    main()
