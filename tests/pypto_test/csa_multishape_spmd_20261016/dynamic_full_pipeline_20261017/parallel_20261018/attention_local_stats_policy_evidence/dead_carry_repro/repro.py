# SPDX-License-Identifier: Apache-2.0
"""未使用的Tile循环状态闭环最小CPU复现，不运行设备。"""
import argparse
import json
from pathlib import Path
import pypto
import pypto.language as pl
from pypto import passes,pypto_core
from pypto.ir.compile import compile


@pl.jit.incore
def carry_core(src: pl.Tensor[[1,32],pl.FP32], steps: pl.Tensor[[1],pl.INT32],
               out: pl.Out[pl.Tensor[[1,32],pl.FP32]]):
    seed = pl.load(src,[0,0],[1,32])
    ghost = pl.create_tile([6,32],dtype=pl.FP32)
    count = pl.cast(pl.read(steps,[0]),pl.INDEX)
    for i,(live,unused) in pl.range(count,init_values=(seed,ghost)):
        if i % 2 == 0:
            unused_ready = pl.yield_(unused)
        else:
            unused_ready = pl.yield_(unused)
        updated = pl.add(live,1.0)
        result,ghost_result = pl.yield_(updated,unused_ready)
    pl.store(result,[0,0],out)
    return out


@pl.jit
def entry(src: pl.Tensor[[1,32],pl.FP32],steps: pl.Tensor[[1],pl.INT32],
          out: pl.Out[pl.Tensor[[1,32],pl.FP32]]):
    with pl.spmd(1):
        out=carry_core(src,steps,out)
    return out


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    artifact=compile(entry.specialize(),platform="a2a3",memory_planner=passes.MemoryPlanner.PYPTO,
                     output_dir=str(a.output/"kernel"),dump_passes=True)
    files=list((Path(artifact.output_dir)/"passes_dump").glob("*after_AllocateMemoryAddr.py"))
    assert len(files)==1
    text=files[0].read_text()
    report=dict(pypto=pypto.__file__,core=pypto_core.__file__,compiled=True,device=False,
                allocated_ir=str(files[0]),ghost_retained="ghost" in text,
                evidence=[line for line in text.splitlines() if "ghost" in line or "unused" in line])
    (a.output/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))
