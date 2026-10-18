# SPDX-License-Identifier: Apache-2.0
"""使用主线公开IR编译接口作同一kernel的CPU对照，跳过设备执行。"""
import argparse
import json
from pathlib import Path
import pypto
from pypto import passes,pypto_core
from pypto.ir.compile import compile
from repro import slice_entry,extract_entry

p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
report=dict(pypto=pypto.__file__,core=pypto_core.__file__,device=False,results=[])
for name,entry in (("slice",slice_entry),("extract",extract_entry)):
    artifact=compile(entry.specialize(),platform="a2a3",memory_planner=passes.MemoryPlanner.PYPTO,
                     output_dir=str(a.output/name),dump_passes=True)
    report["results"].append(dict(name=name,compiled=True,artifact=str(artifact.output_dir)))
(a.output/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(report,ensure_ascii=False))
