# SPDX-License-Identifier: Apache-2.0
"""只比较view形状里内嵌tensor.dim与命名Scalar；无需GM子视图或CSA依赖。"""
import json
import sys
import traceback
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.ir.compile import compile as compile_program

ROWS = pl.dynamic("NESTED_VIEW_ROWS")


@pl.jit.incore
def nested_core(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[1, 256], pl.BF16]]):
    view = pl.tensor.view(x, shape=[pl.tensor.dim(x, 0) * 2, 256])
    value = pl.load(view, [0, 0], [1, 256])
    pl.store(value, [0, 0], out)
    return out


@pl.jit.incore
def named_core(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[1, 256], pl.BF16]]):
    rows = pl.tensor.dim(x, 0)
    view = pl.tensor.view(x, shape=[rows * 2, 256])
    value = pl.load(view, [0, 0], [1, 256])
    pl.store(value, [0, 0], out)
    return out


@pl.jit
def nested(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[1, 256], pl.BF16]]):
    with pl.spmd(1):
        out = nested_core(x, out)
    return out


@pl.jit
def named(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[1, 256], pl.BF16]]):
    with pl.spmd(1):
        out = named_core(x, out)
    return out


if __name__ == "__main__":
    folder = Path(sys.argv[1]).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    reports = []
    for name, entry in (("nested", nested), ("named", named)):
        row = dict(case=name, compiled=False, device_execution=False)
        try:
            artifact = compile_program(entry.specialize(), platform="a2a3", output_dir=str(folder/name),
                                       dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)
            row.update(compiled=True, artifact=str(artifact.output_dir))
        except Exception as exc:
            row.update(error=str(exc), traceback=traceback.format_exc())
        reports.append(row)
    (folder/'result.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in x.items() if k!='traceback'} for x in reports],ensure_ascii=False))
    raise SystemExit(0 if all(x['compiled'] for x in reports) else 1)
