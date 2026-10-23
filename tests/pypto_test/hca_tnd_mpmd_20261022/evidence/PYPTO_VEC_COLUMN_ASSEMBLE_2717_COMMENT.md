补充2026-10-23复现：HCA TND attention的H16归约结果插入H32宽的UB环时，同类列assemble仍失败。

复现环境为本地已验证共享快照PyPTO `26e94e05627eb47710dcc214bae1774b8829df8a`、
PTOAS `914863ed3d0eb3e92a4433a4a706a86cabf4c91f`，a2a3 / MemoryPlanner.PYPTO。
这不是最新upstream main的验收结论；共享SDK没有因为此问题被修改。

下面是可独立CPU运行的缩减：只需PyPTO及PTOAS，无模型、权重或NPU执行。
`python repro_reduction_ring_insert.py --output /tmp/reduction_ring_unique`（目录应不存在）。

- ring=[3,32]，每次在列0/16插入[1,16]：PartialCodegenError，PTOAS报
  `pto.tmov op expects A2/A3 non-mat tmov to use matching src/dst shapes`。
- 同样96个FP32的环改成[6,16]，每次在两相邻行插入[1,16]：编译通过。

前端允许这项assemble；生成PTO在表面上是源/目标均[1,16]的TMOV，目标为[3,32]底层buffer的列subview。
因此建议继续核对PTOAS subview规范化后继承的物理row stride/cols与TMOV契约，或由PyPTO
对A3 Vec列插入选择合法的TINSERT/矩形复制lowering，不建议直接删除TMOV形状检查。
当前算子用[6,16]环绕开，逻辑head顺序和总字节数不变。

本报告与#3017的PTOAS planner行/列carry handle错配不同：这里是PYPTO planner，
目标列subview；和本issue的原始症状及生成模式一致，因此补在原issue，不另建重复issue。

```python
# SPDX-License-Identifier: Apache-2.0
"""纯CPU核对H16归约插入H32环的A3 lowering；两种布局的逻辑数据次序一致。"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions


def make_entry(columns):
    ROWS = 3 * 32 // columns
    ROW_STEP = 32 // columns
    HEAD_ROW_STEP = 1 if columns == 16 else 0
    HEAD_COL_STEP = 16 if columns == 32 else 0
    COLS = columns

    @pl.jit(auto_scope=False)
    def entry(x: pl.Tensor[[32, 128], pl.FP32], out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]]):
        with pl.spmd(1):
            worker = pl.tile.get_block_idx()
            seed = pl.tile.full([ROWS, COLS], dtype=pl.FP32, value=0.0)
            for step, (old,) in pl.range(worker, 2, init_values=(seed,)):
                for head, (ring,) in pl.range(2, init_values=(old,)):
                    score = pl.load(x, [head * 16, 0], [16, 128])
                    tmp = pl.create_tile([16, 128], dtype=pl.FP32)
                    maximum = pl.row_max(score, tmp)
                    updated = pl.tile.assemble(ring, pl.reshape(maximum, [1, 16]),
                                               [step * ROW_STEP + head * HEAD_ROW_STEP, head * HEAD_COL_STEP])
                    head_done = pl.yield_(updated)
                step_done = pl.yield_(head_done)
            pl.store(step_done, [0, 0], out)
        return out

    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {"device_execution": False}
    for columns in (32, 16):
        try:
            make_entry(columns).compile(config=CompileOptions(
                platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                dump_passes=True, output_dir=str(args.output / str(columns)),
            ))
            results[str(columns)] = {"compiled": True}
        except Exception as error:
            results[str(columns)] = {"compiled": False, "type": type(error).__name__, "error": str(error)}
    (args.output / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

```

生成PTO的关键片段：

```mlir
      %assemble_view = pto.subview %seed__ssa_v0[%step__idx_v0, %2] sizes [1, 16] valid [%c1_index, %c16_index] : !pto.tile_buf<loc=vec, dtype=f32, rows=3, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16, v_row=1, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tmov ins(%t__tmp_v0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%assemble_view : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16, v_row=1, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```
