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
