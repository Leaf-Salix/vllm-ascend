# SPDX-License-Identifier: Apache-2.0
"""CPU对照：没有任何用户的Tile初始化是否仍进入最终设备C++。不运行NPU。"""

import argparse
import json
import os
from pathlib import Path
import re

import pypto
import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions


@pl.jit(auto_scope=False)
def clean(src: pl.Tensor[[1, 32], pl.FP32], out: pl.Out[pl.Tensor[[1, 32], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="clean_core"):
        value = pl.load(src, [0, 0], [1, 32])
        pl.store(pl.add(value, 1.0), [0, 0], out)
    return out


@pl.jit(auto_scope=False)
def unused(src: pl.Tensor[[1, 32], pl.FP32], out: pl.Out[pl.Tensor[[1, 32], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="unused_core"):
        dead_grid = pl.tile.full([4, 32], dtype=pl.FP32, value=0.0)
        dead_columns = pl.cast(pl.tile.arange(0, [1, 32], dtype=pl.INT32), pl.FP32)
        value = pl.load(src, [0, 0], [1, 32])
        pl.store(pl.add(value, 1.0), [0, 0], out)
    return out


@pl.jit(auto_scope=False)
def used(src: pl.Tensor[[1, 32], pl.FP32], out: pl.Out[pl.Tensor[[1, 32], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="used_core"):
        live_grid = pl.tile.full([4, 32], dtype=pl.FP32, value=2.0)
        live_columns = pl.cast(pl.tile.arange(0, [1, 32], dtype=pl.INT32), pl.FP32)
        value = pl.load(src, [0, 0], [1, 32])
        combined = pl.add(pl.add(value, live_columns), live_grid[0:1, :])
        pl.store(pl.add(combined, 1.0), [0, 0], out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(args.output.resolve() / "build")
    os.environ["PYPTO_CACHE_DIR"] = str(args.output.resolve() / "cache")
    config = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
    report = {"pypto": pypto.__file__, "shared_toolchain": os.environ.get("CSA_SHARED_TOOLCHAIN_RELEASE"),
              "device_execution": False, "scope": "生成PTO及设备C++，不是设备时间或最终机器码归因", "cases": {}}
    for name, entry in (("clean", clean), ("unused", unused), ("used", used)):
        entry.compile(config=config)
        cpp = next((args.output / "build").glob(f"{name}_*/kernels/aiv/{name}_core.cpp"))
        text = cpp.read_text()
        report["cases"][name] = {"cpp": str(cpp.resolve()), "bytes": cpp.stat().st_size,
            "TCI_calls": len(re.findall(r"\bTCI<", text)), "TCVT_calls": len(re.findall(r"\bTCVT\(", text)),
            "TEXPANDS_calls": len(re.findall(r"\bTEXPANDS\(", text)),
            "dead_name_lines": [line.strip() for line in text.splitlines() if "dead_" in line],
            "intrinsic_lines": [line.strip() for line in text.splitlines()
                                if any(op in line for op in ("TCI<", "TCVT(", "TEXPANDS("))]}
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
