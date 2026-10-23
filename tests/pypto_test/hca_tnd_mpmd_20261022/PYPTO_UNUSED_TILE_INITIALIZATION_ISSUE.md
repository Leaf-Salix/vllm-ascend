## 问题范围

在HCA算子按编译期策略拆分核体时发现：完全没有使用者的本地Tile初始化仍保留在最终PTO及设备C++中。
下面进一步缩为**没有循环、没有if、没有Tile别名逃逸**的CPU对照。`UnusedVariableCheck`已提示变量未使用，但输出仍包含对应的TCI/TCVT/TEXPANDS调用。

这是死代码消除/生成代码质量问题，本例没有证明数值错误，也没有把C++调用数量直接当作实测微秒损失或最终机器码证据。

关联 #3037：该issue涉及loop carry/if-phi死闭环；本例连循环/phi都不需要，建议分别保留回归覆盖，不能默认根因完全相同。

## 已复现环境

- PyPTO：`26e94e05627eb47710dcc214bae1774b8829df8a`，当前共享验证分支，非未修改的最新main。
- PTOAS：`914863ed3d0eb3e92a4433a4a706a86cabf4c91f`。
- 平台a2a3，`MemoryPlanner.PYPTO`；CANN 9.2 beta2，Python 3.10。
- 仅CPU编译生成PTO和设备C++，不需要模型、权重、NPU或Simpler Worker执行。没有修改共享SDK来复现或绕过。

## 实际结果

| 核体 | TCI | TCVT | TEXPANDS |
|---|---:|---:|---:|
| clean：只有有效load/add/store | 0 | 0 | 0 |
| unused：额外声明无任何消费者的full与arange→cast | 1 | 1 | 1 |
| used：同类Tile真实参与输出 | 1 | 1 | 1 |

`clean`和`unused`的有效输出表达式都严格是`src+1`。`unused`中的两个新Tile既不参与地址/条件/shape，也不传给其他函数、返回、dump或调度属性。预期它们的纯生产链可以消除；`used`是必须保留的反例。

## 完整复现

安装对应PyPTO/工具链并source CANN后，将以下代码保存为`repro_dead_tile_initialization.py`，执行：

```bash
python repro_dead_tile_initialization.py --output /tmp/dead_tile_init_new_dir
```

输出目录须不存在。脚本保存每个核的最终C++路径、相关内在函数调用行及数量；不占设备。

```python
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

```

## 建议的修正方向

1. 在Tile操作仍有明确SSA结果、尚未丢失def-use信息时消除不可达的纯分配/生产链。至少检查`tile.full`、`tile.arange`和`tile.cast`的纯度/副作用分类，以及常量分支折叠后是否重新执行这一步。
2. 只在能证明Tile是本地新建、没有别名逃逸和真实消费者时删除。不能简单删除“返回值没人接”的调用：GM store、修改传入Tile/Tensor的调用、prefetch、同步、submit/task依赖等副作用必须保留。
3. live roots应包含输出、真实写回、控制条件、shape/地址、attrs/deps和显式dump等引用，而非只扫描算术表达式。
4. 回归至少包含本例clean/unused/used三组、编译期分支关闭后的无用户初始化、#3037的循环死闭环，以及有副作用或隐式属性引用的反例。

本地算子当前用分开的源码核体绕过，未修改编译器；此报告不附PR、不添加标签，也不声称已修复。
