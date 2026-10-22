# constexpr分支含manual_scope/parallel时，SSA的branch yield留在scope内部，Simplify失败

## 版本与影响

PyPTO `26e94e05627eb47710dcc214bae1774b8829df8a`，平台a2a3。
纯CPU编译复现，无设备初始化、模型权重或外部算子。未修改PyPTO/PTOAS/Simpler。
HCA按token区间选择O_A生产者时遇到：constexpr分支的一侧直接创建任务，另一侧
在manual_scope内parallel创建多个任务，分支后消费同一Tensor及TaskId数组。

实际错误：

```
pypto.InternalError: control-flow body tail must be YieldStmt when return_vars is non-empty
Check failed: yield_stmt at src/ir/transforms/simplify_pass.cpp:278
```

## 已观察到的IR问题

复现停在第一次Simplify，最后成功dump是`04_after_ConvertToSSA.py`。
其中一个条件已是常量`pl.const(2, INDEX) == pl.const(1, INDEX)`，但仍保留两分支。
简化展示（省略不相关tile计算）：

```python
if const(2) == const(1):
    ...
    deps_phi, tmp_phi = pl.yield_(deps1, tmp1)
else:
    with pl.scope(mode=pl.ScopeMode.MANUAL):
        for group, (deps_iter, tmp_iter) in pl.parallel(2, init_values=(deps0, tmp0)):
            ...
            deps_rv, tmp_rv = pl.yield_(deps_next, tmp_next)
        pl.yield_(deps_rv, tmp_rv)  # 在scope内，且没有绑定到上面的branch phi
# else的直接末尾是ScopeStmt，不是YieldStmt
with pl.spmd(2, deps=[deps_phi]):
    consume(tmp_phi)
```

Simplify的尾部Yield要求与这里的SSA结果不一致。问题在编译阶段，不是算子精度或运行时同步失败。
与已关闭的#1413报错文本相同，但本例没有pipeline、matmul、3D Acc或跨核tfree，
在首次Simplify就失败，因此单独记录这个scope/constexpr触发路径。

## 建议

1. 核对ConvertToSSA处理`IfStmt -> ScopeStmt -> parallel`时的返回值归属。
   外层if分支应具有自己明确的尾部yield，并绑定正确phi名；scope自身的输出也要合法。
2. 如果scope在IR里是调度属性而非值边界，应在SSA/正规化阶段统一这一语义，
   不能只在Simplify中删除assert、随意丢弃yield或穿透scope清理语句。
3. 增加转换后结构验证，在首次产生非规范IR的pass定位；保留manual依赖和TaskId作用域检查。
4. 回归覆盖constexpr两侧、运行时if、单/多生产者、Tensor与TaskId数组同时合流，以及嵌套scope。

当前算子侧工作绕过：把两种生产者分别放在独立inline函数，通过Python构造函数绑定
不同producer，不在这个constexpr分支中嵌manual_scope。完整HCA这样已编译并上卡，
不代表组件问题已修复；没有修改SDK或提交PR。

## 可直接运行的最小复现

`python repro.py --output /absolute/new_directory`。需要已安装的PyPTO编译环境，不占NPU。

```python
# SPDX-License-Identifier: Apache-2.0
"""纯CPU：constexpr分支内manual_scope/parallel的Tensor合流缺少yield。"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions

T = pl.dynamic("T")


@pl.jit.inline(auto_scope=False)
def producer(x: pl.Tensor[[T, 32], pl.FP32], tmp: pl.Tensor[[T, 32], pl.FP32], group: pl.Scalar[pl.INDEX]):
    with pl.spmd(1) as ready:
        row = group * 32 + pl.tile.get_block_idx() * 32
        count = pl.min(32, pl.tensor.dim(x, 0) - row)
        value = pl.load(x, [row, 0], [32, 32], valid_shape=[count, 32])
        pl.store(pl.add(value, 1.0), [row, 0], tmp)
    return tmp, ready


@pl.jit.inline(auto_scope=False)
def branch(x: pl.Tensor[[T, 32], pl.FP32], out: pl.Tensor[[T, 32], pl.FP32], COUNT: pl.constexpr):
    tokens = pl.tensor.dim(x, 0)
    tmp = pl.create_tensor([tokens, 32], dtype=pl.FP32)
    deps = pl.array.create(COUNT, pl.TASK_ID)
    if COUNT == 1:
        tmp, ready = producer(x, tmp, 0)
        deps[0] = ready
    else:
        with pl.manual_scope():
            for group in pl.parallel(COUNT):
                tmp, ready = producer(x, tmp, group)
                deps[group] = ready
    with pl.spmd(COUNT, deps=[deps]):
        row = pl.tile.get_block_idx() * 32
        count = pl.min(32, tokens - row)
        value = pl.load(tmp, [row, 0], [32, 32], valid_shape=[count, 32])
        pl.store(value, [row, 0], out)
    return out


@pl.jit(auto_scope=False)
def entry(x: pl.Tensor[[T, 32], pl.FP32], out: pl.Out[pl.Tensor[[T, 32], pl.FP32]]):
    if pl.tensor.dim(x, 0) <= 32:
        out = branch(x, out, 1)
    else:
        out = branch(x, out, 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"device_execution": False}
    try:
        entry.compile(config=CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                                            dump_passes=True, output_dir=str(args.output / "build")))
        report["compiled"] = True
    except Exception as exc:
        report.update(compiled=False, error=str(exc), exception_type=type(exc).__name__)
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

```
