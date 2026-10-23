# [Pass Bug] LowerPipelineLoops clones an enclosing Scalar IterArg into an unbound free variable

## 现象与复现版本

在HCA attention的跨query两级FIFO中，外层循环携带的Scalar地址偏移被内层
`pl.pipeline(..., stage=2)`使用时，CPU编译失败。相同body改成普通`pl.range`
则可以生成代码。条件yield并非必要触发条件：直接算术得到偏移也复现。

- PyPTO共享调试快照：`26e94e05627eb47710dcc214bae1774b8829df8a`；
  此为本地集成快照，不冒称在最新upstream main已复现。
- 平台`a2a3`，`passes.MemoryPlanner.PYPTO`，Linux aarch64。
- PTOAS源码`914863ed3d0eb3e92a4433a4a706a86cabf4c91f`。
- **仅CPU编译，不需模型、权重、设备输入或NPU执行。**
- 未修改共享SDK；正式算子试验在地址描述符处临时绕开该组合。

执行：保存下方脚本为`repro_fifo_offset_codegen.py`，在PyPTO环境中运行：

```bash
python repro_fifo_offset_codegen.py --output /tmp/pypto_fifo_repro_unique
```

输出目录应不存在。四组对照的实际结果：

| Scalar来源 | 内层循环 | 结果 |
|---|---|---|
| 条件yield | pipeline(2, stage=2) | PartialCodegenError |
| 直接算术 | pipeline(2, stage=2) | PartialCodegenError |
| 条件yield | range(2) | 编译通过 |
| 直接算术 | range(2) | 编译通过 |

错误：

```text
PTO codegen cannot materialize symbol 'old' in function 'entry_incore_0_aic':
it is not a physical tensor dimension, a scalar parameter, or a loop variable of this kernel.
Check failed: !name.empty() at src/codegen/pto/pto_codegen.cpp:2682
```

`old`实际上是外层range的合法IterArg，并非用户在TensorView中使用的未传参symbol。
当前报错中的TensorView建议不适合此用例。该复现中所有地址的row为0或16、列为0或64，
均在x[32,128]范围内；输出反复覆盖只用于编译复现，没有作为数学性能用例运行。

## pass边界证据与修复建议

实际dump中，`25_after_ExpandMixedKernel`及`31_after_LowerPipelineToSlots`仍是：

```python
for step, (old, new) in pl.range(3, init_values=(previous, recent)):
    # ...
    lhs = pl.tile.load(x, [old, chunk * 64], [16, 64], ...)
```

**`32_after_LowerPipelineLoops`首次变成free variable**：

```python
for step, (old, new) in pl.range(3, init_values=(previous, recent)):
    # 两份pipeline展开体
    lhs0 = pl.tile.load(x, [old_1__FREE_VAR, chunk * 64], [16, 64], ...)
    lhs1 = pl.tile.load(x, [old_2__FREE_VAR, (chunk + 1) * 64], [16, 64], ...)
```

后续Simplify还能消掉已经不再被引用的原IterArg，最后codegen才暴露错误。
因此这不是PTOAS或设备同步问题，也与#3009的while条件依赖遗漏触发点不同。

源码定位：

- `src/ir/transforms/lower_pipeline_loops_pass.cpp:299`，`ReplicateBody`调用
  `DeepClone(body, sub_map, clone_def_vars=true)`；sub_map包含当前pipeline自身的迭代变量。
- `src/ir/transforms/utils/deep_clone_utils.cpp:101`，`VisitExpr_(IterArgPtr)`遇到
  不在map里的IterArg就新建一份；与普通`VarPtr`分支的“external variable return as-is”不同。
  此处外层`old`只有use、没有在clone subtree内定义，仍被克隆。

建议在DeepClone区分**子树内部定义**和**外部捕获的IterArg**。只在进入For/While定义域时
预注册fresh IterArg；表达式访问应优先查映射，外部未映射的IterArg保持原对象。
思路示意（不是已验证补丁，也没有提交PR）：

```cpp
ExprPtr VisitExpr_(const IterArgPtr& op) override {
  if (auto it = expr_map_.find(op.get()); it != expr_map_.end())
    return it->second;
  return op;  // external use; owned definitions must have been registered beforehand
}
// For/While DefField handling must register fresh IterArgs and remap init/type metadata.
```

需同时保留嵌套循环内部定义的fresh SSA，以及init、shape/stride/offset/valid_shape等
类型内表达式的映射。不要只删除codegen检查，也不要按name_hint强绑同名变量。
建议补充回归：外层Scalar IterArg在内层pipeline的GM地址、有效长度和TensorView元数据中使用；
内部嵌套For/While的IterArg仍正确克隆；同名不同作用域不会误绑。

## 完整可复现脚本

```python
# SPDX-License-Identifier: Apache-2.0
"""纯CPU：条件yield经过两层Scalar FIFO后用于GM load偏移。"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions


@pl.jit.inline(auto_scope=False)
def conditional_offset(step: pl.Scalar[pl.INDEX]):
    if step % 2 == 0:
        offset = pl.yield_(pl.cast(0, pl.INDEX))
    else:
        offset = pl.yield_(pl.cast(16, pl.INDEX))
    return offset


@pl.jit.inline(auto_scope=False)
def direct_offset(step: pl.Scalar[pl.INDEX]):
    return step % 2 * 16


def make_entry(offset_for_step):
    @pl.jit(auto_scope=False)
    def entry(x: pl.Tensor[[32, 128], pl.BF16], weight: pl.Tensor[[128, 32], pl.BF16],
              out: pl.Out[pl.Tensor[[16, 32], pl.FP32]], marks: pl.Out[pl.Tensor[[2, 16], pl.INT32]]):
        with pl.spmd(1):
            worker = pl.tile.get_block_idx()
            for query, (previous, recent) in pl.range(
                worker, 2, init_values=(pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX)),
            ):
                for step, (old, new) in pl.range(3, init_values=(previous, recent)):
                    source = offset_for_step(step)
                    if step < 2:
                        saved = pl.yield_(source)
                    else:
                        saved = pl.yield_(new)
                    if step >= 1:
                        for chunk in pl.pipeline(2, stage=2):
                            lhs = pl.load(x, [old, chunk * 64], [16, 64], target_memory=pl.MemorySpace.Mat)
                            rhs = pl.load(weight, [chunk * 64, 0], [64, 32], target_memory=pl.MemorySpace.Mat)
                            product = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
                            pl.store(product, [0, 0], out)
                    old_done, new_done = pl.yield_(new, saved)
                previous_done, recent_done = pl.yield_(old_done, new_done)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                marker = pl.tile.full([1, 16], dtype=pl.INT32, value=1)
                pl.store(marker, [lane, 0], marks)
        return out, marks
    return entry


def make_entry_no_pipeline(offset_for_step):

    @pl.jit(auto_scope=False)
    def entry(x: pl.Tensor[[32, 128], pl.BF16], weight: pl.Tensor[[128, 32], pl.BF16], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]], marks: pl.Out[pl.Tensor[[2, 16], pl.INT32]]):
        with pl.spmd(1):
            worker = pl.tile.get_block_idx()
            for (query, (previous, recent)) in pl.range(worker, 2, init_values=(pl.cast(0, pl.INDEX), pl.cast(0, pl.INDEX))):
                for (step, (old, new)) in pl.range(3, init_values=(previous, recent)):
                    source = offset_for_step(step)
                    if step < 2:
                        saved = pl.yield_(source)
                    else:
                        saved = pl.yield_(new)
                    if step >= 1:
                        for chunk in pl.range(2):
                            lhs = pl.load(x, [old, chunk * 64], [16, 64], target_memory=pl.MemorySpace.Mat)
                            rhs = pl.load(weight, [chunk * 64, 0], [64, 32], target_memory=pl.MemorySpace.Mat)
                            product = pl.matmul(lhs, rhs, out_dtype=pl.FP32)
                            pl.store(product, [0, 0], out)
                    (old_done, new_done) = pl.yield_(new, saved)
                (previous_done, recent_done) = pl.yield_(old_done, new_done)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                marker = pl.tile.full([1, 16], dtype=pl.INT32, value=1)
                pl.store(marker, [lane, 0], marks)
        return (out, marks)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"device_execution": False}
    for name, offset, factory in (("conditional", conditional_offset, make_entry), ("direct", direct_offset, make_entry),
                                  ("range_conditional", conditional_offset, make_entry_no_pipeline),
                                  ("range_direct", direct_offset, make_entry_no_pipeline)):
        try:
            factory(offset).compile(config=CompileOptions(
                platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                dump_passes=True, output_dir=str(args.output / name),
            ))
            report[name] = {"compiled": True}
        except Exception as error:
            report[name] = {"compiled": False, "type": type(error).__name__, "error": str(error)}
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

```
