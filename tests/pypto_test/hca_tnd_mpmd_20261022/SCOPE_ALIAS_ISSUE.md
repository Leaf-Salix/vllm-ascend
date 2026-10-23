## 问题与范围

在 HCA TND 中间值诊断中，父作用域创建的 manual_dep Tensor，经 `pl.scope()` 内的
inline helper原地写入并返回同一Tensor别名后，在scope外消费这个别名，会生成引用
未声明局部变量的编排C++。数据本体在父作用域分配；依赖通过父作用域的
`Array[1, TASK_ID]`保存，没有把局部TaskId直接带出已关闭scope。

独立复现的结果：两版IR/PTOAS lowering都通过；outside编排C++编译失败，
inside（把同一个采集任务留在生产者scope内）完整编译通过。
只做CPU编译，没有NPU执行，不涉及模型权重，也没有修改SDK。

## 版本

- PyPTO本地kernel调试分支26e94e05627eb47710dcc214bae1774b8829df8a，含之前CSA兼容修复。
  尚未在干净最新main重复验证，不声称所有分支均复现。
- PTOAS 914863ed3d0eb3e92a4433a4a706a86cabf4c91f。
- Simpler 57e499238fd4ee3548947ad860be456aaeeb16b5，A3 TMR runtime。
- 错误是PyPTO生成的C++词法作用域错误，不是PTOAS内存规划或设备同步超时。

## 运行方式

```bash
python repro_scope_alias.py --output /tmp/hca_scope_new_output
```

目录必须不存在。此脚本同时完成kernel lowering与编排C++编译，不初始化Worker、不占NPU。

```python
# SPDX-License-Identifier: Apache-2.0
"""纯CPU检查：scope内回传的manual_dep别名在外部消费时是否生成未声明SSA。"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import passes
from pypto.runtime import CompileOptions
from pypto.runtime.kernel_compiler import KernelCompiler


@pl.jit.inline(auto_scope=False)
def produce(x: pl.Tensor[[16, 32], pl.FP32], q: pl.Tensor[[16, 32], pl.FP32]):
    with pl.spmd(1) as ready:
        row = pl.tile.get_block_idx() * 16
        q[row:row + 16, :] = pl.add(x[row:row + 16, :], 1.0)
    return q, ready


@pl.jit(auto_scope=False)
def outside(x: pl.Tensor[[16, 32], pl.FP32], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            captured: pl.Out[pl.Tensor[[16, 32], pl.FP32]]):
    q = pl.create_tensor([16, 32], dtype=pl.FP32, manual_dep=True)
    signals = pl.array.create(1, pl.TASK_ID)
    signals[0] = pl.system.task_invalid()
    with pl.scope():
        q, ready = produce(x, q)
        signals[0] = ready
        with pl.spmd(1, deps=[signals]):
            row = pl.tile.get_block_idx() * 16
            out[row:row + 16, :] = pl.mul(q[row:row + 16, :], 2.0)
    with pl.spmd(1, deps=[signals]):
        row = pl.tile.get_block_idx() * 16
        captured[row:row + 16, :] = q[row:row + 16, :]
    return out


@pl.jit(auto_scope=False)
def inside(x: pl.Tensor[[16, 32], pl.FP32], out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
           captured: pl.Out[pl.Tensor[[16, 32], pl.FP32]]):
    q = pl.create_tensor([16, 32], dtype=pl.FP32, manual_dep=True)
    signals = pl.array.create(1, pl.TASK_ID)
    signals[0] = pl.system.task_invalid()
    with pl.scope():
        q, ready = produce(x, q)
        signals[0] = ready
        with pl.spmd(1, deps=[signals]):
            row = pl.tile.get_block_idx() * 16
            out[row:row + 16, :] = pl.mul(q[row:row + 16, :], 2.0)
        with pl.spmd(1, deps=[signals]):
            row = pl.tile.get_block_idx() * 16
            captured[row:row + 16, :] = q[row:row + 16, :]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    compiler = KernelCompiler(platform="a2a3")
    result = {"device_execution": False}
    for name, entry in (("outside", outside), ("inside", inside)):
        directory = args.output / name
        row = {"lowering": False, "orchestration": False}
        try:
            entry.compile(config=CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                                                output_dir=str(directory), dump_passes=True))
            row["lowering"] = True
            sources = list(directory.glob("orchestration/*.cpp"))
            if not sources:
                raise AssertionError("未生成编排C++")
            for source in sources:
                compiler.compile_orchestration("tensormap_and_ringbuffer", str(source))
            row["orchestration"] = True
        except Exception as error:
            row["error"] = str(error)
        result[name] = row
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

## 实际生成代码

简化片段如下，原始生成的Tensor名是`q__ssa_v2`：

```cpp
const Tensor& q = alloc_0.get_ref(0);       // 父scope分配
TaskId signals[1];
SIMPLER_SCOPE() {
    // producer写q；ready写到父scope的signals[0]
    Tensor q__ssa_v2 = q;                  // alias声明在子scope
    signals[0] = ready;
    // scope内consumer使用q__ssa_v2
}
CoreTaskArgs params_t2;
params_t2.add_input(q__ssa_v2);             // error: was not declared in this scope
```

原始报错：

```text
outside/orchestration/outside.cpp:65:25:
error: 'q__ssa_v2' was not declared in this scope
```

## 修复建议

1. 编排codegen需要对跨scope活跃的Tensor别名选择可支配所有使用点的C++声明位置，
   或对已证明为同一个外部/父scope Tensor的别名复用原描述符。
2. 如果返回的是切片/reshape，需保留其shape/stride/offset，不能无条件替换成父Tensor。
3. 不要删除scope或放宽局部TaskId逃逸检查。本复现已使用文档建议的Array保存依赖，
   应修复Tensor别名的声明生命周期，而不是跳过真实ordering edge。
4. 如果该Tensor数据流确实不被DSL支持，应在IR阶段明确拒绝并给出可操作诊断，
   不应让`.compile()`成功后，到实际注册/运行时才发现编排C++编译失败。
5. 回归覆盖父Tensor原地返回、返回view、manual_dep开关、嵌套scope及通过Array传依赖。

本地诊断暂把采集任务移入生产者scope解决，正式HCA算子未改。仅提issue和建议，不提PR。
