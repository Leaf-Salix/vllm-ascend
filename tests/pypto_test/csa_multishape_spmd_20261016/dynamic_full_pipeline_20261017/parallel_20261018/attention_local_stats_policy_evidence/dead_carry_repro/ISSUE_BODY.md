### 问题及范围

已无任何数据消费者的Tile状态，经loop carry和if-phi形成一个只传递自身的闭环，
`Simplify/EliminateDeadYieldSlots`仍把循环内的语法引用当作live use。最终MemoryReuse/
AllocateMemoryAddr保留无用片上分配。

这是**死状态消除与资源规划优化问题，不是本例已证明的数值错误**。它使constexpr
关闭的优化仍可能改变完整kernel的UB布局；不能仅从前端开关=False推断关闭成本为零。

关联：#1603已解决无普通用户的dead phi；本例有彼此相互引用的死闭环。
不是#3009：while条件真正读取的iter_arg必须保留，不能为此删除真实条件依赖。

### 复现环境

2026-10-18核对main为`14736f65c04a7db06547edb656f515054950555c`。
以下CPU复现在其派生独立树`72cadb1e5`，另有FloorMod/view/While修正，非干净
main的NPU测试；本问题相关的`utils/dead_code_elimination.cpp`与`simplify_pass.cpp`
未作本地修改。目标a2a3、PyPTO memory planner。不需要模型、权重或设备。

### 完整最小复现

保存为`repro.py`，执行`python repro.py --output /tmp/dead_carry_new_dir`。

```python
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
```

### 实际与预期

编译成功，但最终分配IR仍包含：

```python
ghost: pl.Tile[[6,32], pl.FP32, pl.MemRef(mem_vec_4, 128, 768), pl.Mem.Vec]
for i, (live, unused) in pl.range(count, init_values=(seed, ghost)):
    if i % 2 == 0:
        unused_ready = pl.yield_(unused)
    else:
        unused_ready = pl.yield_(unused)
    result, ghost_result = pl.yield_(live + 1.0, unused_ready)
```

上面IR只缩短SSA名称用于阅读；原样输出由脚本打印并留在passes_dump中。
`ghost/unused/unused_ready/ghost_result`没有参与计算、地址、条件、store或调度元数据，
本例的768B分配应可消除。输出只取决于seed和count。不要按变量名字匹配来修。

实际CSA中，一个仅开关开启才使用的768B统计环，在关闭constexpr后仍以这类闭环
存活。已发现关闭版比原版多保留该分配并改变其它Vec窗口放置，尚未把这个代码差异
宣称为确定的微秒损失，也未通过篡改容量或放宽别名约束来绕过它。

### 原因与建议

`EliminateDeadYieldSlots`当前收集所有语法Var use，任何一条if yield或纯别名赋值
都会维持loop iter_arg存活。固定点删除“零用户”slot不能打破没有外部根的循环。

建议使用真正的live-root到定义的可达性，或先实现保守的pass-through子集：

```text
live roots = 返回值、保留的有效调用/副作用的输入、控制条件、shape/地址/attrs/deps等真实引用
edges:
  普通alias结果 -> 来源
  if返回slot -> 两条分支对应yield
  loop iter_arg -> init值、对应回边yield
  loop返回slot -> 对应yield
从live roots做worklist可达性；删除不可达的纯数据slot、对应yield及随后无用的分配。
```

必须保留现有保护：

- TaskId/调度channel以及attrs/deps等隐式引用，不能只扫算术表达式。
- init/yield里的call或submit副作用；无法证明纯的调用不能因结果不用就删。
- while条件使用的iter_arg、循环后读取的结果、真实跨迭代计算和存储。
- 外层捕获变量和SSA定义域；不要把if局部定义替换到其不支配的使用点。

建议至少覆盖：本例nested-if纯传递、直接alias传递、真正读取/写入该Tile的反例、
循环后读取、while条件读取、TaskId和有副作用yield/init反例。工作量宜按SSA引用图
保持线性/近线性，避免在每个slot上反复扫描完整循环体。

本次先提交问题、复现和建议；**没有提交修改PR、没有添加标签，也没有声称已有修复**。
