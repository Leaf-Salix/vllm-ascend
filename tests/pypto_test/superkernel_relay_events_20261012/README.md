# 事件不足时通过计算流水中转尾部通知

2026-10-12，过程074。**该项保留为私有TMR功能扩展，没有新增CSA性能结论。**
PyPTO `847e341cc`、PTOAS `6dee5a69`；Simpler `9fddae626`、PTO-ISA及公共算子未改。
仍由作者在原task中标识局部Wait/Set，诊断模式串行回退保持；没有改变dispatch、ACK、核选择或early-resolve。

## 解决的限制

过程072在某个通知pipe pair的八个事件ID都可能被显式同步占用时，尾回调只能阻塞等待。
即使其他pipe pair有空闲ID，也无法在这段时间观察后到的已派发task。

现在优先保留原直接汇合；若直接路径不足，编译器尝试以下中转链：

| 核型 | 中转链 |
| --- | --- |
| AIV | `MTE2 → V → MTE3` |
| AIC | `MTE2 → MTE1 → M → FIX → MTE3` |

每条边分别选择作者未使用的ID。Set/Wait按链逐边发出，Wait在目的流水中执行，
该流水后面的Set继续携带前面所有流水的完成约束。最终MTE3排在这些约束和原有MTE3工作之后再写通知，
Scalar可以继续轮询。作者原来的事件、顺序和数学运算保持不变，不复用仍可能活跃的事件。

对应的常规流水连接在本地固定PTO-ISA的A3示例
`tests/npu/a2a3/src/st/testcase/tpushpop_cv/tpushpop_cv_kernel.cpp` 中已有使用；
中转会经过原计算流水，因此不能单凭指令数假设性能改善。
当前CSA正常路径没有触发这个事件不足回退，无合同的默认回调保持原生成文本；本轮不重复CSA计时。

新合同逐函数携带核型对应的pipe chain及ID，旧包装器编译失败；PyPTO检查核型、链完整性、
ID范围、重复或相互冲突的直接/中转/阻塞合同。显式事件的未知范围保守占用该pair的全部八个ID。
两条现有路径均无足够资源时仍阻塞，不把本轮扩展宣称为所有资源布局都已覆盖。

## 编译与边界检查

- 21项后端UT通过，覆盖默认不变、直接事件分配、中转链次序、核型、错误/冲突合同和阻塞回退。
- 初次8项FileCheck通过；补齐MIX降级形式后，又检查中转合同、Wait前GM读取拒绝和Wait前显式同步拒绝。
- MIX在CPU代码生成通过后才再次提交设备任务。Ruff通过。

MIX补测发现PTOAS尾部验证器只认识降级前的通信名字；再次编译已经降级的合法IR时，
`initialize_l2g2l_pipe`、`declare_tile`、`tpush/tpop/tfree`被拒绝。
已在`runtime_barrier_all`合同下补齐这些形式，推送计入尾部输出，push/pop计入流水工作；
Wait前的独立安全验证没有放宽。首次失败的两份降级输入和错误信息保留为回归证据。

## 单卡功能结果

CANN9.2、TMR、卡0。纯AIC/AIV的直达`MTE2→MTE3`事件ID由block变化覆盖0～7，
编译器为通知选择空闲中转事件。两路分别为原单次探测和新增通知；work=1、Scalar steps=64，
两次输入及图重放均检查输出、Scalar结果、完整fanin读取、输出/trace保护区和本核完成时序。
`repeats=0`，没有采性能样本。

| 核型/输入 | 实际中转通知 | 同核交接 | 其中回调后交接 |
| --- | ---: | ---: | ---: |
| AIV/0 | 48 | 0 | 0 |
| AIV/1 | 44 | 5 | 1 |
| AIC/0 | 24 | 0 | 0 |
| AIC/1 | 24 | 3 | 3 |

图重放AIV通知46次、同核交接6次（其中回调后4次）；AIC通知24次、同核交接3次（均为回调后）。
原单次探测对照不发通知，功能也通过。这证明新回调实际执行且能交接后到任务；
完整尾部包含新增通知DMA，不能把上述交接次数直接解释为原计算重叠次数或收益。

MIX用例保留原自动生成的同步，再在两个物理成员的尾部加入原本未使用的直达pair八个ID，
分别强制验证AIC/AIV中转。两次输入都满足：

| 项目 | 输入0 | 输入1 |
| --- | ---: | ---: |
| 生产组完成前进入安全前缀的AIC | 24/24 | 24/24 |
| 生产组完成前进入安全前缀的AIV | 48/48 | 48/48 |
| 生产者AIC/AIV通知 | 24/0 | 24/0 |
| 消费者AIC/AIV通知 | 24/48 | 24/48 |

数值、保护区、尾部时序检查通过；原MIX整组完成语义保持。
“早于生产组完成”不是每条都与本核旧计算重叠的证明。
[功能摘要](functional_summary.json)、[纯核原始结果](pure/results.json)、
[MIX逐成员trace](mix_verified/mixed_chain_results.json)、[生成的通知回调](compiler/generated_helpers.json)。

## 失败过程与任务

| 任务 | 结果 |
| --- | --- |
| `task_20261012_031553_30011657983` | 纯AIC/AIV通过；MIX在pytest导入辅助模块时失败，未执行MIX；exit1 |
| `task_20261012_031754_30739915482` | 修复包/脚本双入口导入后，MIX编译暴露降级通信形式未支持；exit1 |
| `task_20261012_032119_321007727040` | 编译器修正、CPU代码生成通过后，只重试MIX；两次输入通过，exit0 |

三次均释放设备；未重跑已经通过的纯核用例。
首次日志在`mix/`，编译失败及输入在`mix_retry/`，成功记录在`mix_verified/`。
辅助模块现在兼容pytest包导入和独立脚本入口，没有通过修改全局搜索路径绕过问题。

## 复现与剩余范围

完整版本见[versions.json](versions.json)。增量patch以PTOAS `9ddc2e25`、PyPTO `7a084029d`为基线，
使用`git apply --unidiff-zero`；源码及测试副本在`source/`，不包含编译二进制。
在既有私有环境中重建PTOAS后，CPU执行：

```bash
python tests/st/runtime/scheduling/task_boundary_allocated_event_probe.py \
  --event-id exhausted --compile --work 1 --compiled ../artifacts/relay_event_probe_compiled_20261012
```

这里`exhausted`指原直达pair的八个ID，并非所有中转pair。`repro/`脚本需置于私有根目录的`artifacts/`，
依赖该目录下既有`env.sh`和匹配版本，所有设备运行均通过`task-submit --device auto`提交；
不是新克隆仓库可直接运行的独立软件包。

一般只读GM前缀仍缺跨函数写者及实际别名合同；两种通知路径都不足时，阻塞回调仍不能观察中途派发。
这些是未完成的实现范围。手工Wait和chip swimlane/PMU/参数dump下的串行回退是用户已确认的范围，
不再列作待消除限制。最新128K/B16性能继续引用过程073，没有新增七档、Native、16卡或token结果。
