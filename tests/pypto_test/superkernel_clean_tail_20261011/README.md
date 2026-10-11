# 过程059：移除每次Set的GM写入，128K/B16同SDK四路对照

日期：2026-10-11。接续[过程058](../superkernel_runtime_late_poll_20261011/README.md)。
单卡A3、CANN9.2、TMR；局部Wait/Set接口、原early-resolve、派发与ACK流程不变。
不做自动前缀识别；chip swimlane、PMU和参数dump继续回退。公共默认及其他会话环境未改。

本轮减少了可由编译元数据替代的控制状态写入，但尚未消除后到通知成本。
新方案较旧写入版mean少0.48μs、max少7.40μs；较原单次探测版mean仍多3.56μs。
前一项均值变化太小，不能声称稳定加速。保留功能与代码清理，继续把通知成本和有效重叠分别评估。

## 修改与相同SDK对照

- Simpler `5085a6ce0`：在原CoreTaskArgs/TaskPayload能力中增加逐成员回调掩码，
  只有该成员同时具有runtime tail和回调能力才选择`RuntimeCompiledPollClean=8`。
  本核原Set不写GM，正常无诊断收尾不再为这些写入flush控制缓存行。
  诊断写入继续flush；旧`RuntimeCompiledPoll=7`保留原写回，供同SDK隔离对照。
  没有新增dispatch触发、提前ACK、修改核选择或更改用户early-resolve。
- PyPTO `207741212`：单核和MIX按已有局部Set及分配器scratch属性生成上述能力，
  删除每个Set写scratch与tail kind的adapter。尾部回调、通知DMA、首次探测和完整等待不变。
- 私有vLLM测试驱动 `16e3d7c5`：四路共享冻结算子包和同SDK，分别生成普通、单次探测、
  旧Set写入、新能力元数据。旧写入路径只在测试wrapper恢复过程058的两次GM写入。
- PTOAS仍`c63b7d28`，本轮未改。scratch和event7预留、每核通知区及其冷初始化也不变。

精确版本见[versions.json](versions.json)。补丁从过程058最终版本起，使用零上下文，应用时
需`--unidiff-zero`；runtime gitlink按Simpler版本单独同步。没有额外的算子作者注解。

## 功能验证

12项调度CPU测试通过，新增覆盖AIC/AIV0/AIV1成员能力隔离、无尾部不允许回调、
槽位复用时清零诊断位和不改变early-resolve。8项相关编排生成检查通过。
初次编排检查把scratch的实际int属性按int64读取而失败，已按实际类型修正后通过。
测试驱动另有一次缩进问题被Ruff捕获并修正，未进入设备计算。

任务`task_20261011_211954_169757631100`，卡0，退出0并释放设备：

- AIC/AIV，各两种计算长度、两组输入；逐bit输出、fanin实际读取值、保护区与图重放通过。
- 原首次探测后的后到交接仍有效。下表只列新路径，各单元为“总handoff / 后到handoff / 进入轮询数”。
- MIX两项检查通过，含跨核FIFO、sync_start、成员分组完成和后继等待。

| 类型 | loops | 输入0 | 输入1 | 图重放 |
| --- | ---: | --- | --- | --- |
| AIV | 1 | 0 / 0 / 48 | 1 / 0 / 47 | 0 / 0 / 48 |
| AIV | 4 | 11 / 1 / 38 | 15 / 2 / 35 | 13 / 0 / 35 |
| AIC | 1 | 1 / 1 / 24 | 16 / 16 / 24 | 11 / 11 / 24 |
| AIC | 4 | 21 / 21 / 24 | 23 / 23 / 24 | 22 / 22 / 24 |

不同轮次到达时序并不相同，不把本表与过程058计数相减作为优化效果。
trace的完整尾等待包括新增通知DMA，独立原计算重叠的证据仍见过程057。
此处没有再次增加人工Scalar循环或改变被测task的计算。

## 128K/B16 CSA四路性能

任务`task_20261011_212234_18104924066`，卡0，退出0并释放设备。
TMR、NZ2、atomic0、deterministic level0、诊断关闭；第2层真实权重、128K合成历史、B16×verify6。
测HC_pre+norm+CSA+HC_post的单次完整NPUGraph重放，NPU events，状态reset在区间外。
每路预热5次，轮换各30次，共120条样本。

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 普通版 baseline | 971.72 | 988.50 | 1004.24 |
| 原单次探测 once | 966.30 | 981.03 | 998.76 |
| 后到通知，旧GM写入 written | 960.60 | 985.07 | 1011.42 |
| 后到通知，新能力元数据 mixed | 962.60 | 984.59 | 1004.02 |

once、written与mixed使用同一MIX标记包、scratch和事件预留。once在生成时同时关闭回调入口
和新提交能力，保留原单次探测；written关闭新提交能力并恢复旧Set写入，故仍走协议7的flush。
mixed使用协议8。普通版没有局部标记，四路都包含本SDK通知区冷初始化。

新元数据版相对：

| 对照 | mean变化（μs） | max变化（μs） | 30组配对更快数 |
| --- | ---: | ---: | ---: |
| 普通版 | −3.91（−0.40%） | −0.22 | 19 |
| 单次探测 | +3.56（+0.36%） | +5.26 | 12 |
| 旧GM写入 | −0.48（−0.05%） | −7.40 | 14 |

旧写入→新元数据的配对差值中位数为+2.99μs，均值改善并不普遍；max下降也是这批样本结果，
尚不是稳定的多卡长尾结论。一次清理不足以覆盖通知DMA、回调和轮询的剩余成本。
较普通版的0.40%均值改善也只能按本轮记录；不能从跨卡、跨SDK或跨轮绝对时间差推断因果。

9项输出/缓存/状态、改变输入后的图重放与普通PTO逐bit一致，每路31项保护区通过。
这不是新的Native数值或整模型token验收，没有16卡任务或七档重测。
[完整结果](runs/csa_clean_tail_128k_b16_20261011/results.json)、[120条计时](csa_samples.csv)、
[摘要](summary.json)均保留。

首次CSA任务`task_20261011_212129_177068210748`在卡0因env加载前不在Git目录退出128，
尚未执行设备计算。已将脚本的cd前移后完成上述任务；失败脚本和日志保留在本轮runs中。

## 复现与后续

源码增量及中文提交说明见各仓patch与commits文件。`source/`保留新驱动，基础fixture沿用过程058；
`generated/`保留新AIC/AIV入口与编排。构建和单卡运行脚本在`repro/`，先进入Git目录再加载env。
私有四路完整产物位于：

```text
/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/artifacts/csa_clean_tail_128k_b16_20261011/compiled/
```

后续优先把剩余成本拆成回调进入、完成通知、等待轮询，结合真实CSA后到命中判断有效重叠。
继续使用128K/B16同SDK对照；未经新的证据，不把所有early_dispatch计为有收益的重叠，
也不把数据未ready或无剩余本核计算时的等待当成新增资格限制。公共默认尚未切换。
