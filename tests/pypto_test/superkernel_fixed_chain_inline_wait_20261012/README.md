# 过程106：内联Wait完成单卡对比，收益不明确，暂不保留

2026-10-12。针对has_predecessor的普通tagged调用，在编译适配层直接执行现有Wait协议，
测试是否减少间接回调和寄存器保存的开销。没有预设这些成本就是当前主要GAP。

只接受无trace且前驱为空，或无trace的RuntimeBarrierAll/RuntimeCompiledPollClean/RuntimeDirectPollClean前驱。
顺序与原回调相同：排空本核全部流水→清理当前控制行→原前驱FIN→完整输入门铃及EXIT检查→当前ACK。
旧协议、非clean前驱或诊断仍调用原runtime函数；向量mask/atomic状态仍由既有适配器在Wait之后初始化。
不改变派发、选核、ACK之后流程、early-resolve、原算术或完整后到通知。

此为私有编译进程内的适配器替换，不改活动PyPTO或Simpler源码、不替换公共安装。
Runtime继续用097冻结二进制；plain直接使用正常编译器生成的原plain产物。
首次编译因内核缺少平台头搜索路径失败；增加该进程编译器的架构include路径后通过，未更改任何常量。
[源码比较](source_comparison.json)表明算子Python、PTOAS计算C++与orchestration相同；PTO IR仅源码位置路径不同。
新适配器在[wait_adapter.cpp](wait_adapter.cpp)，不把它混作用户计算代码变化。

原任务`task_20261012_151624_31923512827`在候选加载阶段失败，后续修正及补测见下文。
同卡旧→097→内联Wait→097→旧，固定17×48链、Scalar256、work5/167和全部原数值/保护区检查。
三项原诊断用例会保留trace回调路径，它们不能单独证明无trace内联分支已执行；
固定无trace链用于验证AIV内联分支的数值，若拟通用保留，须再核对无trace AIC/MIX覆盖。
下述完整固定链和三项诊断已通过，但不足以声明通用内联分支验收或295/250μs目标完成。

[驱动](repro/task_boundary_inline_wait_probe.py)、[运行脚本](repro/run.sh)。
私有产物位于`artifacts/boundary_inline_wait_20261012/`，依赖acceptance驱动和097冻结runtime。

## 加载失败与对照异常

- 原151624卡0：plain产物使用软链接，加载器拒绝非实体祖先目录；候选未执行，无性能数据。
- 将plain改为同工作区内原产物的实体副本。加载阶段也会核对编译输入，因此编译与run进程都需安装相同的架构include路径；两种产物CPU加载预检通过。
- 补测160613卡9：在缓存097对照的5μs/17级换输入图重放同步处报507011，CANN另报AICPU errorCode=0x91；内联候选未执行。这与此前507018的根因是否相同未知。
- 原驱动在finally中先reset graph，二次清理异常阻止results.json落盘，导致这一失败进程的内存样本未保存。不能补造这些数据。驱动已改为先保存现有样本/原异常，再清理并记录清理异常；不改计算及计时区间。

## 完整补测

任务`task_20261012_160920_183437712935`卡8退出0，顺序旧→内联→缓存097→旧。
保持固定17×48、Scalar256、work5/167、8次图内调用、30样本和全部精确检查。
min/max是30个八调用均值的极值，不是单次调用的极值。

| 运行 | 5μs plain min/mean/max | 5μs early min/mean/max | 50μs early min/mean/max |
| --- | --- | --- | --- |
| 旧SDK前 | 283.97 / 288.24 / 291.80 | — | — |
| 内联Wait | 285.44 / 290.92 / 296.02 | 301.27 / 305.72 / 310.20 | 998.89 / 1002.35 / 1005.55 |
| 缓存097后 | 285.66 / 291.36 / 296.01 | 303.22 / 306.66 / 309.97 | 1000.91 / 1004.41 / 1009.25 |
| 旧SDK后 | 286.78 / 291.68 / 297.85 | — | — |

短档比紧邻缓存对照均值少0.94μs（0.31%），长档少2.06μs（0.21%）。旧SDK前后均值相差3.44μs，
不能把这个小差值视为已证实的稳定收益；候选暂不保留、不修改公共安装。
短档标记305.72μs相对旧SDK前后中点289.96μs仍慢约5.43%；250μs目标未达到。
三项AIC/AIV/MIX trace诊断通过，它们保留回调路径；无trace固定链覆盖AIV内联路径，
不冒充已验证无trace AIC/MIX通用覆盖。
[全量样本及错误](device/)、[摘要](device_summary.json)、[重跑脚本](repro/run_boundary_inline_wait_20261012_retry3.sh)。
