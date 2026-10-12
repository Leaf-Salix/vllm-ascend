# 过程106：局部Wait普通分支内联候选，单卡排队

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

任务`task_20261012_151624_31923512827`已提交单卡队列，尚无设备结论。
同卡旧→097→内联Wait→097→旧，固定17×48链、Scalar256、work5/167和全部原数值/保护区检查。
三项原诊断用例会保留trace回调路径，它们不能单独证明无trace内联分支已执行；
固定无trace链用于验证AIV内联分支的数值，若拟通用保留，须再核对无trace AIC/MIX覆盖。
目前仅离线编译通过，不宣称功能、性能或295/250μs目标完成。

[驱动](repro/task_boundary_inline_wait_probe.py)、[运行脚本](repro/run.sh)。
私有产物位于`artifacts/boundary_inline_wait_20261012/`，依赖acceptance驱动和097冻结runtime。
