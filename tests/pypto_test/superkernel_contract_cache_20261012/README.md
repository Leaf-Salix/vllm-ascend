# 过程097：提交时缓存交接配置，设备结果待回收

2026-10-12。每个任务的 tail kind、prefix 能力与 scratch 配置不随 SPMD block 改变。
本候选在提交且尚未发布时计算三种成员的配置，放入 TaskPayload 原有 predicate 行的 padding；
保持整个结构大小、张量/标量偏移、每核实际前驱判定、fan-in、原派发及 ACK 流程。
所有未知/legacy/带标记入口在发布前仍初始化消费状态，普通入口采用编译器证明后的短路径。
没有更改 17×48 AIV、Scalar256 轮、Vector5/167 轮、图内8次调用及30样本计时方式。

19项CPU检查通过，完整SDK已隔离编译、生产源码恢复。
任务 `task_20261012_122443_40075121781` 排队中，没有设备性能或功能结论。
该任务使用正常编译器生成的普通入口证明，顺序为旧SDK前测→候选两路→旧SDK后测，
随后运行 AIC/AIV/MIX 三项功能检查。

[候选补丁](candidate.patch)、[CPU结果](cpu_results.xml)、[CPU日志](cpu.log)、
[构建日志](build.log)、[复现脚本](repro/run.sh)、[提交记录](submission.json)。

CPU机器码中初始化辅助函数从过程096的536字节降为180字节；这只是静态证据，
不能据此声称设备时间下降。目标仍是普通链 mean<295μs、标记链 mean<250μs。
