# 过程086：原有完整屏障之后复用已消费的通知事件

2026-10-12。功能扩展保留；没有新增 CSA 性能数据。

用户确认的接口范围继续保持：由局部 `wait_pre_task_end()` 标识安全 Scalar 前缀，
不自动寻找边界；chip swimlane、PMU、参数 dump 开启时仍串行回退。
不改原调度、核选择、ACK 或 `allow_early_resolve`。

## 问题及处理

显式同步函数过去把“曾使用的 ID”一直视为占用，即使它已经经过完整消费屏障。
若直接和中转通知都找不到 ID，第一次交接探测失败后就会进入阻塞回调，
无法继续观察等待期间才派来的 task。

不能只按源码中 Set/Wait 配对就释放 ID。Set 不等待目的流水消费；
Wait 真正执行时才清除标志，重复 Set 可能引发未定义行为。
依据是 Ascend 官方的 [SetFlag/WaitFlag](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.html)
及 [PIPE_ALL](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier_ISASI.html) 说明。

私有 PTOAS `fdebb38b` 在单基本块中按顺序记录顶层静态事件，仅当其 Set/Wait 已配对，
并经过**源码原有且会保留的 PIPE_ALL** 后，才允许通知复用其 ID。之后的新使用重新占用。
不增加设备指令或屏障，不改作者事件。动态、嵌套、隐藏和不完整使用仍保守；
带 `pto.auto_sync_tail_barrier` 的尾部屏障可被延后，不能作为释放依据。
扫描和固定九组事件状态的更新为线性复杂度。

改动：PTOAS 三文件 +217/−9 行，其中生产 C++ +43/−9 行；
PyPTO `a330b3eb7` 仅新增 74 行测试，生产代码和 SDK 不变。
Simpler 仍为 `7043f8c1b`，PTO-ISA、vLLM 算子及公共默认均未改。

## 验证及实际覆盖

PTOAS 增量构建通过。新增 FileCheck 覆盖八类输入：完整消费、
屏障后重新使用、仅配对、单流水屏障、可延后尾屏障、未配对、嵌套、动态。
另外五个原显式分配/高层事件/动态范围/耗尽/中转 FileCheck 通过。
生成 C++ 保留原屏障与事件；不安全情况仍保留其 ID。

CPU 编译首轮 MIX 通过，AIC/AIV 均在编译完成后被测试夹具误报：
夹具漏计了普通后继也会生成的 helper，预期函数集合已修正。
只有编译检查失败，没有设备超时或失败重跑；原日志保留。

单卡任务 `task_20261012_080445_122269722736`，设备 0，CANN 9.2，TMR，
三个用例通过，退出 0，已释放卡。每类核均使用两次不同输入：

| 用例 | 输入 1 | 输入 2 | 说明 |
| --- | ---: | ---: | --- |
| 纯 AIC 同核完整尾部交接 | 24/24 | 24/24 | 均走立即交接，没有执行通知回调 |
| 纯 AIV 同核完整尾部交接 | 48/48 | 48/48 | 均走立即交接，没有执行通知回调 |
| MIX 前驱组结束前进入前缀 | 24 AIC / 48 AIV | 24 AIC / 48 AIV | 不等价于逐成员原始计算重叠 |
| MIX 通知次数：生产 AIC/AIV、消费 AIC/AIV | 24/0/24/48 | 24/0/24/48 | 两种物理核都实际执行复用事件的回调 |

输出、Scalar 值、保护区、本核完整尾部和组级等待检查通过。
测试构造的事件和已有屏障位于 Wait 后、原计算之前，覆盖资源占用及释放；
不是在生产编译路径中为释放 ID 新增屏障。
设备结束后补充的通知计数断言已对这两组原始样本离线执行，未重复占卡。

当前 CSA 的通知成员未触发资源耗尽，自动同步路径本轮也未改，
因此没有重跑 CSA、Native、七档、token 或 16 卡；不将这些插桩结果当性能成绩。
一般 GM 读取的依赖/别名证明、循环局部边界及无法证明消费的资源耗尽情况仍未解决。

## 产物及复现

- [版本与任务](versions.json)、[结果摘要](summary.json)。
- [PTOAS 补丁](ptoas-src.patch)、[PyPTO 测试补丁](pypto.patch)。
- `checks/`：构建、首次 CPU 编译失败及离线通知断言；`compiler/`：FileCheck 输入对应的实际生成结果。
- `device/`：最终测试日志、原始 trace、手工同步 IR 和实际 kernel 源码。
- `repro/`：测试及编译器合同；[单卡命令](repro/run.sh) 使用私有环境与 task-submit 分配的设备。

在记录版本上应用补丁并重建 PTOAS，以资源限制配置编译。新编译检查命令在
[IR 回归](repro/compiler.pto) 的 RUN 行；设备用例为
`tests/st/runtime/scheduling/test_local_task_boundary_drained_events.py`。
正常 runtime/torch 桥接沿用 Simpler `7043f8c1b`，无需重新生成 SDK。
