# 128K/B16 单 MIX：进度协议诊断

记录日期：2026-10-16。仅针对当前静态 history=131072、B16、每请求 6 query
token（T96）。保留完整静态源码和全部失败快照，不更改数学分片，不扩展档位。

## 已确认的事实

完整 native 任务 `task_20261016_155540_123832232627` 已完成 CCE/调度库编译，
单个 24 组 MIX 已在设备运行，但未完成。Simpler 报告 `S1:running-stalled`：
completed=0/1、running=1、orch_done=1，最终返回 507018。
没有完整静态输出、精度或性能结果。

为定位停滞，`diagnose_static_protocol.py` 从完整 B16 源码抽取控制段，保留
records/waits、核号映射、逐核顺序、代际以及原 cacheinvalid/fence 操作。
计算段按已核对的边界移除，并增加 200000 次有限轮询和失败字段；失败后不发布
虚假的完成标记。只保留 3 个协议 Tensor，使用 program 模式显式 AIC/AIV 类型，
仍是单个 24 组 MIX。这是控制协议诊断，不能替代 132 参数完整 CSA 验证。

## 对照与当前状态

| 尝试 | 改动 | 结果 / 任务 |
| --- | --- | --- |
| protocol_v1 | 初始抽取诊断 | CPU 常量写入类型错误，未上卡；修正 INT64 cast，并改用显式 AIC/AIV。 |
| protocol_v2 | 原控制协议，移除数学并加入有限轮询 | `task_20261016_162004_206538014143` 已结束；设备正常返回，但 72 核均记录等待失败。 |
| protocol_v3_volatile | 仅把诊断生成 C++ 的 progress 标量访问改为 volatile；表及轮询预算不变 | `task_20261016_162414_222408211822` 已结束；3 个代际的控制进度符合预期。 |
| protocol_v4_fence | 只在消费者失效缓存后增加现有 `pl.system.fence()` | `task_20261016_163312_271626813951` 已结束；设备返回，首轮 72 核有限等待失败。 |
| protocol_v5_volatile_read | 仅 progress 读改为 volatile | `task_20261016_164435_356209620998` 已结束；3 个代际的控制进度符合预期。 |
| protocol_v6_volatile_write | 仅 progress 写改为 volatile | `task_20261016_164436_356260019239` 已结束；设备返回，首轮 72 核有限等待失败。 |

以上是去掉业务计算后的诊断观察，不能证明跨核缓存一致性、payload 交接或完整 CSA
正确。当前无本会话待执行的对照任务。用户要求先复核 Scalar 非一致缓存协议，
暂停继续增加接口或上卡；最新解释见
[Scalar 缓存协议复核](SCALAR_CACHE_PROTOCOL_REVIEW_20261016.md)。

以上目录位于工作树的 `tests/pypto_test/results/csa_static_spmd_feasibility_20261015/`，
完整目录名以 `static_b16_` 开头。归档文本在本目录 `evidence/`；大产物仅保留
本地路径与大小，不计算内容 hash。

## v2 可以说明什么

例：core0 等待 core32 的 1025，有限轮询记录值一直没有达到目标，最后为
1024；Host 取回的最终快照中 core32 已是 1025。72 核均进入当前初始代际，
其中 17 个失败等待的目标在最终快照中已达到。

这说明移除数学计算后，同一控制表仍能出现等待失败，优先排查协议实现。
但最终快照不包含精确写入时刻，不能仅凭上述差异断言编译器做了 CSE/LICM；
有限轮询预算和执行先后也要由同预算对照排除。

当前生成 C++ 的 progress 读取是普通 GM 指针读取。只读核对发现，现有
PTO-ISA 的 TWAIT/TTEST 对信号读取使用 volatile，并配合缓存与流水操作。
两者表达不同是待验证线索，尚未据此认定 PyPTO、PTOAS 或 ISA 的责任。

CCE 的 `-emit-llvm` 与 `-print-after=licm` 都不受当前工具链支持；
llvm-objdump 输出为 `not available`。失败命令和输出已保留，没有可用的
优化 IR 或汇编证据。

## 接续边界

全部对照已结束，不能沿用此前“只有 volatile 有效就推进组件修复”的推断。
读侧 volatile 改变诊断结果，但它不保证绕过 Scalar DCache；尚无优化 IR 或汇编
证据可认定具体编译器错误。写侧单写者不因存在多个读者就必须使用 atomic，
需要分别核对 DCCI + DSB 发布、读取路径和业务数据同步，保留原逐前驱依赖。
当前仅整理协议审查结果，不继续实现或测试。

协议通过后仍必须返回完整静态 128K/B16 native 验证，检查数学输出、六份
cache/state、保护区和跨调用代际。协议诊断的成功不等于完整 CSA 成功。
本轮尚未修改 PyPTO、Simpler 或 PTOAS 的组件源码。
