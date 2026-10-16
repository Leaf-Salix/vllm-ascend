### 需求概述

希望 PyPTO 提供用于**同一个融合后的 InCore task / MIX SPMD 内部**的核间点对点通知与等待接口：

> 生产者核完成指定阶段并发布数据后，通知依赖它的消费者核；消费者只等待自己的必要前驱，无依赖的核继续执行。

希望支持普通本地 GM 信号 Tensor、动态进度阈值和循环代际。可以扩展现有信号 API 或复用已有 PTO-ISA 能力，具体命名与实现方式由组件设计决定。

这是功能需求，不将当前原型的停滞直接认定为 PyPTO 编译器 bug。

### 使用场景与动机

我们正在将 DeepSeek-V4 Flash CSA 中原有多个 InCore task 的计算体与依赖，融合到一个 MIX SPMD 中。当前目标场景为 A3、24 个核组，即 24 AIC + 48 AIV；先验证 history=128K、B16 的静态原型，最终仍需要动态 batch/sequence metadata。

融合前，任务之间的依赖由运行时调度维护。融合后，同一 task 内需要保留原有的核间生产者/消费者关系：

- 某个生产者完成阶段 k 后，消费者 A、B 可以继续读取该阶段结果。
- 与该生产者无依赖的消费者 C 不需要等待。
- 一个消费者可能有多个前驱，必须等所有必要前驱达到各自的进度。
- 核间执行顺序和每核工作量可以不同，不能把所有阶段变成全体集合 barrier。
- 信号只表示阶段完成；相关 workspace 的覆盖/复用仍必须保留原依赖或独立确认，消费者不能因看到更大的进度值就读取已被覆盖的数据。

希望将上述依赖完全保留在已启动的 InCore task 内，阶段之间不重新通过 AICPU 派发。此需求不预设一定能取得性能收益，首先要求同步语义正确、可验证、可维护。

### 与现有接口的区别

已核对公开主线版本 [8325a8de8422678d6699e17f31cef953ace66445](https://github.com/hw-native-sys/pypto/commit/8325a8de8422678d6699e17f31cef953ace66445)，当前并非只有 syncall：

| 接口 | 已有能力 | 本次希望补齐的部分 |
| --- | --- | --- |
| [pl.system.syncall](https://github.com/hw-native-sys/pypto/blob/8325a8de8422678d6699e17f31cef953ace66445/python/pypto/language/op/system_ops.py#L150) | 参与核集合的 barrier，包含 HARD/SOFT 模式 | 按指定前驱和进度等待，保留无关核之间的独立推进。 |
| [pl.system.sync_set / sync_wait](https://github.com/hw-native-sys/pypto/blob/8325a8de8422678d6699e17f31cef953ace66445/python/pypto/language/op/system_ops.py#L90) | Cube/Vector 硬件事件，event_id、pipe、ffts_mode 等 | 普通 GM 信号上的可复用进度阈值、逻辑 worker 间依赖，以及单写者多消费者的明确合同。 |
| [pld.system.notify / wait](https://github.com/hw-native-sys/pypto/blob/8325a8de8422678d6699e17f31cef953ace66445/python/pypto/language/distributed/op/system_ops.py#L124) | DistributedTensor 通信窗口上的跨 rank 信号，lowering 到 TNOTIFY/TWAIT | 同一设备、同一 InCore task 内的普通 GM Tensor 信号，无需为此引入通信组和跨 rank 窗口。 |

相关 issue：

- #2072 已提供显式 Cube/Vector 事件接口；本需求不是重复要求增加 sync_set/sync_wait。
- #2130 涉及流水与跨核调度表达，和本需求相关，但本次范围是底层通知/等待合同。
- #1906 关注跨 rank TPUT 的完成与发布；本需求优先解决单设备本地 GM 的核间交接，不要求一并解决远端 TPUT 完成问题。

### 建议的 API 与行为

以下仅为行为示意，**不是当前可直接执行的 PyPTO API，也不限定最终名称或签名**。代码位于同一个已启动的 InCore/SPMD 内：

```python
# signals：普通 GM 信号 Tensor；每个生产者的信号独占所需缓存行。
# worker：此次启动中的逻辑 worker，包含明确的 AIC/AIV lane 映射。
# epoch_step / required_step：设备端进度值，可在循环中变化。

if worker == producer:
    produce_payload(...)
    # 按接口合同完成所需的数据流水、写回和发布顺序。
    pl.system.notify(signals, offsets=[producer, 0], value=epoch_step)

if worker == consumer_a or worker == consumer_b:
    pl.system.wait(
        signals, offsets=[producer, 0],
        expected=required_step, cmp="ge",
    )
    # 按接口合同完成消费者所需的数据缓存维护与读取顺序。
    consume_payload(...)

if worker == independent_worker:
    independent_work(...)
```

希望接口至少明确以下语义：

1. **点对点及一对多依赖。** 单写者可以有多个消费者；wait 不消费或清零信号。消费者可以逐个等待自己的前驱，也可由组件提供等价批量接口。不要求所有启动核在相同位置调用。
2. **动态阈值和复用。** 支持设备端 slot/expected value、循环中的单调进度及代际复用。说明信号类型、访问宽度/对齐、初始化、溢出处理和跨调用生命周期，防止上一轮信号误满足本轮等待。
3. **单写者与多写者分开。** 单写者 Set 不因多个读者而必须使用 atomic RMW。若提供多写者计数聚合，则明确 AtomicAdd 等操作及其合同；这不是最小单写者接口的前置条件。
4. **信号发布和读取保证。** notify 的写入应按规定发布；wait 必须重新观察进度，不能因编译器复用旧值或 Scalar DCache 命中旧值而无法前进。A3/A5 后端各自处理必要的缓存维护、内存顺序和读取路径。
5. **业务数据同步合同。** 明确 notify/wait 是只同步信号，还是同时提供对指定 payload 的 release/acquire 保证。若需要显式数据 fence/cache 操作，请提供可组合的公开接口与正确示例，明确其覆盖范围及执行次序。不能只保证标记变化，却暗示消费者一定可读业务结果。
6. **编译器与 MIX 语义。** 标记为具有同步副作用的操作，防止错误删除、提升、合并或重排；明确 AIC、AIV0、AIV1 的执行归属，避免一次逻辑通知在 MIX 展开后重复发布。同时说明与自动同步、内存复用和现有事件资源的相互作用。
7. **可推进条件。** 说明参与核驻留、启动规模和等待拓扑的前提。接口不承诺挽救循环依赖或等待未获调度核造成的死锁，也不应隐含插入全卡 barrier。

### 内存语义方面需要避免的误解

- Scalar DCache 非一致性是独立于编译器 volatile 的问题；volatile 不能建立跨核缓存一致性。
- 原子更新解决并发修改冲突，不自动等价于业务数据发布。
- DCCI 可能包含 clean/writeback，不能一律理解为纯失效；必须考虑整条 cacheline 的所有权。
- DSB 是相应本核操作的完成与顺序约束，本身不是跨核会合。
- 单写者进度可以通过正确发布与读取协议实现，不要求机械加入多写者 atomic。
- A3 的 st_atomic 等路径与 A5 的原子接口需要分别处理，不能概括成“A3 没有任何 atomic”。

这些细节应由公共接口合同和后端实现集中承载，避免每个融合算子各自手写普通 Scalar load/store、DCCI、轮询循环。

### 已考虑的替代方案

- **阶段间 syncall(SOFT)：** 改变原有依赖偏序，引入无关核之间的等待；不能作为点对点通知的等价替换。
- **继续保留独立 task：** 现有正确路径可以保留，但不能满足本次单个融合 task 内部表达依赖的目标。
- **手写 GM 进度轮询：** 当前原型采用每 worker 单写自己的进度行、消费者按前驱表轮询。必须自行承担缓存发布、编译器读取、代际及 payload 交接的完整合同，适合诊断但不适合作为各算子重复实现的基础机制。
- **手改生成 C++ 的 volatile：** 简化控制表诊断中观察到结果变化，但不构成缓存协议正确或完整 CSA 通过的证据，也不是公共 API 方案。
- **复用现有通知后端：** 欢迎基于已有 TNOTIFY/TWAIT 或其他硬件机制实现；本需求不预设要重新发明底层同步，只要求普通本地 GM 场景的公开表达和明确合同。

### 建议的验收范围

不要求以完整 CSA 模型作为组件的最小验收用例，建议用独立小例子覆盖：

- 单生产者/单消费者、单生产者/多个消费者，以及消费者有多个必要前驱。
- 同一个 MIX 中的 AIC/AIV、不同核组与双 AIV lane，另有无关 worker 独立推进。
- 多轮动态进度及重复调用，不被旧代际误唤醒；明确计数回绕/重新初始化规则。
- 实际 payload 跨多个 cacheline 的读写交接，包含 Scalar 与搬运流水的代表路径，不只检查信号最终值。
- 最小 kernel-mode/JIT 路径及声明支持的 program 路径；若支持图重放，覆盖固定地址信号的复用合同。
- 代码生成验证同步副作用与 lane 归属；设备用例有有限超时和可诊断的失败信息。优先 A3，并明确 A5 的支持与验证状态。

当前完整融合 CSA 尚未得到功能/精度通过结论；本 issue 不宣称根因已归属于某个组件，也不把几轮控制表诊断通过当成同步协议正确性证明。
