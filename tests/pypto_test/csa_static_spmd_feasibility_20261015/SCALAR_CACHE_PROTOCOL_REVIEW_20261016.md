# CSA 单 SPMD：Scalar 非一致缓存协议复核

## 当前结论和范围

2026-10-16 用户指出：A3/A5 普通 Scalar cache 的非一致性问题相同，不能因平台不同而
忽略；两代原子指令接口则需区分。停止推进 volatile 接口或提交完整设备续测，先复核协议。所有静态源码、
失败与诊断通过记录保留。本次只读代码并整理记录，没有新增组件修复或设备任务。

此前“只读改为 volatile 就通过”的结论仅描述诊断结果，不能证明缓存一致性、payload
发布、跨核交接或完整 CSA 正确。现阶段不能将问题定性为 PyPTO/PTOAS 缺少 volatile
接口就能解决的 bug。首先要证明本仓生成的跨核协议成立。

## 已读参考与适用边界

已查看指定分支同目录三个 usage guide 的结论、接口语义、验证边界及相关采集方法。
参考快照和来源见 `references/simpler_atomic_guides_20261016/source.json`，对应
nalinaly/simpler `fdwic-swimlane-deps` 的 Git 版本
`3699b7ae4ff47abcfa1e2916881d4309cf39c00f`。不计算文件内容 hash。

- [ATOMIC_USAGE_GUIDE.md](https://github.com/nalinaly/simpler/blob/3699b7ae4ff47abcfa1e2916881d4309cf39c00f/tests/atomic_probe/ATOMIC_USAGE_GUIDE.md)：
  普通 Scalar 缓存访问没有跨核 coherence；DCCI 可能 clean/writeback，并非按地址纯
  invalidate；DSB 约束当前核的完成顺序；控制交接与 payload 可见性必须分别建立协议。
  其 A5 atomic 方案及特定路径可省 DSB 的实测不能照搬到 A3。
- [cache_preload_usage_guide.md](https://github.com/nalinaly/simpler/blob/3699b7ae4ff47abcfa1e2916881d4309cf39c00f/tests/atomic_probe/cache_preload_usage_guide.md)：
  preload 是性能 hint，不能替代写回、同步和交接；普通 store 被接受与完成 GM 发布
  是不同边界。current-PC ICache 预取还受实际代码地址与独立工作窗口约束。
- [icache_miss_usage_guide.md](https://github.com/nalinaly/simpler/blob/3699b7ae4ff47abcfa1e2916881d4309cf39c00f/tests/atomic_probe/icache_miss_usage_guide.md)：
  诊断代码即使运行时不落记录，仍可能改变代码布局、到达时序和竞争。PMU 与泳道采用
  各自构建和统计边界，不能逐 tick 相减；A5 selector/频率不能直接作为 A3 常量。

## 当前源码的实际协议

代码位置以 `generate_static_entry.py` 为准，下面行号为本轮核对时的位置。

| 环节 | 当前实现 | 尚缺的论证 |
| --- | --- | --- |
| 进度布局 | 371～379：每核 64 个 INT64，即 512 B；单核写自己的进度，多核读取 | 间隔不等于 coherence。仍须检查实际基址对齐、整个 line 的访问者及跨调用生命周期。 |
| 发布进度 | 430～436：本核 fence、整缓存 DCCI、fence、普通 Scalar 写进度、单行 DCCI、fence | 普通写与 DCCI 不能仅因诊断通过就被当成有明确契约的跨核通知。payload 与控制交接分别证明。 |
| 轮询进度 | 419～424：单行 DCCI 后普通 Scalar 读取；循环中重复 | 原版未在这些 DCCI 后显式等待完成。volatile 仅影响编译器访存，既不绕过 DCache，也不建立跨核同步。 |
| 消费 payload | 426～427：先 fence，再整缓存 DCCI，随后进入数学体 | fence 在 DCCI 之前，不能据此证明随后的失效/写回已在 payload 访问前完成。 |
| 全缓存维护 | 每个阶段进出均无参 cacheinvalid | 实际生成 C++ 为 ENTIRE_DATA_CACHE，可能涉及当前阶段之外的缓存 entry，不能按单行失效理解。 |

完整 B16 源码中的无参 cacheinvalid 静态调用点为 AIC 80 处、AIV 88 处，共 168 处；
这不是动态执行次数。该策略由本仓生成器加入，没有逐条 cacheline 所有权证明。
因此首先审计生成器协议，不应先把问题归咎于编译器。

## 初步布局核对（不是完整正确性证明）

- records/waits 是本轮执行期间只读表，和不断变化的 progress 不应按同类对象处理。
- progress 的 arena offset 为 0、每 worker 间隔 512 B；当前适配器按 512 B 规划
  offset/guard，但没有显式检查实际 allocation 基址的 cacheline 对齐。
- B16 数学体共有 4 处 Scalar GM 写入语句，落在 temporary_14/15/54。
  temporary_14/15 分别为 64 B 的两份 compact row offsets，由同一个原
  `csa_row_offsets` worker 写入；不能误报为多核争写同一行。
- temporary_54 为 [96,16] INT32，行距 64 B；原 valid-QK-plan 分片按 8 行划分。
  已核对行布局，但完整消费者、重复调用及 Scalar/MTE 可见性还没有闭合证明。
- 当前 Simpler toolchain 显式关闭 `cce-aicore-dcci-insert-for-scalar`，未显式关闭
  `cce-aicore-dcci-before-kernel-end`。后者的实际编译产物仍需核对，不能仅按源码
  推断所有 DCCI。没有在本轮改动这些编译选项。

## 保留的对照结果及其限度

所有以下任务均已终止，当前没有本会话排队或运行的设备任务。

| 同一 B16 控制表诊断 | 任务 | 观察 |
| --- | --- | --- |
| 原标量访问 | task_20261016_162004_206538014143 | 首轮 72 核有限轮询失败；设备返回。 |
| progress 读写都加 volatile | task_20261016_162414_222408211822 | 连续 3 个代际的进度值符合预期。 |
| 仅消费者加 fence | task_20261016_163312_271626813951 | 首轮 72 核有限轮询失败；设备返回。 |
| 仅 progress 读加 volatile | task_20261016_164435_356209620998 | 连续 3 个代际的进度值符合预期。 |
| 仅 progress 写加 volatile | task_20261016_164436_356260019239 | 首轮 72 核有限轮询失败；设备返回。 |

这些用例移除了数学体，只保留 3 个控制 Tensor，使用 program 显式 AIC/AIV；
没有覆盖真实 payload 的缓存状态和共享访问。三轮完成不能升级为完整协议或 CSA
通过，读写差分也不能单独证明某个 CCE 优化就是根因。

在用户纠正之前，另外只做过 CPU lowering 可表达性探针：带私有函数声明的
func.call 可以生成 C++；直接 emitc.call_opaque 携带 !pto.ptr 被当前 PTOAS 拒绝。
它们没有上卡，也没有形成 PyPTO/Simpler/PTOAS 源码补丁，现停止沿此方向直接修接口。

## 后续讨论的前提

保留原逐前驱依赖关系，优先审查单写者进度发布与绕过 Scalar DCache 的读取方式，
逐类核对 payload 写回、消费者缓存刷新、cacheline 归属及跨调用代际。单写者可以
消除并发更新冲突，不因有多个读者就必须增加 atomic；但发布和读取仍需完整成立条件。
不能用“512 B 间隔”“加 volatile”或几轮通过代替该论证，也不能无依据把原依赖图
改成阶段间全卡屏障。

本次只完成问题复核并保存证据；同步实现方向待与用户讨论，尚未恢复修改或测试。

## 用户补充分析：A3 软 barrier 与单写者进度发布的区别

已只读核对当前本地 `pto-isa-src/include/pto/common/syncall_soft.hpp` 第 35～65 行、
96～116 行，以及 `pypto/python/pypto/language/op/system_ops.py` 第 154～202 行。
这是本地源码核对，不等于独立编译路径已经采用该实现，更不等于本轮进行了设备验证。

A3 的共享计数器到达流程为：

```cpp
set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
dcci(counter, SINGLE_CACHE_LINE);
st_atomic<int32_t>(1, counter);
dcci(counter, SINGLE_CACHE_LINE);
dsb(DSB_DDR);
```

职责必须分开：

- `st_atomic` 保证多个参与核累加同一计数器时不丢更新。A3 源码实际存在这条路径，
  因此“A3 没有 atomic”的笼统表述不准确；不能与 A5 的原子接口混为一谈。
- DCCI 清理并失效对应 cacheline；该源码明确把写后的 DCCI 作为原子更新发布点。
  DSB 保证相关本核内存操作完成与顺序，不能单独建立跨核 cache coherence。
- 轮询使用 `ld_dev`，绕过本核 Scalar DCache；不等于每次都直接访问 HBM，
  也不代替计算流水完成或业务数据的跨核交接。

`syncall(SOFT)` 是参与集合的 barrier。每个参与者本轮到达后等计数达到本轮阈值；
CSA 原型则只等依赖表中的必要前驱。不能为复用前者，把后者的所有阶段强制改成
72 核集合。源码中的局部 `before = load; atomic_add(1)` 也不是返回旧值的单条 RMW；
此处只描述已有 barrier 算法，不把它当作通用 fetch-add。

对于当前每 worker 自己写的进度：

1. 确保整条进度 cacheline 只有该 worker 写入；其他核不能写同一行的邻接字，
   也不能留下会覆盖新值的旧 dirty line。静态模板还需保证串行复用和正确代际。
2. 先完成生产者计算与数据写回，再普通写进度，按所需顺序 DCCI + DSB 发布进度。
   这里没有多写者更新竞争，不需要仅为“其他核要读”引入 atomic RMW。
3. 消费者可采用相应宽度的 bypass load 轮询进度；若仍用缓存读取，必须给出
   相应的缓存维护与编译器读取保证。`volatile` 不等同于 bypass load。
4. 观察到进度后，再按业务数据的实际访问路径完成消费者缓存处理与顺序约束。
   标记到了与业务数据可读是两个必须连接起来的条件。

当前进度为 INT64，而上述 barrier 计数器为 INT32；未来实现不能只复制一行
`ld_dev` 就忽略访问宽度、对齐和非撕裂读取要求。本轮未修改类型或同步实现。

ATOMIC_USAGE_GUIDE 中“控制变量使用 atomic/硬件原语”是其 A5 工程保守规则；
它提醒审查普通 Scalar/DCCI 混用的风险，不能据此推导“A3 单写者协议必然需要
atomic”。用户此次修正已采纳：原子更新、缓存发布、读取路径和 payload 同步分别论证。

## 单个 incore task 内的点对点通知接口建议

用户确认需求：将原多 task 的依赖保留在一个融合 incore task 内，PyPTO 应提供
生产者通知、必要消费者等待的接口。这一抽象方向合理，不能用全体 barrier 替代。
当前本地源码并非只有 syncall，需区分现有接口的适用范围：

| 现有接口 | 本地定义与范围 | 与本次需求的关系 |
| --- | --- | --- |
| pl.system.syncall | language/op/system_ops.py：参与核集合 barrier | 等全部指定参与者到达，不能等价替换逐前驱依赖。 |
| pl.system.sync_set / sync_wait | 同文件：Cube/Vector 硬件事件，event_id、pipe、ffts_mode、core_type | 已有细粒度核间事件；接口没有普通 GM 信号 Tensor 与动态进度阈值，不能直接视为本次任意 worker 进度接口。 |
| pld.system.notify / wait | language/distributed/op/system_ops.py:124、160：DistributedTensor、peer rank、信号 offset | 已有跨 rank 信号抽象并 lowering 到 TNOTIFY/TWAIT；当前入参是通信窗口上的 DistributedTensor。 |

因此需求可准确表述为：为普通 GM 信号提供单个 SPMD 内的核间 notify/wait，或扩展
现有信号接口以支持该场景。底层优先核对复用已有 PTO-ISA 能力，不预设一定重写实现。
用户随后明确授权提交正式功能需求，已创建
[PyPTO #3020](https://github.com/hw-native-sys/pypto/issues/3020)。
尚未实现 API 或修改组件；正文及回读记录在
`upstream_issues/incore_p2p_notify_wait_20261016/`。

建议明确以下合同：

1. 用逻辑 worker 对应的信号 slot 和动态进度值表达依赖，支持单写者、多读者，
   消费者等待 value >= expected；通知不因一个消费者读取而被清除。
2. 支持代际和循环复用，说明计数宽度、溢出、初始化以及工作区跨流并发规则。
3. 区分 Set 与 AtomicAdd：单写者无需 RMW；多写者聚合必须选用相应原子更新。
4. 明确发布/等待的内存顺序和 payload 合同：接口是否等待生产者流水并处理相关
   数据缓存，或需要调用者显式执行哪些操作；不能只保证信号变化却暗示业务数据已可读。
5. 编译器应把这些操作作为有同步副作用的操作，保留轮询读取及相关访问顺序；
   MIX 展开须明确 AIC/AIV 和双 AIV lane 的执行归属，避免重复通知。
6. 后端封装 A3/A5 的实际访问、DCCI/DSB 等差异；说明参与核驻留与可推进前提。
   等待仍发生在执行核内，不通过 AICPU 重新派发阶段。


## 后续实现与验证（2026-10-16 18:07 更新）

上文保留暂停审查时的判断。用户随后授权临时普通GM notify/wait，现已完成独立实现。
首版信号进度正确，但AIC写回尚未完成就可能发出通知；在CSA notify封装加入本核
PIPE_ALL后，24组数据交接小例及完整128K/B16各三次调用通过。完整输出、Top-K及
六份cache/state对冻结动态PTO完全相同，所有保护区通过。此结果不代表通用协议证明、
A5验证、动态模板或Native数值验收。详见LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md。
