# early_dispatch 到 early_start 的静态覆盖分析

2026-10-11，接续七档测量。本文分析对象为Simpler `d00f8f421`及其预编译产物，分析当时没有新增设备测试或协议修改。

后续修正：Simpler `5c8418a02`已移除本文识别的“本核前驱必须是直接WAIT生产者”限制，
跨核依赖最小用例两轮均48/48合法交接且数据等待、输出及保护区通过。
以下保留七档测量版本的分析口径；未将旧性能成绩改标为新版本。
[过程050及证据](../DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#52-过程-050移除本核前驱必须是直接-wait-生产者的额外限制)。

## 数量口径

此前七档只记录整体耗时及功能，**没有记录实际 early_dispatch、handoff 和有效流水重叠次数**。
因此不能将 63 处 early-resolve 设置、73 个编译 kernel 定义或下面的静态数量当作实际交接命中数。
`allow_early_resolve` 在运行时是生产者标记：`propagate_dispatch_fanin` 根据直接 WAIT 生产者的
标记及全量 block 发布进度，使消费者获得提前派发资格；不等于该生产者自身必然提前派发。

按本轮128K/B16输入（T=96、max_seq_len=131078）展开生成的orchestration，排除未走的条件分支，
将8个O projection组展开，排除AICPU dummy/barrier任务：

| 阶段 | 实际分支静态展开的task组数 | AIC/AIV单核block调用数 |
| --- | ---: | ---: |
| HC_pre + norm | 6 | 78 |
| metadata + rope sign | 2 | 17 |
| QKV | 8 | 139 |
| Compressor | 4 | 62 |
| Indexer compressor | 8 | 78 |
| Indexer | 9 | 294 |
| Sparse attention | 4 | 105 |
| O projection + HC_post | 25 | 200 |
| 总计 | 66 | 973 |

MIX的一个block按1个AIC和2个AIV成员计数。这些是源码代入形状后的调用数量，不是实测调度计数。

- 66组中62组生产者设置allow_early_resolve；未设置的是qproj_matmul、两个compress_state_commit、
  当前长档Indexer Score MIX。未设置是否阻断某消费者，仍取决于其真实WAIT图。
- 64组具备后继Wait入口，另外2组MIX（Indexer Score与Sparse QK/PV）没有接入。
- 当前代码又排除sync_start后继：kv_rms_norm_rope（3个block）与proj_b_act_hc_post_0（48个block）。
- 所以仅看入口能力和sync_start限制，最多还剩**62组、778个单核block调用**可能作为后继。
  这是资格上界，仍包含无本核前驱、正常ready派发、不同直接生产者等不能等同于early-dispatch交接的情况。
- 2组MIX占144个单核调用，另2组纯AIV sync_start占51个；195/973约20.04%的调用先被覆盖限制排除。
  MIX同时不能作为当前异步尾部前驱，影响还会延伸到后继；不能只用这20.04%估计时间损失。

[全部提交点及能力字段](static_submit_sites.json)、
[128K/B16静态展开、每组block数和来源行](static_128k_b16_expansion.json)。

## 当前实现额外缩窄了什么

代码位置均相对于私有Simpler根目录：

1. `scheduler/scheduler_dispatch.cpp:245`只给`to_pending`的后继开放交接。
   已经early-dispatch到idle核的任务仍停在普通doorbell gate前，没有执行算子Scalar前缀。
   这种情况没有本核前驱流水可重叠，若要提前做跨核依赖等待前的Scalar，应单独评估，不能冒充同核收益。
2. 同处直接排除带predicate和sync_start的后继。sync_start排除是当前原型覆盖限制，
   不表示理论上必须把整个Scalar前缀放到整组启动门之后。
3. 对gated pending后继，`is_scalar_prefix_wait_producer`（同文件:66）要求本核前驱是同资源类型的
   **直接WAIT生产者**，还须验证live fanout。仅仅“后继已正常dispatch到本核”并不足够。
   ready pending路径没有这个直接依赖要求，因此其收益也不一定来自early_dispatch队列。
4. 对已经正常派发的后继，理论上可先与本核前驱的尾部重叠安全Scalar，再在Wait完成本核前驱，
   发布前驱真实FIN，随后继续等待后继自己的完整fanin。安全Scalar不读取生产者数据时，
   数据生产者不必恰好就是本核前驱。因此第3项是当前额外保守条件，优先审查其必要性，
   不通过改变核选择、ACK后派发或allow_early_resolve来提高命中率。
5. O-A(AIC)→quant(AIV)→O-B(AIC)在每个O projection组中重复。
   这类跨资源依赖不能天然满足“本核前驱恰好是直接生产者”。例如O-B已经pending在某AIC前驱后面，
   但其所有直接WAIT生产者都是AIV quant，当前gated交接会被拒；仍须考虑实际TMR额外依赖，
   不能仅凭这条显式链断言本次所有O-B都没有交接。
6. `aicore/aicore_executor.cpp:129`在前驱返回后只探测一次已经可见的后继。
   后继若在此后、普通PIPE_ALL等待期间才被提前派发，这一次不会再进入交接。
   “发生在FIN前的early_dispatch”因此仍不等于“赶上可用尾部窗口”。

Indexer Score与QK/PV承载主要计算阶段；当前MIX交接未接入，是实现覆盖缺口。
不能根据64/66组有Wait，或七档加权仅改善0.1569%，推出该机制本身只有这么大的收益上限。

## 同核排队前驱不等于数据生产者

用户追问：后继既然已经被派到忙核后面，是否只有本核前序task可能是它的生产者？
当前调度器没有这种保证。`scheduler/scheduler_types.h:433`的pending选核仅检查资源类型、
核正在忙以及pending槽未占用；`scheduler/scheduler_dispatch.cpp:717`的`prepare_from`
从可用槽集合取`pop_first()`，不检查该核当前task是否是消费者的直接WAIT生产者。
提前派发资格检查的是其生产者已完整发布等条件，不是生产者与消费者同核。

例如以下是源码允许的排布，尚未声称在本轮CSA中实测到：

```text
数据依赖：A → B；X与B没有数据依赖。
核0：X正在执行，B已被正常early_dispatch到pending槽。
核1：A正在执行，尚未完成。
```

普通执行先等X完成，再等待B的完整数据依赖。early_start可让B的安全Scalar先与X尾部重叠，
在Wait处完成本核流水等待、发布X的真实FIN，然后继续等待A等全部生产者。
现有`aicore/aicore_executor.cpp:93`的Wait回调将这两个等待分开；
然而派发侧新增的直接WAIT生产者条件会拒绝上例，因为X不是B的生产者。
不能通过改变选核去满足这一额外限制，后续应审查如何在保持原派发和数据gate语义的前提下放宽。

修正此前交流中“主要等待另一核的数据就被挡”的表述：实现不判断“主要”等谁，
而是判断X是否属于B的直接WAIT生产者集合。若B同时依赖X和A，则不会因另一核A而被这一项拒绝；
若X不在该集合中，才会被这一项拒绝。实际被拒的次数、有效重叠机会及收益仍待计数。

## 理论收益与下一项证据

对确实可提前执行的同核后继，令R为开始后继Scalar时前驱剩余流水时间、S为安全Scalar前缀耗时，
δ为新增协议成本，则理想局部净收益为`min(R,S)−δ`。在R、S为正且δ近零的假设下，局部推理成立。
R为零可能是未赶上窗口；不能据此把整个task定性为Scalar-bound。
生成的Scalar前缀也只是task开头可安全执行的一段，不是其全部Scalar指令。

多核局部收益不直接相加为CSA收益：24个核同时各节省0.1μs，整段可能只缩短约0.1μs。
非关键路径上的收益还可能不改变CSA终点。覆盖不足和这些并行约束需要分别量化，不能互相代替解释。

需要的下一项证据，是无profiler路径下按block/核统计：

1. 实际early_dispatch到idle和pending的数量；ready队列的pending单独记账。
2. 因MIX、sync_start、直接WAIT生产者条件等原因被拒的数量。
3. 前驱返回时后继已可见并真正handoff的数量。
4. 实际安全Scalar与前驱流水重叠的窗口，以及其中影响CSA关键路径的部分。

这些数据目前未采集。本次不提交新上卡任务，也不先改调度或解除上述约束。
