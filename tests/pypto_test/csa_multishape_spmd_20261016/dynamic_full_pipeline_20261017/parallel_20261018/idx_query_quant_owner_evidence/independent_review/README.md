# Indexer Hadamard → Query Quant 单 producer 等待：独立只读复审

**结论：正式 66882634 的现行分工下可成立，未找到尾部、跨 wave、视图或隐含后继依赖反例。** AIV 的逻辑 worker v<T 时，只需等待 AIC `v%24` 的 `PHASE_IDX_QUERY_HAD`；v≥T 没有量化任务，可跳过该等待。原 Hadamard/量化数学、全部阶段发布、后续全 Query Quant 等待与缓存维护须保持。

本报告独立审查正式源码，不与正在执行的 Cube 16-owner 候选组合。没有修改源码、编译、上卡或另建枚举模型；ultra_indexer_policy 的模型候选由其独立维护。本轮核对了实际函数体和后继访问顺序，没有仅凭 tile 名称推断。

## 唯一 producer 的证明

`indexer_query_stages.py:201–217`：AIC c 处理 token `c+24*k<T`，一次读取 64 个 head、写同 token 的全部 FP32 Hadamard 输出。

`indexer_query_stages.py:221–255`：AIV v 处理 token `t=v+48*w<T`，唯一外部 payload 读取为 `idx_query_had[64*t:64*(t+1),0:128]`。因此：

```text
Hadamard owner(t) = (v+48*w)%24 = v%24
```

同一 AIV 的全部 wave 都来自同一 AIC。现行 `cube_schedule.py:311` 在本核全部 Hadamard wave 发出、原发布协议完成业务写回后才通知一次，足以覆盖所有这些 token。两个 AIV v 与 v+24 可以等同一标记；signal wait 不清零或消费信号。

目标调用应为：

```python
if worker < tokens:
    signal = wait_producer(signal, PHASE_IDX_QUERY_HAD, worker % GROUPS, CORES, generation)
# indexer_query_quant_stage 与 PHASE_IDX_QUERY_QUANT 原发布保持原位置。
```

这里 producer 是 AIC 物理核 0..23，**不加 24**。T25/v24 必须等待 AIC0；若沿上一条 RoPE 等待方案加 24，就会等到 AIV 的非 HAD 发布槽。这是实现时最重要的编号反例。

T≤24 时活动 v 的 owner 就是 v；T25..48 时开始两个 AIV 共用一个 AIC；T49、97 等跨 wave 仍满足同一同余式。T 是整数 token 数，`T*64//64=T`，不存在半个 head tile 尾部，也不读取 T 之外的 padding。T0 的两个循环均为空，但正式 host policy 拒绝 T0；这里只说明防御性条件。混合 Q、dummy、动态 L/position 不改变这两阶段按实际 T 计算的 token 领取方式。

## 视图、缓存和无任务角色

`binding.py:195–197` 分配三个独立连续缓冲：Hadamard FP32[T*64,128]、Quant INT8[T*64,128]、scale FP32[T*64,1]。Hadamard 与 Quant 使用同一头行视图，不存在额外转置或跨 token reshape。每 token 的输入为 32768 B，INT8 输出为 8192 B，scale 为 256 B；它们不别名或复用存储。

`binding.py:240` 的 scale heads view 为同址 [T,64]，地址 `4*(t*64+h)` 与 [T*64,1] 一致。量化读取的是自己 token 的完整 128 列，再按原两个 64 列块求 amax；没有跨 token 归约，也没有读取其它 head weights 或 Native metadata。

发布/获取仍使用原协议：Hadamard 发布先完成本核 FIX/业务写回，信号单写者按物理核隔离；`wait_producer` 读取同代标记后执行 fence/cacheinvalid/fence，与旧多 owner wait 末尾的业务获取维护一致。缩小的是 signal 等待集合，不是删除业务缓存维护。

不同 token 的 Hadamard 写入不重叠。Quant 输出各 token 也唯一写；原实现已经允许 48 个 AIV 并发写各自 INT8/scale，本改动不增加输出间的别名或 Scalar 读改写。不能用 signal 的 512 B 隔离跨度推断 payload 的硬件 cacheline 大小。

v≥T 的量化循环没有 payload load/store；此前 `PHASE_ORIGINAL_CACHE` 发布仍完成自己的业务写回，其后的空 Quant 发布也保留。跳过无任务角色的 HAD 获取不会让它消费未就绪的 query_had，也不改变 72 核 begin、代际或通知数量。

## 旧全 HAD 等待是否代传其它数据

这是该候选需要保留的关键后继证明。旧 `vector_schedule.py:697` 实际等待的是 `min(T,24)` 个 HAD owner，T≥24 时才是全 24。

从该位置到 `vector_schedule.py:711` 的下一个全 Query Quant 获取之间，代码依次为：Query Quant 数学、Query Quant 发布、WEIGHTS 信号等待。唯一业务 payload 读取正是本 worker 的 query_had，已被单 producer 覆盖；没有在这一窗口读取其它 AIC 的业务结果。

第 711 行仍等待全部实际 Query Quant producer：

```text
V = {0,...,min(T,48)-1}
每个 v∈V 的 Quant 发布都在 HAD[v%24] 获取之后
{v%24 | v∈V} = {0,...,min(T,24)-1}
```

所以这次原样保留的全 Quant 获取，**在进入 coefficients 数学前恢复了旧全 HAD 等待的全部前驱可见性**。该证明也适用于无 Quant 任务的 AIV：它们虽跳过本地 HAD 等待，仍会取得所有实际 Quant producer。不能用无任务 AIV 的提前空发布充当完成门槛；现有 row-owner 公式不会等待那些槽。

此外，weights 在第 710 行显式获取；coefficients 所需 plan/Native 描述符仍沿原路径获取。compact Hadamard、主 QB、主投影分别保留独立的 `PHASE_IDX_HADAMARD`、`PHASE_QB`、`PHASE_MAIN_PROJECT` 等待。Cube Score 的 `cube_schedule.py:350` 同样仍获取全部实际 Query Quant producer。未发现后续阶段只能借旧全 HAD 等待取得某个非 query_had 数据的情况。

**因此第 711 行和 Cube Score 的全 Query Quant 等待不能随本候选一并收窄。** 若以后改变它们、Hadamard 发布位置或 48/24 的循环步长，必须重新证明跨 token 数据与可见性，不能复用本报告结论。

## 边界与交叉意见

固定 48 AIV 是 24 AIC 的整数倍，是单 producer 同余关系成立的条件。例如若量化步长改成 32，v0 会消费 t0/t32，Hadamard owner 分别为 0/8，单 owner 即错误。当前 source 的 WORKERS 实参固定为 VECTORS=48 与 GROUPS=24，和 Indexer G2/G6 策略无关。

已向 ultra_indexer_policy 提供独立推导；其反馈同样确认量化唯一外部 payload load、独立缓冲、v24/T25 编号及全 Quant 后继依赖恢复。其正在准备从正式 668 单独出发的候选，不合并 Cube 16-owner 源。

本报告不把等待次数下降写成性能收益。全 Quant 门槛仍在，只是允许各量化角色在自己的 Hadamard producer 完成后推进；实际关键路径及完整 L1 由独立设备对照决定。

