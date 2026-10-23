# Native 数值不变的下一块 KV 预取候选

## 目标

本轮最大区间差是 TopK→heads。我方当前 QK/PV 在 AIV 发布概率后立即等待 PV，尚未利用该时间搬运下一块 Key。借鉴 nalinaly 的阶段重叠，但不复制它的 K128 softmax、概率 rint 或在线归并顺序。

本候选独立于 H3、L1 驻留、索引检查和 UB 统计，成对对照始终是 `aad137911` 的正式组合版。

## 数据生命周期

- 原每核一个 `[512,512]` BF16 KV GM 区改为两个，总增量 `24*512*512*2=12MiB`。根仍64个Tensor；不增加业务输入或每请求padding。
- 工作编号 `w=((t-core)//24)*SPARSE_BLOCKS+sb`，槽为 `w%2`。使用全局工作号，而非仅 `sb%2`，也适用于跨query和奇数块数。
- 当前概率已发布后，AIV只预取紧邻下一项且有效的work，写另一个槽，完成后发布它的 KV_READY，再等当前 PV。
- 下一项若前驱有效，直接消费已经预取的数据；前驱无效则同步采集并发一次 KV_READY。
- 无效work双方均跳过；最后一项不多发信号，不读下一query。

## 必须保留的顺序

每项仍是 `KV_READY → QK → SCORE_READY → softmax → PROB_READY → PV → PV_READY → oi更新`。新的提前搬运不改变 QK/PV MMA、K512分块、actualK求和树、sink初态、概率round、alpha/oi更新或最终BF16/逆RoPE边界。

单槽 score/probability/PV 仍有保护：下一概率必须等当前 PV 完成并消费后才产生；下一 QK 只能在上一项 PV 完成、下一个 KV_READY 到达后开始。两 AIV 都参与原 FFTS 配对，不能把一个lane的进度当整组完成。

## 审查和验收

1. 独立审查双槽地址、跨query、无效块、两个AIV异步进度及尾部事件数。
2. 无设备编译完整根、实际ABI及artifact加载，查生成物资源/事件。
3. 小型8K/B32成对完整FULL graph计时；无收益停止，不做完整精度扩展。
4. 收益成立后Native A/B八边界及ABA；必须覆盖T>24同核多query、短/混query及失效块。
5. 反向捕获及128K单层复测后再考虑和其它优化组合。组合要重新测试，不能相加单项收益。
6. 通过后的正式候选再验真实8K/B40→128K/B24两轮192 token及接受事件。

每项设备任务外层780秒、内部750秒；保留原基准和SDK。当前只有独立诊断源码，尚不宣称性能或精度已通过。
