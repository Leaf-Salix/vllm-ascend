# 下一大方向：保留当前数据路径的Score/Attention跨请求流水

## 为什么不能直接重跑旧候选

旧score_attention_interleave_current_20261019修复漏执行原KV之后，B16
全精确/图通过，但同轮正式811.10、候选861.06μs，回退约50μs。
这份结果完整保留，本轮没有新测该版本，也不能拿它与当前跨轮数字相减。

只读核对旧source_candidate_v2：实际交错路径调用attention_tick_stages.py、
score_vector_steps_g6.py，而不是旁边的完整attention_stream_stages.py。
后者文件与当前完全相同，不代表交错路径采用了当前实现。实际tick模块
没有MGATHER、PLAN_PHYSICAL_ROWS、LOCAL_ATTENTION_STATS；Score step
也没有SCALE_PAGE_COALESCING。旧README明确只使用逻辑plan/GM统计。
当前Score又已采用双QK早FIX/合并WS写回，不能直接保留旧M384 step计算。
indexer_group_stages与旧候选逐字相同，不能假称叶分配规则有了新变化。

## 下一次实现的实质区别与边界

首先从当前正式完整函数拆出可暂停的数学阶段，保持已采用的物理行plan/
MGATHER、区间local mi/li、连续scale以及N64早FIX Score；它们必须真正
接入活跃的step/tick调用，不能只搬文件或保留无效policy字段。刚结束的
N128候选不叠加进来。现有动态wave边界、真实root/plan发布和首wave原KV
修复可参考，但必须重新证明参与者可推进和完整资源生命周期。

目标仍是一个完整MIX，在不同请求之间重叠Cube主导Score与Vector供数较多
的Attention。当前带探针B28两大区间约330/305μs，说明值得研究整段重叠；
累计READY113μs不是免费窗口，不能据相加/相减预记可省量。Left重载、两套
状态共存、wave控制和新增跨波等待可能抵消收益，这是旧负例已经说明的风险。

准备时优先做三项具体门槛：
1. 实际活跃数学/页保护与当前版本一致，保留原有效块/尾query/FP32顺序；
   原5块/3槽的协议和替代2槽L1复用必须明确区分，不靠删除等待容纳数据。
2. 联合编译证明Left64KiB、Right64KiB、Acc128KiB、Mat512KiB及UB预算满足，
   保留所有真实覆盖等待；不使用冻结静态body，不扩大硬件容量。
3. 两代表档原完整三方五窗验收，只有一档净省>=20μs且另一档无明确退步
   才扩档；若完整收益不足则停止该组合，不重新包装旧861.06当新证据。

本文件只是下一方向的源码审查与实现门槛；尚未创建新交错候选、编译或
提交新设备任务。当前正式保持13259f92，无新性能结论，所有仓库禁止push。
