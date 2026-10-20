# 新10%目标：cannbot知识的使用范围

用户参考入口：https://gitcode.com/cann/cannbot-knowledge/tree/master/knowledge/ops。
网页访问工具未能打开目录；本轮实际阅读本地同仓知识，版本
`b64539127c5a0365723bd7c335b992fdc0d86326`，不声称它是此刻远端最新版本。
知识仓只读，不修改或提交其内容。

## 采用的思路

- `triton/optimizations/operator_optimization/loop_invariant_hoisting.md`：
  查找重复索引构造和加载，优先复用已生成数据。本轮PTOAS先复用索引前缀
  倍增构造；没有宣称已把索引Tile外提到整个CSA循环之外。
- `triton/optimizations/operator_optimization/vector_core_partition.md`：
  根据真实工作量检查分核。B1用G2有3组query，原8192叶至多15个AIC工作项；
  以4096半叶形成24个主历史工作项，同时统计新增query/系数加载和尾项。
  实际设备为24组AIC+双AIV，不照抄知识卡里的40/48核示意数字。
- `triton/optimizations/operator_optimization/latency_bound_tile_merge.md`：
  固定发射/同步成本可能大于算术成本，需要减少重复处理；本轮以删除压缩
  暂存复制为候选，而不是根据指令数量直接宣称完整CSA收益。
- `triton/optimizations/operator_optimization/workspace_decoupling.md`：
  同时核算存储和重算。直接压缩候选增大输出容量、缩小scratch，局部净省1KiB；
  完整符号编译与上卡验证分别检查资源及正确性。

## 没有直接套用的条目

- `preinitialized_output_optimization.md`：不能把每步必要的初始化偷偷移到
  Host且从计时中删除，也不能让图重放读取上一步未定义的scratch。
- `ascendc/optimizations/ol_57_sort_prefer_hardware_sort_api.md`及
  `ol_254_multi_core_outer_blocks_partitioning.md`标注950；本机A3不能直接套用
  算法阈值、核心数或案例加速比。当前已用Sort32/MrgSort，不存在改为硬件
  排序就自动获得文中倍率的前提。
- `p_p26_setflag_waitflag_fine_grained_event_sync_replaci.md`也标注950。
  精确依赖是可参考的原则，但本轮未删除CSA的DCCI/DSB、发布获取或业务数据同步。
- 知识中的attention案例只作为一般存储/调度方法参考；用户已放弃的Attention
  数学重构不因此重新启动。

## 测量约束

新目标冻结当前多task源码于`frozen_pypto_goal10`，见`SHARED_SOURCE_GOAL10.json`。
它包含此前多task刚合入的优化，不能用旧多task数据计算本轮10%。
公式固定为`(多task完整L1−动态SPMD完整L1)/动态SPMD完整L1`。

B1同源双实例、相同150tick的首轮出现408.40/395.22μs差异；
策略相同且变换前C++去注释/空白一致，已记录`GOAL10_B1_AA_AUDIT.json`。
这证明当前测量有需要解释的角色相关偏移，尚未证明是工作区地址导致。
因此对小差异谨慎判断，显著候选做角色交换复核，并记录地址描述符。
不共享工作区、不改采样行为，不以较快实例覆盖原始结果。
