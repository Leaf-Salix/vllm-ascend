# 全有效 KV 块省去 UB 清零先导

未纳入正式源码。仅测8K/B40，基线为同轮双query合并规约候选，
其 qk_pv AIC 四窗口均值325.94–342.47 μs、本体1470.46 μs、Native1418.81 μs。
本候选 qk_pv 为320.15–334.00 μs，本体1436.35 μs、Native1419.66 μs；
核内范围重叠，也未优于 V10 本体1428.20 μs，不认定稳定收益或继续七档扩测。

plan 的 compressed valid_block_mask 从0/1扩为0/1/2，2表示整块均有效；Cube仍按>0判断。
AIV使用原位 fillpad：全有效块不清零，其他块清零后再按已有有效索引搬运。
不改变候选顺序、概率或矩阵算术。直接条件分配零块导致UB 213248>188416字节，
已在CPU阶段改为同一缓冲的fillpad_inplace，ABI、PTOAS及AICPU C++编译通过。
没有改编译器/ISA，也没有让编译失败版本上卡。

单卡layer4正式权重＋合成历史，S6/TP1/mode2/atomic1/确定性0，EPLB关闭。
预热5次/20次无profiler采样，另4个DFX窗口。
任务task_20260927_150252_14677743863退出0，保护区/索引结构通过、非有限值0，
Top-K替换901，max_abs=0.03125，RMSE=0.0032928496；零容差FAIL。
[统计和原始路径](report.json)；[Sparse候选](candidate.patch)；[本轮Indexer基底](indexer_base.patch)。
