# 71153bb3固定归约：统一源码七档EP16确认

两档受影响shape的atomic0干预已通过：128K/B8正式快0.85%，8K/B16快4.75%，token/DSpark一致。
本轮覆盖全部七档并把执行顺序改为PTO后Native，确认小幅优势不依赖固定的Native先测顺序。
没有改bank、权重、容量、NZ或验收窗口，没有混入尚无明确merge核内收益的UB Top-K候选。

源码仍`.cache/csa-forward-boundary-71153bb3`，生产算子71153bb3，只使用已有atomic0开关；
双方mode2、det0、HCCL=false、TP1/DP=EP16、出5验6、EPLB关，原Native cache与流程不变。
128K/B4/8/16预算256，8K/B16/24/32/40预算400；固定容量40，capture24/48/96/144/192/240。
各侧每种history只初始化一次模型，按batch扫描。8步warmup后连续10步无profilerforward，
独立3步profile保留所有rank原始数据；token、DSpark、位置、P95/max和逐步慢卡一起检查。
此次先完成全部七档才调整默认，没有把代表档通过当成全矩阵通过。

[运行命令](run_model.sh)、[严格收集器](collect_model.py)。

任务：`task_20260928_051900_412378912384`，完成，退出0。

## 七档正式结果与采用策略

七档全部通过，正式forward快1.35%～4.45%；573440输出token零差异、112组rank的DSpark和10步请求位置一致。
P95、最大值和逐步最慢rank均下降；8K四档PTO P95/P50为1.014、1.012、1.017、1.020。
[完整均值、P95、最慢rank表](model/RESULTS.md)、[全部rank样本和实际配置](model/forward.json)。

性能版采用默认atomic0，精度版保持默认atomic1；显式环境变量0/1可覆盖，须在导入/编译前设置。
默认变更仅选择本轮已实测策略，不改变核内实现；9项CPU配置回归通过。
Native cache分配和流程保持，PTO内部消化分页读取，不需外部拆分或写回。
750μs目标、精度版后续性能迁移和逐层误差分析仍未关闭。

## Profile交付

[CPU导出](export_profiles.sh)、[七档PTO DFX采集](run_swimlanes.sh)、[汇集脚本](bundle_profiles.py)。
DFX任务`task_20260928_054335_24490514569`已完成退出0，同71153bb3/atomic0，第二CSA层的真实权重与合成历史。
每档只补一个窗口，不重复正式模型计时。14份模型rank0 PyTorch JSON与7份单卡DFX已汇集到[download/](download/README.md)，
输入和计时范围单列；其他rank原始profile仍保留，不把DFX当整模型耗时。

[模型CSA/FFN/GMM差距](model/MODEL_GAP.md)、[当前核内/调度与上游参考](WORKER_GAP.md)，
由[只读汇总脚本](summarize_profiles.py)生成。模型CSA完整区间均下降7.88%～20.85%；
专家GMM每step增量为−0.749～+0.135ms，旧atomic1七档为+1.01～+1.68ms。
这与固定规约方向一致，但未采本轮实际路由索引，且profile与正式轮独立，不能宣称唯一因果或精确分账。

当前8K/B16 DFX Worker窗口784.30μs，历史上游727.98μs；分段差距主要在norm前与norm到Sparse启动之间。
该历史图缺完整配置，只是调度参照，不等于等输入算法差额，也不代替750μs整模型验收。

[七档Native Q/KV投影与当前核内对照](NATIVE_PROJECTION.md)只读取上述已有模型trace，
按QA的归一化消费者及辅助stream中KV→head投影的次序匹配；441个CSA区间计数和stream检查通过。
保留实际kernel名称及任务号，处理CANN输出ACL名称/编译后名称两种情况，不新增设备采样。
