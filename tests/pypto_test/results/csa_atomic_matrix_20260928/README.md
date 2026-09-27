# 71153bb3固定归约：统一源码七档EP16确认

两档受影响shape的atomic0干预已通过：128K/B8正式快0.85%，8K/B16快4.75%，token/DSpark一致。
本轮覆盖全部七档并把执行顺序改为PTO后Native，确认小幅优势不依赖固定的Native先测顺序。
没有改bank、权重、容量、NZ或验收窗口，没有混入尚无明确merge核内收益的UB Top-K候选。

源码仍`.cache/csa-forward-boundary-71153bb3`，生产算子71153bb3，只使用已有atomic0开关；
双方mode2、det0、HCCL=false、TP1/DP=EP16、出5验6、EPLB关，原Native cache与流程不变。
128K/B4/8/16预算256，8K/B16/24/32/40预算400；固定容量40，capture24/48/96/144/192/240。
各侧每种history只初始化一次模型，按batch扫描。8步warmup后连续10步无profilerforward，
独立3步profile保留所有rank原始数据；token、DSpark、位置、P95/max和逐步慢卡一起检查。
当前不提前修改生产默认，也不把代表档通过当成全矩阵通过。

[运行命令](run_model.sh)、[严格收集器](collect_model.py)。

任务：`task_20260928_051900_412378912384`，运行中。
