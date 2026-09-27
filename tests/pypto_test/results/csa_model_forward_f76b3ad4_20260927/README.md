# f76b3ad4整模型纯decode forward七档（采集中）

用户要求当前性能版整模型forward对照，并指出旧结论可能受PTO长尾影响。
固定算子及测试源码`.cache/csa-cache-f76b3ad4`；不把当前单层CSA或旧版本全模型数据写成当前forward。
`../csa_cache_accuracy_20260927/`的两档单卡隔离与16卡token/DSpark看护已通过；当前执行这里的性能采集。

矩阵：128K的B4/8/16，8K的B16/24/32/40。每个上下文两侧各加载一次模型，在同次加载里扫描batch。
复用已有audit通过的真实prefill bank，不重建权重、bank或做hash扫描。
正式W8A8、TP1/DP=EP16、出5验6、EPLB关闭、mode2两侧相同；性能版atomic1、Native确定性0/HCCL=false。
沿用上一轮模型扫描容量40、捕获24/48/96/144/192/240、128K预算256/8K预算400，避免把配置变化当作算子收益。

主指标：每个rank丢弃前8步，紧接连续10步`_model_forward`设备事件；16rank等权均值。
同时报告同组样本的p50/p95/max以及每步最慢rank分布，检查EP16放大的拖尾，不裁异常样本。
不计初始化、编译、metadata准备、logits、采样、草稿、步间等待或完整decode周期。
现有performance入口会另采3步Level0 trace用于CSA区间及拖尾定位；该独立采集不进入主计时。
每档仍核对逐token和DSpark统计，不用单层零差异代替模型看护。

[16卡采集命令](run_context.sh)、[两上下文顺序入口](run_all.sh)。
任务`task_20260927_223522_77333427523`在同一设备组顺序跑8K、128K，精度任务已退出后才提交。
[只读汇总](collect.py)复用既有rank有效性检查，不等trace离线解析即可报告纯forward；
`--available`只纳入Native/PTO各16rank都测完的档位，不拼旧版数据，不截取部分rank。
结果尚未产生。历史旧版本整模型报告仅见`../csa_six_case_profiles_20260926/`，不混入当前表。
