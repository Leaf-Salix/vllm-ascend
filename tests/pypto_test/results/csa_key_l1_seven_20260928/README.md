# 554b3bca阶段七档验收

冻结源码`.cache/csa-key-l1-seven-554b3bca`，性能版纳入独立Key L1池、Top-K UB和修正后的QR输入驻留。
不包含主工作树其他会话未提交的WO_A/Runner改动。当前目标优先128K，8K代价按长短变化率7:3评价；
七档按128K三档、8K四档各自等权后再7:3，P95、token、DSpark独立列出。

先单卡、再真实权重EP16；同权重、mode2、atomic0、det0，EPLB关闭。
单卡用第二个CSA层（layer4）真实权重/合成历史，反向物理页与逐物理行Indexer scale。
模型只计预热8步后连续10步纯decode forward，另采独立rank0三步PyTorch profile。
模型阶段启用已准备的metadata builder子区间诊断；它是嵌套主机观测，不扣除设备forward中的等待。

## 单卡覆盖

长短B16已对同一算子实现完成状态/图重放、20次计时和四窗口，直接复用：
[128K/B16](../csa_score_key_l1_20260928/h131072_b16/evidence.json)、
[8K/B16](../csa_score_key_l1_20260928/h8192_b16/evidence.json)。
该候选`decode_indexer.py`原样保留到554b3bca；其余生产算子与2d2f9ca0相同。
只新增可选模型主机观察，与单卡CSA实现无关，不将不同算术或不同策略版本拼表。
原任务与缺失短档DFX补采的来源、失败记录均保留；不重新跑两档以挑选更好样本。

[其余五档入口](run_remaining_layer.sh)：128K/B4、B8，8K/B24、B32、B40。
单卡任务`task_20260928_181207_305949327631`已提交；尚未取得五档结果。
每档一次Native/PTO计时、候选自重放/保护区及A→B→A，再采两个独立单根DFX窗口。
这五档是阶段覆盖，不重复四窗口单因素实验，也不额外导出大块cache状态供无用途比较。
Native零容差浮点/量化差异仍诊断记录，不冒充逐bit一致；最终token/DSpark由整模型验收。

阶段结束保留七档全部模型PyTorch JSON及PTO泳道，汇集下载目录。
本目录当前处于准备/单卡阶段，尚无554b3bca七档模型结果；2d2f9ca0模型数字不替代本轮。

[七档模型入口](run_model.sh)已准备，要求五档单卡原任务成功退出且分配16卡后才能执行，当前未提交模型任务。
长档一次sweep B4/8/16，短档一次sweep B16/24/32/40，长短交换Native/PTO执行顺序。
[forward收集](collect_model.py)复用逐rank token/DSpark、位置、配置及事件门禁，七档全通过才汇总7:3；
可通过`--available`读取已齐全档位，但不足七档时不输出阶段加权收益。
[主机及builder诊断](analyze.py)保留异常rank十步标记，不根据P95好看而删除尾部。
