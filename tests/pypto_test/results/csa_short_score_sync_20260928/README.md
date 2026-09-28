# 8K/B16 相邻 CSA 波动与短档 Score 整组准入

用户指出的765/807μs已在现有模型图中精确找到；波动发生在设备CSA内部。
当前尚无证据把这42μs全部归给Score，单变量候选已编译通过并排队，生产算子未改。

## 已有图的核实

用户提供的文件是
[03单层泳道](../../downloads/DSV4_7cases_20260928_30f2b228/04_8K_B16_03_PTO_Swimlane_SingleCSA_SyntheticHistory.json)。
该文件只有一次根调用，Worker首尾784.60μs，包含正式layer4权重与合成历史。
相邻CSA来自同目录的
[02整模型图](../../downloads/DSV4_7cases_20260928_30f2b228/04_8K_B16_02_PTO_PyTorch_FullModel_Rank00.json)，
算子30f2b228、TP1/DP=EP16、mode2/atomic0/det0、EPLB关闭。
两份图是独立采集，不能把单层泳道中的task直接映射为模型某一层的内部事件。

同一次第3步主模型图内，第12层和第14层是相邻的两个CSA（中间还有其他模型计算）：

| 区间 μs | 第12层 | 第14层 | 差值 |
| --- | ---: | ---: | ---: |
| aicore_kernel_mode_0_mix_aic | 765.84 | 807.84 | +42.00（+5.48%） |
| Simpler AICPU根执行区间 | 776.62 | 814.86 | +38.24（+4.92%） |

设备事件起点分别为1790552537949023.455、1790552537951926.975μs；
同一Model Id 46、同一设备stream 77，Task Id分别16、19。AICPU在stream78。
两个CSA期间rank0图只显示配套的根执行和EVENT_WAIT/NOTIFY_WAIT，没有其他计算或HCCL kernel重叠。
这排除了“42μs只是两次CSA调用之间空隙”的读图解释；未测其他rank资源争用，不能据此排除所有外部影响。

3步×21个CSA的设备AIC区间：min750.42、均值778.10、P50 776.82、P95 802.22、max808.10μs。
根AICPU本体：均值788.52、P50 786.30、P95 811.60、max818.48μs。
这里本体去掉首层metadata，与七档完整CSA均值790.25μs的范围不同。
以上均为独立profile采样，不能替代无profiler正式10步forward。

## Score已有证据与边界

03单层泳道中，24个AIC和48个AIV各执行一份Score，没有同核重复派发或漏核。
AIC实际kernel启动分散14.04μs，AIV分散13.94μs；AIC核内31.68～41.76μs。
因此存在启动不齐，但这一窗口不足以证明模型相邻层多出的42μs来自Score。

另一组同e58算子、同输入的两个既有窗口中，Worker首尾770.34/788.48μs；
较慢窗口Score的AIC启动反而更齐（0.34对13.26μs）。互斥分段如下：

| μs | 770.34窗口 | 788.48窗口 | 差值 |
| --- | ---: | ---: | ---: |
| 首Worker→norm结束 | 97.16 | 86.08 | −11.08 |
| norm结束→Sparse首次接收 | 317.90 | 345.96 | +28.06 |
| Sparse首次接收→merge结束 | 172.34 | 177.58 | +5.24 |
| merge结束→末Worker | 182.94 | 178.86 | −4.08 |

这组是单卡layer4、det1的独立证据，不是模型12/14层分账；说明还须检查Score上游与后续关键链，
不能只凭启动更整齐就认定本体和P95改善。

## 定向对照

基底生产算子e58ddc94；候选工作树`.cache/csa-short-score-sync-e58ddc94`基于8da16784（后续仅文档提交）。
只把短档Cube Score的两个分支`sync_start=False→True`，`allow_early_resolve=True`保持，长档与Vector不变。
数学、任务数、分组和cache布局不改。它区别于旧R11同时修改整组准入与消费者提前释放的试验，
也区别于已否定的Indexer query整组准入；旧结论不能直接代替本次单变量结果。

完整CPU lowering/PTOAS/CCE/链接通过，[候选补丁](candidate.patch)、[编译入口](compile.py)、[编译日志](compile.log)。
设备任务`task_20260928_112109_314698014843`：仅8K/B16/S6、layer4、mode2/atomic0/det0，
同卡两侧各5预热/100次无profiler图计时，独立各4个DFX窗口，复用已有fixture和比较器。
保留全部样本，检查均值/P50/P95/max、Score分配/启动/结束、Sparse启动及完整Worker首尾。
PTO八类状态与图重放要求精确，Native det0只作控制，不要求其跨进程浮点逐bit一致。

先核实是否能改善当前短档波动，有收益才扩大到受影响短档与真实EP16；
不因为单窗口启动变齐就合入，不将短档改动冒充长档的新成绩。
原HC候选任务`task_20260928_103131_11716825293`在pending时取消，未执行；
HC源码与编译产物保留，按用户优先级等待本问题收尾后恢复。

[已有图逐层数据](existing_trace.json)、[只读提取脚本](inspect_existing.py)、
[设备命令](run_layer.sh)、[待结果齐备后运行的汇总脚本](summarize.py)。
