# QR/KV 自适应分组：统一七档 EP16 对照

源码：30f2b228 的性能算子，冻结于 `.cache/csa-qa-adaptive-88d0744f`。
128K/B4、B8、B16 与 8K/B16、B24、B32、B40；TP1、DP=EP16、DSpark 出5验6、EPLB关闭。
两侧 NZ mode2，部署确定性关闭；PTO atomic0，原始 Native cache 物理布局。
预热8步后10步 forward，独立3步 PyTorch profiling；初始化和编译不计入。
每个上下文 PTO 先运行，Native 后运行；复用同一离线 bank，不重建。

状态：task_20260928_072337_13373713806 已完成，退出0；统一七档正式结果已齐。

七档独立单层DFX任务：task_20260928_074745_190439310251，使用同源码、mode2/atomic0/det0，退出0。

原DFX排队任务task_20260928_072622_137973025089执行前取消，顺延至单档EP16入场诊断之后；未丢弃已执行样本。


## 正式结果与未关闭问题

七档forward均值快1.04%～7.11%，逐步最慢rank均值全部下降，67/70步更快；
573440输出token零差异，112组rank DSpark一致，正式10步位置一致。
[完整结果](model/RESULTS.md)、[224份rank记录的汇总](model/forward.json)。

128K/B8和B16的PTO P95为57.684/83.143ms，分别高于Native57.370/72.637ms；尾部验收未通过。
复用同一次正式事件可见rank4/rank6相对晚进入forward约3.980/14.060ms，其余rank耗时相应增加。
[入场证据与限制](ARRIVAL.md)。不剔除异常点，不从正式均值中扣除等待。
单档主机/GC诊断另行排队，保持同一算子。[诊断入口](../csa_forward_entry_20260928/README.md)。

14份模型rank0 PyTorch JSON已离线导出，全16rank原始profile保留；CPU解析结束时没有与本矩阵正式计时重叠。
[独立profile的CSA/Attention/FFN分段](model/MODEL_GAP.md)。七档CSA均值低9.45%～23.30%，
8K/B16完整CSA均值790.25μs、中位786.30μs，仍未达到750μs；不能以单层或profile收益替代正式forward。
PTO专家GMM任务相对Native增量−0.709～+0.169ms，没有旧atomic1七档普遍增加1ms以上的模式；
但未采实际专家索引，也不把另一轮profile变化作为正式10步的精确因果分账。


## 图表汇集已完成

[下载目录](download/README.md)约153MiB，共21份JSON：14份模型rank0 trace与7份单卡第二CSA层DFX。
[完整来源](download/manifest.json)、[当前核内/调度与上游](WORKER_GAP.md)、
[Native QR/KV定位](NATIVE_PROJECTION.md)。没有将单层合成输入DFX冒充模型同输入A/B。
当前8K/B16 Worker单窗口784.60μs；历史上游727.98μs配置不完整，只用作时序参考。
本轮新七档Indexer query启动分散21.56～119.22μs，两档调度候选单独验证，未混入此处源码。
