# 554b3bca：128K/B8的模型区间差异

本轮Native/PTO独立rank0三步profile；不与正式8步预热后10步forward混算。
两侧mode2/atomic0/det0，实际CANN event模式0/1。首层通信可能包含跨rank到达等待，
没有其他rank对应的同一步核内轨迹，不能将通信变慢直接判为CSA引起。

| 每步均值 | Native ms | PTO ms | PTO−Native ms |
| --- | ---: | ---: | ---: |
| CSA本体合计 | 21.363 | 18.373 | −2.991 |
| 其他Attention | 12.412 | 11.561 | −0.850 |
| FFN合计 | 23.841 | 25.218 | +1.377 |
| 区间外 | 0.134 | 0.278 | +0.145 |
| 主图区间 | 57.750 | 55.430 | −2.320 |
| 首层FFN | 0.913 | 2.483 | +1.569 |
| 首层MoeDistributeDispatchV2任务 | 0.464 | 2.041 | +1.577 |

前四个互斥区间之和为主图区间；后两行已包含于FFN，不能重复相加。
首层Dispatch三步任务耗时Native为665.96/347.82/377.24μs，PTO为1818.42/2751.92/1551.88μs。
同一首层两种专家GMM任务合计均值141.64→136.16μs，没有形成对应增加；
全部FFN的专家GMM任务合计4.750→4.813ms，任务时间也不是不重叠的关键路径。

这些数据表明该profile中有一部分CSA节省被首层通信耗时抵消；但profile主图区间仍快2.320ms，
正式十步forward则55.394→55.439ms（+0.08%）。两者轮次、边界和rank覆盖不同，
不能用本表算出正式forward的全部差额，也不能据此关闭B8无收益或入场迟到问题。

[全模型分项及原始边界](model_gap_rank0.json)、
[逐步逐层FFN任务记录](h131072/b8/ffn_breakdown_rank0.json)、
[正式forward](RESULTS.md)、[完整profile下载](../download/README.md)。
