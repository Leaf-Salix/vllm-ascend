# 六档 decode forward 对比与 profiling

| 历史长度 | 每卡 B | Native forward ms | PTO forward ms | PTO 耗时变化 | token / DSpark |
| ---: | ---: | ---: | ---: | ---: | --- |
| 131072 | 4 | 48.483 | 51.715 | +6.67% | 一致 / 一致 |
| 131072 | 8 | 58.681 | 67.259 | +14.62% | 一致 / 一致 |
| 131072 | 16 | 73.699 | 94.280 | +27.93% | 一致 / 一致 |
| 8192 | 24 | 79.210 | 79.279 | +0.09% | 一致 / 一致 |
| 8192 | 32 | 90.437 | 91.371 | +1.03% | 一致 / 一致 |
| 8192 | 40 | 102.236 | 106.241 | +3.92% | 一致 / 一致 |

正值表示 PTO 较慢。每 rank 预热后连续 10 次纯 forward，再对 16 rank 等权汇总；逐 rank 原始样本、最慢 rank 指标、CSA 层区间和 DSpark 统计见 summary.json。

每个 h{历史}_b{batch} 目录直接打开三个主文件：

- native_rank0_pytorch.json：Native 整模型 rank0，独立 Level0 三步。
- pto_rank0_pytorch.json：PTO 整模型 rank0，相同采集档位。
- pto_layer4_swimlane.json：PTO 第二个 CSA 层（model.layers.4）单卡 DFX。

其余 15 rank 的原始 PyTorch JSON 在 native_other_ranks/ 和 pto_other_ranks/。dfx_source/ 保留泳道原始任务、依赖与本次 JIT 的真实名称表。全部文件可直接下载后打开，不依赖服务器上的符号链接；files.json 记录来源和大小。

泳道采用正式第二层权重、合成输入/历史和 Native 页式 cache/state，复用前层 compact metadata；它与整模型 trace 的输入内容、物理页号及执行模式不同，只用于观察相同形状下的任务与调度。DFX/Level0 都独立于无 profiler 计时，不能把诊断耗时填入性能主表。

两侧 mode=2、TP1/DP=EP16、DSpark 出5验6、EPLB关、Native实际level0、HCCL=false；PTO性能版atomic=1。容量40、捕获24/48/96/144/192/240；128K预算256、8K预算400。本轮没有精度版性能数据，也不代替 H8192/B16 的750微秒目标或完整数值验收。
