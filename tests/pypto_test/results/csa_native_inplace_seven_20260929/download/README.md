# 当前七档profiling下载包

28份原始JSON：每档Native主性能、PTO主性能、PTO泳道及Native核内诊断各一份。
01：Native SuperKernel开、static compile开；04：SuperKernel关、static compile开，仅分析核内。
两套Native均显式torch.compile backend=npugraph_ex、dynamic=False、fullgraph=True、inplace_pass=True。
03：预先固定第4个DFX窗口（window_3），不挑最快窗口；全部四窗仍在原目录。
生产算子55b89ee26c826c2e8a9190369ff421d542644eb5，CANN9.2/mode2/det0，PTO atomic0，layer4真实权重及独立合成历史。
范围为HC_pre+norm+CSA+HC_post；不是16卡模型forward或token/DSpark验收。
主性能表来自5预热/20次无profiler设备事件，独立profile不能替代正式样本。
Native主性能中的SuperKernel包含多个算子，不可标成单个QLI或Sparse核时。

[性能及核内对照](RESULTS.md)、[任务与pipeline明细](TASKS.md)、[原文件映射](sources.json)、[精简证据](summary.json)。
JSON直接复制，未修改或拼接事件，未把诊断组计时混入主性能表。
