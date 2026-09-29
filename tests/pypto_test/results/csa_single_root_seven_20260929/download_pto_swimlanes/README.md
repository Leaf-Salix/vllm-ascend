# 本轮七档PTO单次CSA泳道

性能版算子提交：2ed8ae2e；七档使用同一冻结私有包。
CANN9.2、NZ mode2、deterministic=0、atomic_add=0，单卡layer4（第二个CSA层）。
真实层权重及合成历史KV；每份JSON含一次HC_pre+norm+CSA+HC_post图重放。
统一取第4个DFX窗口（window_3），直接复制原始merged JSON，不改事件、不挑最快窗口。
文件可用Perfetto或Chrome tracing打开；原始四窗与PyTorch profile路径见SOURCES.tsv。

| 文件 | 档位 | 来源 |
| --- | --- | --- |
| [01_128K_B4_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json](01_128K_B4_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json) | 128K/B4 | 本轮补测 |
| [02_128K_B8_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json](02_128K_B8_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json) | 128K/B8 | 本轮补测 |
| [03_128K_B16_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json](03_128K_B16_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json) | 128K/B16 | 复用本轮同源码两档A/B的候选 |
| [04_128K_B24_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json](04_128K_B24_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json) | 128K/B24 | 本轮补测 |
| [05_8K_B16_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json](05_8K_B16_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json) | 8K/B16 | 本轮补测 |
| [06_8K_B24_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json](06_8K_B24_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json) | 8K/B24 | 复用本轮同源码两档A/B的候选 |
| [07_8K_B32_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json](07_8K_B32_PTO_Swimlane_2ed8ae2e_SingleCSA_SyntheticHistory.json) | 8K/B32 | 本轮补测 |

## 独立正式计时参考

单位μs，5次预热后20次设备事件；与DFX独立采集，泳道开销不能当成正式CSA性能。
Native复用CANN9.2的最新标准基线：npugraph_ex、dynamic=False、inplace_pass/static/SuperKernel开启。
两侧采样来自不同任务，不用于归因单项优化收益；不是16卡模型forward或token/DSpark验收。

| 档位 | Native均值 | PTO均值 | PTO耗时变化 | PTO P95 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 748.30 | 650.23 | -13.11% | 662.62 |
| 128K/B8 | 859.56 | 746.70 | -13.13% | 760.54 |
| 128K/B16 | 1130.85 | 976.95 | -13.61% | 998.52 |
| 128K/B24 | 1281.89 | 1249.67 | -2.51% | 1271.48 |
| 8K/B16 | 757.48 | 781.51 | +3.17% | 799.36 |
| 8K/B24 | 915.37 | 956.56 | +4.50% | 981.44 |
| 8K/B32 | 1062.08 | 1065.57 | +0.33% | 1087.72 |
