# CANN9.2：128K/B24首次当前源码单卡对照

正式layer4权重/合成历史；mode2/atomic0/det0/S6，完整HC_pre→HC_post。
每侧预热5次，20次无profiler图计时，单位μs。

| 实现 | 均值 | P50 | P95 | 最大值 |
| --- | ---: | ---: | ---: | ---: |
| native | 1489.736 | 1490.110 | 1498.500 | 1500.380 |
| pto | 1362.396 | 1362.030 | 1374.220 | 1398.360 |

PTO完整CSA耗时变化-8.548%；新档自重放、图状态及保护区通过，Native数值差异单列。
不与旧七档拼成同轮结果；本轮未运行新的EP16。

## 核内与包络

四DFX窗口指标均值；含DMA/内部等待，非纯算术。

| Task | 核内均值 | 最慢核 | 包络 | 启动分散 | 单核最多份数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| indexer_score_topk_native_pair_aic | 383.890 | 401.000 | 404.310 | 1.945 | 1 |
| indexer_score_topk_native_pair_aiv | 412.053 | 420.480 | 423.895 | 2.725 | 1 |
| indexer_topk_query_merge | 14.260 | 16.665 | 19.360 | 4.385 | 1 |
| qk_pv_aic | 203.848 | 208.115 | 221.920 | 0.360 | 1 |
| qk_pv_aiv | 205.866 | 210.315 | 224.210 | 0.760 | 1 |
| merge_norm | 40.134 | 42.965 | 57.970 | 4.360 | 1 |
| proj_a_mm | 27.381 | 32.220 | 88.580 | 60.820 | 3 |
| proj_b_mm | 18.966 | 21.175 | 83.845 | 65.190 | 4 |
| hc_post | 22.886 | 25.295 | 46.470 | 0.680 | 1 |

## JSON

PyTorch profile与四DFX分别采集，不是同一次调用。

- [128K_B24_01_Native_PyTorch.json](download/128K_B24_01_Native_PyTorch.json)
- [128K_B24_02_PTO_PyTorch.json](download/128K_B24_02_PTO_PyTorch.json)
- [128K_B24_03_PTO_Swimlane_SingleCSA_SyntheticHistory.json](download/128K_B24_03_PTO_Swimlane_SingleCSA_SyntheticHistory.json)
- [PTO泳道窗口0](swimlane/dfx/merged_swimlane.json)
- [PTO泳道窗口1](swimlane/dfx/window_1/merged_swimlane.json)
- [PTO泳道窗口2](swimlane/dfx/window_2/merged_swimlane.json)
- [PTO泳道窗口3](swimlane/dfx/window_3/merged_swimlane.json)
