# CANN 9.2：e33d842a七档单卡结果

同一卡、同一已保留算子；每侧5次预热/20次无profiler图计时。单位μs。
冻结e33d842a，不包含采集期间新增的3b27c7fd WO_A NZ布局修正。
Native算术差异单列于matrix.json；自重放通过不等于Native逐bit或整网token/DSpark通过。

| 档位 | Native均值 | PTO均值 | 变化 | Native/PTO P95 | Native/PTO最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 863.12 | 730.70 | -15.34% | 866.66/744.76 | 867.20/750.00 |
| 128K/B8 | 997.25 | 828.19 | -16.95% | 1002.30/842.22 | 1003.20/862.96 |
| 128K/B16 | 1279.55 | 1078.85 | -15.69% | 1283.16/1087.12 | 1283.18/1104.04 |
| 8K/B16 | 924.75 | 804.29 | -13.03% | 928.66/821.16 | 929.18/824.10 |
| 8K/B24 | 1108.69 | 969.46 | -12.56% | 1114.26/981.06 | 1124.40/984.94 |
| 8K/B32 | 1249.61 | 1140.21 | -8.75% | 1248.48/1166.64 | 1382.56/1167.08 |
| 8K/B40 | 1387.34 | 1315.09 | -5.21% | 1396.74/1336.04 | 1397.16/1349.60 |

各history内batch等权后，七三耗时变化-14.161%。
不使用权重覆盖P95异常；不与旧9.0或不同候选拼表。

## Native独立kernel profile

以下为单次独立profile的原始duration/aicore_time/aiv_time，非20次正式计时均值。
QLI包含本地Top-K/最终归并，Sparse包含其内部规约；与PTO拆分任务范围不同，不直接相减归因。
aicore_time/aiv_time由PMU周期按block数和波次折算，不是纯算术时间或最慢核，定义见[METRICS.md](METRICS.md)。

| 档位 | Native kernel | Duration | aicore_time | aiv_time |
| --- | --- | ---: | ---: | ---: |
| 128K/B4 | VllmQuantLightningIndexer | 235.640 | 66.855 | 66.536 |
| 128K/B4 | SparseAttnSharedkv | 64.520 | 58.186 | 62.117 |
| 128K/B8 | VllmQuantLightningIndexer | 236.340 | 124.422 | 124.119 |
| 128K/B8 | SparseAttnSharedkv | 110.540 | 101.659 | 105.164 |
| 128K/B16 | VllmQuantLightningIndexer | 355.700 | 240.925 | 240.421 |
| 128K/B16 | SparseAttnSharedkv | 182.820 | 176.008 | 179.262 |
| 8K/B16 | VllmQuantLightningIndexer | 53.400 | 38.58 | 38.037 |
| 8K/B16 | SparseAttnSharedkv | 116.140 | 108.018 | 110.984 |
| 8K/B24 | VllmQuantLightningIndexer | 54.940 | 52.448 | 52.097 |
| 8K/B24 | SparseAttnSharedkv | 180.840 | 170.281 | 172.635 |
| 8K/B32 | VllmQuantLightningIndexer | 92.300 | 65.106 | 64.546 |
| 8K/B32 | SparseAttnSharedkv | 233.800 | 220.917 | 223.541 |
| 8K/B40 | VllmQuantLightningIndexer | 92.600 | 77.205 | 76.626 |
| 8K/B40 | SparseAttnSharedkv | 293.200 | 278.789 | 281.158 |

## PTO独立四窗口DFX

每格为四个窗口指标的均值。最慢核指各窗口kernel最大值；核时包含内部等待。
包络含启动分散，不能称为纯调度时间；各任务有交叠，不累加成整层时长。

| 档位 | Task | 核内均值 | 最慢核 | 包络 | 启动分散 | 单核最多份数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | indexer_score_topk_native_pair_aic | 134.262 | 144.670 | 147.840 | 1.720 | 1 |
| 128K/B4 | indexer_score_topk_native_pair_aiv | 142.025 | 145.920 | 149.025 | 1.965 | 1 |
| 128K/B4 | indexer_topk_query_merge | 6.743 | 10.765 | 11.585 | 2.910 | 1 |
| 128K/B4 | qk_pv_aic | 42.883 | 45.400 | 57.495 | 3.885 | 1 |
| 128K/B4 | qk_pv_aiv | 44.757 | 47.370 | 59.665 | 4.160 | 1 |
| 128K/B4 | merge_norm | 12.474 | 15.120 | 29.000 | 0.755 | 1 |
| 128K/B8 | indexer_score_topk_native_pair_aic | 166.674 | 172.290 | 175.095 | 3.965 | 1 |
| 128K/B8 | indexer_score_topk_native_pair_aiv | 176.767 | 183.665 | 186.520 | 2.730 | 1 |
| 128K/B8 | indexer_topk_query_merge | 6.665 | 9.795 | 12.930 | 6.295 | 1 |
| 128K/B8 | qk_pv_aic | 77.515 | 80.920 | 90.965 | 3.975 | 1 |
| 128K/B8 | qk_pv_aiv | 79.484 | 82.730 | 93.255 | 4.095 | 1 |
| 128K/B8 | merge_norm | 16.615 | 19.335 | 30.405 | 0.755 | 1 |
| 128K/B16 | indexer_score_topk_native_pair_aic | 242.765 | 248.060 | 251.220 | 1.475 | 1 |
| 128K/B16 | indexer_score_topk_native_pair_aiv | 263.622 | 270.990 | 274.150 | 2.245 | 1 |
| 128K/B16 | indexer_topk_query_merge | 10.943 | 14.045 | 16.900 | 6.020 | 1 |
| 128K/B16 | qk_pv_aic | 142.638 | 147.160 | 159.205 | 0.380 | 1 |
| 128K/B16 | qk_pv_aiv | 144.507 | 149.145 | 161.405 | 0.810 | 1 |
| 128K/B16 | merge_norm | 25.145 | 27.500 | 41.465 | 0.740 | 1 |
| 8K/B16 | indexer_score_topk_native_pair_aic | 42.381 | 44.715 | 58.535 | 17.485 | 2 |
| 8K/B16 | indexer_score_topk_native_pair_aiv | 46.791 | 50.180 | 64.300 | 18.810 | 2 |
| 8K/B16 | indexer_topk_query_merge | 8.703 | 9.990 | 35.540 | 0.755 | 1 |
| 8K/B16 | qk_pv_aic | 125.400 | 128.030 | 139.770 | 0.405 | 1 |
| 8K/B16 | qk_pv_aiv | 127.196 | 129.930 | 141.965 | 1.065 | 1 |
| 8K/B16 | merge_norm | 24.152 | 27.025 | 38.915 | 0.790 | 1 |
| 8K/B24 | indexer_score_topk_native_pair_aic | 28.157 | 30.080 | 47.350 | 14.180 | 1 |
| 8K/B24 | indexer_score_topk_native_pair_aiv | 43.124 | 48.770 | 65.705 | 14.170 | 1 |
| 8K/B24 | indexer_topk_query_merge | 10.157 | 11.535 | 35.245 | 0.785 | 1 |
| 8K/B24 | qk_pv_aic | 174.601 | 177.305 | 190.820 | 0.375 | 1 |
| 8K/B24 | qk_pv_aiv | 176.526 | 179.145 | 193.050 | 0.750 | 1 |
| 8K/B24 | merge_norm | 31.761 | 34.480 | 50.375 | 0.725 | 1 |
| 8K/B32 | indexer_score_topk_native_pair_aic | 74.183 | 76.625 | 102.555 | 25.125 | 2 |
| 8K/B32 | indexer_score_topk_native_pair_aiv | 77.959 | 82.320 | 108.285 | 26.480 | 2 |
| 8K/B32 | indexer_topk_query_merge | 10.457 | 12.215 | 51.905 | 0.715 | 1 |
| 8K/B32 | qk_pv_aic | 226.647 | 230.600 | 243.965 | 2.700 | 1 |
| 8K/B32 | qk_pv_aiv | 228.631 | 232.415 | 246.265 | 2.960 | 1 |
| 8K/B32 | merge_norm | 35.741 | 38.595 | 56.025 | 1.275 | 1 |
| 8K/B40 | indexer_score_topk_native_pair_aic | 53.890 | 67.635 | 91.320 | 23.070 | 2 |
| 8K/B40 | indexer_score_topk_native_pair_aiv | 66.212 | 85.170 | 109.645 | 33.860 | 2 |
| 8K/B40 | indexer_topk_query_merge | 12.379 | 13.990 | 70.295 | 0.750 | 1 |
| 8K/B40 | qk_pv_aic | 280.353 | 283.145 | 296.510 | 0.370 | 1 |
| 8K/B40 | qk_pv_aiv | 282.274 | 285.090 | 298.645 | 0.745 | 1 |
| 8K/B40 | merge_norm | 45.329 | 48.625 | 60.380 | 0.685 | 1 |

## 原始JSON

每档两份PyTorch JSON及四份PTO泳道；均为独立采样。

- [128K/B4 native PyTorch](layer/h131072_b4/profile/native/liteserver-hps-365a-00001_2304358_20260928224717825_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [128K/B4 pto PyTorch](layer/h131072_b4/profile/pto/liteserver-hps-365a-00001_2304358_20260928224723463_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [128K/B4 PTO泳道0](layer/h131072_b4/swimlane/dfx/merged_swimlane.json)
- [128K/B4 PTO泳道1](layer/h131072_b4/swimlane/dfx/window_1/merged_swimlane.json)
- [128K/B4 PTO泳道2](layer/h131072_b4/swimlane/dfx/window_2/merged_swimlane.json)
- [128K/B4 PTO泳道3](layer/h131072_b4/swimlane/dfx/window_3/merged_swimlane.json)
- [128K/B8 native PyTorch](layer/h131072_b8/profile/native/liteserver-hps-365a-00001_2327476_20260928224936669_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [128K/B8 pto PyTorch](layer/h131072_b8/profile/pto/liteserver-hps-365a-00001_2327476_20260928224942438_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [128K/B8 PTO泳道0](layer/h131072_b8/swimlane/dfx/merged_swimlane.json)
- [128K/B8 PTO泳道1](layer/h131072_b8/swimlane/dfx/window_1/merged_swimlane.json)
- [128K/B8 PTO泳道2](layer/h131072_b8/swimlane/dfx/window_2/merged_swimlane.json)
- [128K/B8 PTO泳道3](layer/h131072_b8/swimlane/dfx/window_3/merged_swimlane.json)
- [128K/B16 native PyTorch](layer/h131072_b16/profile/native/liteserver-hps-365a-00001_2359619_20260928225200330_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [128K/B16 pto PyTorch](layer/h131072_b16/profile/pto/liteserver-hps-365a-00001_2359619_20260928225206727_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [128K/B16 PTO泳道0](layer/h131072_b16/swimlane/dfx/merged_swimlane.json)
- [128K/B16 PTO泳道1](layer/h131072_b16/swimlane/dfx/window_1/merged_swimlane.json)
- [128K/B16 PTO泳道2](layer/h131072_b16/swimlane/dfx/window_2/merged_swimlane.json)
- [128K/B16 PTO泳道3](layer/h131072_b16/swimlane/dfx/window_3/merged_swimlane.json)
- [8K/B16 native PyTorch](layer/h8192_b16/profile/native/liteserver-hps-365a-00001_2446463_20260928225427235_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [8K/B16 pto PyTorch](layer/h8192_b16/profile/pto/liteserver-hps-365a-00001_2446463_20260928225432768_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [8K/B16 PTO泳道0](layer/h8192_b16/swimlane/dfx/merged_swimlane.json)
- [8K/B16 PTO泳道1](layer/h8192_b16/swimlane/dfx/window_1/merged_swimlane.json)
- [8K/B16 PTO泳道2](layer/h8192_b16/swimlane/dfx/window_2/merged_swimlane.json)
- [8K/B16 PTO泳道3](layer/h8192_b16/swimlane/dfx/window_3/merged_swimlane.json)
- [8K/B24 native PyTorch](layer/h8192_b24/profile/native/liteserver-hps-365a-00001_2468625_20260928225644607_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [8K/B24 pto PyTorch](layer/h8192_b24/profile/pto/liteserver-hps-365a-00001_2468625_20260928225650076_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [8K/B24 PTO泳道0](layer/h8192_b24/swimlane/dfx/merged_swimlane.json)
- [8K/B24 PTO泳道1](layer/h8192_b24/swimlane/dfx/window_1/merged_swimlane.json)
- [8K/B24 PTO泳道2](layer/h8192_b24/swimlane/dfx/window_2/merged_swimlane.json)
- [8K/B24 PTO泳道3](layer/h8192_b24/swimlane/dfx/window_3/merged_swimlane.json)
- [8K/B32 native PyTorch](layer/h8192_b32/profile/native/liteserver-hps-365a-00001_2497044_20260928225904286_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [8K/B32 pto PyTorch](layer/h8192_b32/profile/pto/liteserver-hps-365a-00001_2497044_20260928225910355_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [8K/B32 PTO泳道0](layer/h8192_b32/swimlane/dfx/merged_swimlane.json)
- [8K/B32 PTO泳道1](layer/h8192_b32/swimlane/dfx/window_1/merged_swimlane.json)
- [8K/B32 PTO泳道2](layer/h8192_b32/swimlane/dfx/window_2/merged_swimlane.json)
- [8K/B32 PTO泳道3](layer/h8192_b32/swimlane/dfx/window_3/merged_swimlane.json)
- [8K/B40 native PyTorch](layer/h8192_b40/profile/native/liteserver-hps-365a-00001_2617000_20260928230130195_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [8K/B40 pto PyTorch](layer/h8192_b40/profile/pto/liteserver-hps-365a-00001_2617000_20260928230135916_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [8K/B40 PTO泳道0](layer/h8192_b40/swimlane/dfx/merged_swimlane.json)
- [8K/B40 PTO泳道1](layer/h8192_b40/swimlane/dfx/window_1/merged_swimlane.json)
- [8K/B40 PTO泳道2](layer/h8192_b40/swimlane/dfx/window_2/merged_swimlane.json)
- [8K/B40 PTO泳道3](layer/h8192_b40/swimlane/dfx/window_3/merged_swimlane.json)
