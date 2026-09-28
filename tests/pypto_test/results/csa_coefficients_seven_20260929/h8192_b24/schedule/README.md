# 8K/B24：四窗口调度诊断

当前源码4ffccb7b；复用已完成的四个level-4窗口，没有追加设备执行。
每窗原始AICore/AICPU/解析行数及每任务block数量均通过官方解析器检查。

| 窗口 | 行数 | dispatch→FIN μs | AICore首尾 μs | Static CPM μs | Observed gap μs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1149 | 930.000 | 925.660 | 671.640 | 80.760 |
| 1 | 1149 | 935.660 | 931.760 | 693.620 | 77.500 |
| 2 | 1149 | 934.520 | 930.260 | 680.380 | 101.340 |
| 3 | 1149 | 922.580 | 919.140 | 737.480 | 106.280 |

完整Observed路径预先固定window_3，不选择最快窗口；[路径和阻塞证据](critical_path_summary.md)。
工具中的compute是逻辑任务对关键路径的覆盖，含SPMD启动跨度和核内等待，不等同纯算术。
Score/qk_pv等MIX任务在路径里以首个AIC函数名显示，但跨度含同一逻辑任务的AIC/AIV；分核计时看Worker表。
dummy缺失物理时戳的任务，完整data-ready/FIN归因置空；不采用工具对这些行的局部ready数值。

## Worker非重叠分段

历史727.98μs图输入/工具链不完整，且与当前形状不同；仅作结构参照，不是同输入性能对照。

| 分段 | 历史上游 μs | 当前四窗口均值 μs |
| --- | ---: | ---: |
| first_worker_to_norm_end | 66.620 | 103.740 |
| norm_end_to_sparse_receive | 317.140 | 357.065 |
| sparse_receive_to_merge_end | 184.900 | 246.000 |
| merge_end_to_last_worker | 159.320 | 221.070 |

所有时间与正式无profiler事件分开；不把两种采样相减，也不把多任务重叠时长相加。
全部任务、实际early派发和缺失前置ID见[evidence.json](evidence.json)。
