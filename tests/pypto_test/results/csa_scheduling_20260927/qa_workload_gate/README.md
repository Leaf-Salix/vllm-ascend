# 小工作量恢复Compressor/Q_A交叠：未保留

基底2dd51f15，含O_A行列并行；pypto-lib最新版2164563的TP1路径让两个Compressor与Q_A交叠。
当前Q_A先行此前在B40本体获益，但未证明适用小档；本项先在当前B16重新采基线，再测试同源码候选。
候选用runtime `T>=144` 保留Q_A先行，小档仅沿用原late_dep，不依赖qa_tid；算术和任务网格不变。
完整CPU/PTOAS/AICPU编译通过，生成调度C++正确保留两个TaskId分支。

2026-09-27，A3 device0，8K/B16/S6/TP1，正式第4层权重及合成输入历史，第二CSA层metadata复用；
mode2/atomic1/deterministic0、EPLB关闭，5预热/20次无profiler计时，另4个DFX窗口。
基底task_20260927_181100_283132024459、候选task_20260927_181257_28483392968均退出0。

| 指标 μs | Q_A先行 | 小档交叠 |
| --- | ---: | ---: |
| CSA本体均值 | 793.658 | 794.975 |
| 本体p50 | 790.840 | 795.440 |
| 本体p95 | 813.520 | 807.840 |
| 同轮Native均值 | 923.928 | 912.977 |
| 完整PTO均值 | 1106.036 | 1100.612 |
| Q_A最后结束 | 140.26–147.64 | 186.56–202.38 |
| Attention Compressor投影最后结束 | 202.74–242.20 | 158.78–196.74 |
| Indexer Compressor投影最后结束 | 163.52–210.50 | 120.20–196.22 |
| Top-K merge最后结束 | 405.28–426.70 | 415.64–438.24 |

DFX每窗首Worker归零。Compressor提前，但Q_A明显推迟、最终Top-K链没有提前；本体均值/p50均未改善，
撤回，不保留未经证明的工作量分支，也不扩测长上下文或尾块。
当前与上游Q_A有64对16个Worker的任务划分差异，不能假定相同交叠策略必然最优。
保护区/Top-K结构通过、非有限值0；Native输出max_abs0.03125、RMSE0.003320339、Top-K替换366，零容差仍FAIL。

本轮基底B16均值793.66 μs，旧c7a52af5的832.26 μs含一次长尾，两者都保留，不能以基线变化冒称本项获益。
基底与候选数据分别可复现，不把旧七档不同核内版本的结果混进A/B。

- [补丁](candidate.patch)、[隔离编译](compile_candidate.py)、[基线命令](run_baseline.sh)、[候选命令](run_candidate.sh)。
- [统计脚本](summarize.py)、[数据](report.json)。
- [当前基底计时](../../csa_split_optimization_20260927/baseline_2dd51f15/h8192_b16/timing/report.json)。
- [基底泳道](../../csa_split_optimization_20260927/baseline_2dd51f15/h8192_b16/swimlane/dfx/merged_swimlane.json)。
- [候选计时](../../csa_split_optimization_20260927/qa_workload_gate/h8192_b16/timing/report.json)。
- [候选泳道](../../csa_split_optimization_20260927/qa_workload_gate/h8192_b16/swimlane/dfx/merged_swimlane.json)。
