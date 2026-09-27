# 十轮调度：第2轮，Indexer Q整组准入

基底2dd51f15；仅Indexer Q投影24个AIC block增加sync_start，核内、块数、精度入口均不变。
上游2164563同类任务无sync_start；历史727.98 μs图该任务启动分散约1.4 μs，当前基底14.1–72.18 μs，故检验整组准入。
CPU完整根/PTOAS/AICPU编译通过，task_20260927_182348_298761627623退出0。
2026-09-27、A3 device0、8K/B16/S6/TP1、正式第4层权重+合成输入历史、第二CSA层metadata复用，mode2/atomic1/deterministic0、EPLB关。
5预热/20次无profiler计时，另4个DFX窗口。

| 指标 μs | 基底 | 候选 |
| --- | ---: | ---: |
| CSA本体均值 | 793.658 | 796.982 |
| 本体p50 | 790.840 | 791.800 |
| 本体p95 | 813.520 | 825.080 |
| Native均值 | 923.928 | 917.683 |
| 完整PTO均值 | 1106.036 | 1110.346 |
| Indexer Q启动分散 | 14.10–72.18 | 0.34–0.86 |
| Indexer Q首次开始 | 169.62–199.82 | 161.80–263.14 |
| Indexer Q末尾结束 | 199.30–265.50 | 191.04–286.64 |
| Top-K merge末尾结束 | 405.28–426.70 | 403.02–451.36 |

各窗首Worker归零。整组启动确实紧凑，但需要等齐资源；部分窗口整个任务更晚，Top-K尾部也更晚，本体与p50/p95均未改善，撤回。
不能把泳道整齐当成加速。保护区/Top-K结构通过、非有限值0，Native输出max_abs0.03125、RMSE0.003320242、Top-K替换366，零容差仍FAIL。
与过去Q反量化/Score整组启动是不同任务；本轮也不据此断言所有sync_start均无效。

[补丁](candidate.patch)、[设备命令](run.sh)、[统计](summarize.py)、[完整数据](report.json)。
[原始泳道](../../csa_split_optimization_20260927/round02_idx_q_sync/h8192_b16/swimlane/dfx/merged_swimlane.json)。
[原始计时](../../csa_split_optimization_20260927/round02_idx_q_sync/h8192_b16/timing/report.json)。
