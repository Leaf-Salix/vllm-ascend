# 十轮调度：第4轮，O_A在merge期间提前准备

基底2dd51f15；仅merge_norm生产者开放allow_early_resolve，允许O_A预派发。数据依赖、算术、任务块数不变。
最新pypto-lib2164563和基底均未开启此标志；历史727.98 μs图merge结束到O_A首次开始约5.88 μs，当前5.62–7.32 μs。
2026-09-27，A3 device0、8K/B16/S6/TP1、正式第4层权重+合成输入历史、第二CSA层metadata复用，mode2/atomic1/deterministic0、EPLB关。
5预热/20计时，另4个DFX窗口，task_20260927_182955_30301211234退出0。

| 指标 μs | 基底 | 候选 |
| --- | ---: | ---: |
| 本体均值 | 793.658 | 788.269 |
| 本体p50 | 790.840 | 787.590 |
| 本体p95 | 813.520 | 805.460 |
| Native均值 | 923.928 | 920.142 |
| 完整PTO均值 | 1106.036 | 1088.642 |
| merge末尾→首O_A kernel | 5.62–7.32 | 4.34–4.80 |
| merge末尾→HC_post结束 | 177.76–183.46 | 178.80–185.92 |
| O_A平均setup | 0.58–0.63 | 5.16–6.64 |

预派发确实让启动隙略短，但更多等待移入setup，尾段整体没有缩短。无profiler均值约−0.68%，Native约−0.41%；
当前不足以确认局部策略带来完整加速，撤回，不扩测。完整PTO变化不能直接归于本体末段。
保护区/Top-K结构通过、非有限值0；Native输出max_abs0.03125、RMSE0.003320184、Top-K替换366，零容差仍FAIL。

[补丁](candidate.patch)、[运行命令](run.sh)、[统计](summarize.py)、[数据](report.json)。
[原始泳道](../../csa_split_optimization_20260927/round04_merge_early/h8192_b16/swimlane/dfx/merged_swimlane.json)。
[原始计时](../../csa_split_optimization_20260927/round04_merge_early/h8192_b16/timing/report.json)。
