# 十轮调度第9轮：重Compressor等待Q_B，撤回

基底07365e52，只暴露Q_B TaskId并把Attention Compressor投影调度依赖从Q_A改为Q_B，轻Indexer保持提前。
数学/任务数不变；CPU根/PTOAS/AICPU通过。最新pypto-lib2164563的TP多卡路径有Q_B→Attention投影顺序，
TP1路径没有这一串行依赖。本轮借鉴其避免Cube竞争的思路，不能将不同TP的模式直接当成TP1最优。
历史727.98 μs图仅作调度参考。

A3 device0、8K/B16/S6/TP1、正式第4层权重+合成输入历史、第二CSA层metadata复用，mode2/atomic1/deterministic0，EPLB关。
5预热/20无profiler采样、4窗口DFX，task_20260927_185624_333050329390退出0。

| μs | 基底 | 候选 |
| --- | ---: | ---: |
| 本体均值 | 781.524 | 800.310 |
| p50 | 781.050 | 799.740 |
| p95 | 791.800 | 814.200 |
| Native均值 | 922.024 | 932.164 |
| 完整PTO均值 | 1084.133 | 1107.031 |
| Q_B启动分散 | 40.00–50.74 | 14.56–17.12 |
| Q_B末尾 | 277.22–303.92 | 249.90–274.08 |
| Attention投影末尾 | 196.48–224.62 | 311.52–334.14 |
| Top-K末尾 | 404.04–430.54 | 420.70–433.34 |

Q_B提前完成，但重Compressor和后续key链变晚，本体+2.40%，比同轮Native+1.10%更差。撤回，不扩测否决方案。
这属于调度次序试验，不是发现了应按incore保留规则保存的核内数学优化。
保护区/Top-K结构/有限值通过，Native零容差FAIL；max_abs0.03125、RMSE0.0033201739、Top-K替换366。

[补丁](candidate.patch)、[CPU编译](compile_candidate.py)、[命令](run.sh)、[统计](summarize.py)、[数字与泳道](report.json)。
