# 十轮调度：第1轮，取消O_B预派发

基底2dd51f15，8K/B40/S6/TP1、mode2/atomic1/deterministic0、EPLB关，A3 device0。
正式第4层权重+合成输入历史，第二CSA层metadata复用；5预热/20次无profiler计时，另4个DFX窗口。
任务task_20260927_181859_29094976610退出0。

只关闭quant生产者allow_early_resolve，O_B需等量化完成后再派发，避免提前占AIC影响O_A后续波次。
最新上游2164563与原接入均开启此标志；本项是针对当前64个O_A块/多波执行的调度假设，算术和块数不变。

| 指标 μs | 基底 | 候选 |
| --- | ---: | ---: |
| CSA本体均值 | 1318.049 | 1311.962 |
| 本体p50 | 1319.180 | 1309.100 |
| 本体p95 | 1364.580 | 1359.340 |
| 同轮Native均值 | 1419.105 | 1412.800 |
| 完整PTO均值 | 1674.572 | 1664.282 |
| O_A Worker窗口 | 128.86–130.68 | 125.12–126.10 |
| O_B Worker窗口 | 91.10–98.06 | 93.32–101.62 |
| merge结束→HC_post结束 | 275.96–285.62 | 278.92–286.42 |

本体约−0.46%、Native约−0.44%；O_A局部略短但O_B略长，尾段没有获益，已撤回。
不把同轮整体计时漂移冒称调度提升，不扩测七档。保护区/Top-K结构/非有限值检查通过，Native零容差仍FAIL。
本项计入新十轮的第1轮；前序O_A行列并行等不重复计数。

[补丁](candidate.patch)、[设备命令](run.sh)、[统计脚本](summarize.py)、[完整数据](report.json)。
[原始泳道](../../csa_split_optimization_20260927/round01_quant_no_early/h8192_b40/swimlane/dfx/merged_swimlane.json)。
[原始计时](../../csa_split_optimization_20260927/round01_quant_no_early/h8192_b40/timing/report.json)。
