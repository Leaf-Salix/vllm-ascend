# HC转换预派发标志：无明确收益，已撤回

基底2dd51f15，含已保留O_A二维网格；2026-09-27，A3 device0，8K/B40/S6/TP1。
正式第4层权重与合成输入历史、第二CSA层metadata复用，mode2/atomic1/deterministic0、EPLB关闭。
5次预热/20次无profiler计时，另4个DFX窗口；任务task_20260927_180839_28136162717退出0。

只为性能版hc_widen增加allow_early_resolve，允许其HC消费者预派发；转换、任务数量、输入布局和算术均不变。
上游最新版2164563直接接受FP32残差，没有hc_widen；这是接入侧额外任务的调度试验，不能称照搬上游标志。
历史727.98 μs图的B16前段66.62 μs，当前B16约87–91 μs，提供了前段排查方向；
本次在有同源码基底的B40试验，不能拿它的绝对耗时与历史B16相减。

| 指标 μs | 基底 | 候选 |
| --- | ---: | ---: |
| CSA本体均值 | 1318.049 | 1314.144 |
| 本体p50 | 1319.180 | 1311.690 |
| 本体p95 | 1364.580 | 1359.200 |
| 同轮Native均值 | 1419.105 | 1412.640 |
| 完整PTO均值 | 1674.572 | 1661.319 |
| 首Worker→norm结束（四窗口） | 120.04–126.94 | 122.78–136.14 |
| hc_pre_linear首次kernel开始（首Worker归零） | 25.76–39.16 | 25.76–47.34 |

本体均值仅−0.30%，同轮Native约−0.46%；前段窗口未缩短，无法证明标志带来收益。
撤回，不扩测小档、七档或整模型。保护区/Top-K结构通过、非有限值0；
Native输出max_abs0.03125、RMSE0.003292630、Top-K替换901，零容差仍FAIL。
前段额外转换仍存在，但没有据此追加核内融合候选；按用户限定继续调度阶段。

- [候选补丁](candidate.patch)、[运行脚本](run.sh)、[统计脚本](summarize.py)、[数据](report.json)。
- [原始计时](../../csa_split_optimization_20260927/hc_widen_early/h8192_b40/timing/report.json)。
- [原始泳道](../../csa_split_optimization_20260927/hc_widen_early/h8192_b40/swimlane/dfx/merged_swimlane.json)。
