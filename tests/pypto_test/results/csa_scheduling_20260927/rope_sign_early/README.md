# RoPE符号准备开放预派发：已撤回

基底2dd51f15；仅csa_rope_sign加allow_early_resolve，Index Q反量化可以预派发。上游2164563 TP1同样未开启该标志；属于接入主动调度试验。
2026-09-27，A3 device0、8K/B16/S6/TP1，正式第4层权重+合成输入历史，第二CSA层metadata复用，mode2/atomic1/deterministic0、EPLB关。
5预热/20次无profiler计时，另4个DFX窗口；task_20260927_181518_28700138050退出0。

本体均值793.658→793.802 μs，p50 790.840→793.620，p95 813.520→817.740；未获收益，已撤回。
Index Q反量化setup由0.54–0.98→3.64–28.97 μs，实际发生更多准备/门控等待，但Top-K结束405.28–426.70→401.58–441.32 μs并未稳定提前。
首Worker→norm结束92.36/90.18/95.76/86.48→88.88/86.34/81.94/82.46 μs发生变化，不能把上游无依赖任务的差值强归到RoPE标志。
保护区/Top-K结构通过、非有限值0；Native输出max_abs0.03125、RMSE0.003320166，Top-K替换366，零容差仍FAIL。

[脚本](run.sh)、[补丁](candidate.patch)、[统计脚本](summarize.py)、[数据](report.json)。
[原始泳道](../../csa_split_optimization_20260927/rope_sign_early/h8192_b16/swimlane/dfx/merged_swimlane.json)。

本项在用户“再做十轮调度”之前完成，不计入新的10轮。新的轮次从round01_quant_no_early开始。
