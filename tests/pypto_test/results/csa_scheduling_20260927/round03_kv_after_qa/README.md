# 十轮调度：第3轮，KV投影等待Q_A

基底2dd51f15；在qkv入口为KV投影增加`late_dep+qa_tid`的dummy依赖。保留原KV数据依赖、seed、块数与算术，精度版不动。
上游2164563让Q与KV并行；当前KV有32个AIC块（历史上游8个），本项试图减少Q_A前段竞争，延后并非改成上游算术。
CPU完整根/PTOAS/AICPU通过；task_20260927_182734_30115537818退出0。
2026-09-27，A3 device0、8K/B16/S6/TP1、正式第4层权重+合成输入历史、第二CSA层metadata复用，mode2/atomic1/deterministic0、EPLB关；5预热/20计时，另4个DFX窗口。

| 指标 μs | 基底 | 候选 |
| --- | ---: | ---: |
| 本体均值 | 793.658 | 801.092 |
| 本体p50 | 790.840 | 803.580 |
| 本体p95 | 813.520 | 821.380 |
| 同轮Native均值 | 923.928 | 926.161 |
| 完整PTO均值 | 1106.036 | 1111.254 |
| Q_A末尾结束 | 140.26–147.64 | 122.02–134.08 |
| KV末尾结束 | 111.24–160.38 | 159.88–234.04 |
| Top-K merge末尾结束 | 405.28–426.70 | 406.64–448.54 |

Q_A确实提前，KV延后挤入后续分支，Top-K未提前，最终本体约+0.94%，撤回。
这说明“Q_A更早”不足以决定完整调度更好；后续必须同时跟踪后段资源集中与分支完成。
保护区/Top-K结构通过、非有限值0；Native输出max_abs0.03125、RMSE0.003320106、Top-K替换366，零容差仍FAIL。

[补丁](candidate.patch)、[设备命令](run.sh)、[统计脚本](summarize.py)、[数据](report.json)。
[原始泳道](../../csa_split_optimization_20260927/round03_kv_after_qa/h8192_b16/swimlane/dfx/merged_swimlane.json)。
[原始计时](../../csa_split_optimization_20260927/round03_kv_after_qa/h8192_b16/timing/report.json)。
