# 选择性关闭Score生产者的预派发：本体未改善，撤回

用户要求检查提前占位使时序更差的任务。以Q_A先行版6cfc737d为基底，
只将 `indexer_score_topk_native_pair` 的allow_early_resolve设为False，阻止其Top-K merge消费者提前占AIV。
Score/merge核内代码、任务数量、其他依赖均不改；并非将merge本身的生产者标志设False。
任务task_20260927_173803_265073027486退出0，单卡B40/H8K，其余配置和上一项完全一致。
5次预热、20次无profiler计时，另4个DFX窗口，不扩大测试矩阵。

| 指标 μs | Q_A先行基底 | 关闭Score预派发 |
| --- | ---: | ---: |
| CSA本体均值 | 1327.86 | 1330.48 |
| CSA本体p50 / p95 | 1322.03 / 1367.88 | 1329.37 / 1366.10 |
| 完整PTO均值 | 1674.93 | 1677.42 |
| Top-K merge平均local_setup四窗口范围 | 8.16–53.01 | 0.68–0.71 |
| Top-K merge最后完成位置范围 | 617.86–706.46 | 621.70–653.70 |
| Score AIC启动分散范围 | 6.32–64.04 | 0.44–14.58 |

预占等待确实消失，四个profile窗口的末尾范围也更窄，但无profiler本体没有改善（+0.20%，近似持平）。
不把profile窗口更整齐等同稳态加速；本项撤回，不扩测。
local_setup包含准备/等待，大数本身不证明该预派发有害；O_A后quant的长等待同理。
保护区、Top-K结构通过，输出/状态非有限值0；Native输出max_abs0.03125，零容差仍FAIL。
[完整计时/数值/原始泳道](report.json)、[补丁](candidate.patch)、[复现](run.sh)。
下一项独立检查Indexer Q反量化生产者标志，阻止Query Hadamard抢占AIC；不叠加本项。
