# O projection 登记顺序先导：未保留

基底 c7a52af5，2026-09-27，A3 device 0，8K/B40/S6/TP1；正式第4层权重、合成输入历史，mode2/atomic1/deterministic0，EPLB关闭。5次预热、20次无profiler计时，另采4个DFX窗口。

仅把各组 `O_A → quant → O_B` 的登记顺序改成全部 O_A 先登记，再登记各组 quant/O_B；消费者仍只依赖本组 O_A，没有全组同步。核内代码、任务数量和算术不变。完整CPU根与AICPU编译通过，任务 `task_20260927_174648_27004935033` 退出0。

| 指标 μs | 基底 | 候选 |
| --- | ---: | ---: |
| CSA本体均值 | 1327.859 | 1324.538 |
| 本体p50 | 1322.030 | 1323.990 |
| 本体p95 | 1367.880 | 1361.060 |
| Native均值 | 1427.994 | 1419.700 |
| 拆分+本体+写回完整均值 | 1674.927 | 1678.539 |
| O_A启动分散（四窗口） | 99.92–109.58 | 92.40–101.64 |
| quant平均local_setup（四窗口） | 79.85–82.50 | 80.49–83.36 |

本体均值仅−0.25%，p50略升，量化等待未缩短；不能认定有效调度收益，撤回，不扩测七档。保护区与Top-K结构通过、非有限值0；Native输出max_abs=0.03125、RMSE=0.003292694，Top-K替换901，Native零容差仍FAIL。未做整模型验收。

- [小结数据](report.json)、[统计脚本](summarize.py)、[候选补丁](candidate.patch)、[隔离编译脚本](compile_candidate.py)、[设备命令](run.sh)。
- [原始timing](../../csa_split_optimization_20260927/o_proj_issue_order/h8192_b40/timing/report.json)
- [原始泳道](../../csa_split_optimization_20260927/o_proj_issue_order/h8192_b40/swimlane/dfx/merged_swimlane.json)

后续按用户要求以历史727.98 μs上游Worker泳道为调度参考；相同任务登记顺序本来就是上游写法，本项失败不否定其行块×列块的任务粒度。
