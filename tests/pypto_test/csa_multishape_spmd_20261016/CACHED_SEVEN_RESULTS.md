# CSA 多档单 SPMD：DCCI 轮询退避与分核实测

2026-10-16，正式模型第4层权重、合成历史，S=6。只比较原动态 PTO 与单 SPMD，未运行 Native forward。
计时为完整 CSA 的 L1 incore 最早开始到最晚结束，每侧3窗取中位数；不计入外部调度头尾。
七档各自编译，采用各自同规格既有动态 L1 的成本和调度策略；均只在信号未就绪时退避100 tick（50MHz下2μs）。

|历史|B|原动态 L1 μs|单 SPMD L1 μs|耗时变化|单 SPMD 三窗 μs|策略|
|---:|---:|---:|---:|---:|---|---|
|131072|4|582.82|541.92|-7.02%|541.92 / 541.98 / 537.00|earliest_ready|
|131072|8|684.32|647.48|-5.38%|647.48 / 649.44 / 644.14|earliest_ready|
|131072|16|892.78|832.28|-6.78%|828.36 / 832.28 / 834.48|earliest_ready|
|131072|24|1169.46|1121.12|-4.13%|1121.12 / 1114.96 / 1124.70|critical_path|
|8192|16|680.02|621.96|-8.54%|613.20 / 622.04 / 621.96|critical_path|
|8192|24|861.62|836.74|-2.89%|838.36 / 834.08 / 836.74|earliest_ready|
|8192|32|972.72|905.72|-6.89%|898.08 / 905.72 / 916.86|critical_path|

七档全部通过 eager、aclgraph 及最终输出/Top-K/六份状态的 exact 对照，Native 和工作区保护区通过，30代信号正确。
每次调用恢复同一 fixture；不把这组结果扩称为连续 decode 轨迹或整网服务验证。
128K/B16 的832.28 μs尚未达到700 μs目标。生产入口没有切换。

图重放 Event 另行报告，它包含的范围与 L1 不同，部分小档仍慢于原动态，不能将 L1 收益说成端到端全面收益。

|历史|B|原动态 Event μs|单 SPMD Event μs|
|---:|---:|---:|---:|
|131072|4|637.01|648.46|
|131072|8|728.59|746.33|
|131072|16|959.50|931.23|
|131072|24|1229.54|1230.75|
|8192|16|717.97|724.35|
|8192|24|909.47|941.56|
|8192|32|1043.86|1025.07|

## 复现与证据

各项任务号、三窗样本和相对路径在 [cached_seven_results.json](cached_seven_results.json)，CSV 同名。
每个结果目录保存 result.json、原动态/单 SPMD各3窗L1、实际入参绑定、分核表和生成的PyPTO源码。
业务长度、位置、slot和页表仍在设备读取；初始化只根据已知shape/stride和Host长度策略选择模板。
不做额外hash检查；大编译产物留独立worktree，路径清单见LOCAL_ARTIFACTS.json。

执行方式：

```bash
task-submit --device auto --max-time 1800 'bash <本目录>/run_device_compare.sh <B> <history> <新结果目录> --balanced-schedule --cached-wait --wait-backoff-ticks 100 --schedule-policy <本档策略> --schedule-cost-profile <本档成本JSON>'
```

128K/B16本次使用 b16_dynamic_l1_cost.json，其余六档使用 per_shape_costs/h<history>_b<B>.json。
读侧每次读前DCCI，写侧DCCI+DSB和业务payload维护均保留；不是依赖Scalar缓存一致性。
等待helper的本地生成边界适配位于signal_polling.py，后续主线notify/wait完备后再收敛替换。
