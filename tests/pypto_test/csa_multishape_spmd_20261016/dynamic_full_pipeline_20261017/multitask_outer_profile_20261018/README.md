# 原分task CSA包含AICPU调度的外部耗时

用户要求七档表增加外部可见完整跨度。原L1只包含incore首尾；其中第六列是
receive_to_start增量，不是AICPU完成时间。缺少scheduler/lifecycle记录时，
不能把它当成完整AICPU边界，更不能把各核任务时长直接相加。

本列使用PyTorch profiler的全部设备流：首个CSA设备kernel开始到最后一个结束，
包含PTO的AICPU调度与启动/收尾，只排除图外fixture恢复和输出毒化。
Native现有列也是外部设备跨度，但不同日期/版本仍不能当作同轮加速比。

六档已有10月12日baseline原分task外部profiling，保留其日期及a00afa9f版本；
不把这些旧外层值减去当前L1来宣称纯调度开销。B28缺少外部记录，本目录只补
这一档，使用当前正式分task源码和指定独立PyPTO/Simpler，不跑Native或静态SPMD。

沿用10月12日已校准的整层CSA服务用例和10月18日Native B28的全流区间统计。
`csa_compiled_case.py`只开放PTO，检查队列设备及依赖来源；使用真实第4层权重、
Native metadata和同一缓存初态，捕获npugraph_ex图，再做图重放、完整state及保护区
检查。5次预热、20次无profiler Event另存，30次独立外部profiling用于新增列。
Event不混作profiler跨度，不把这项测量当成动态单SPMD30%目标验收。

```bash
task-submit --device auto --max-time 1200 \
  'bash /绝对路径/multitask_outer_profile_20261018/run.sh /全新输出目录'
```

`PROVENANCE.json`记录复用脚本来源；输出保存源码快照、实际依赖、原始CSV/trace、
完整report及30次样本摘要。任务状态以队列和结果为准，未完成前不预记性能。

## 本次结果

`task_20261018_122158_10688028513`，队列分配device8，exit0。
`h131072_b28_v1/summary.json`：30次外部跨度中位1357.75μs，范围
1312.50～1438.75μs；20次无profiler Event中位1366.26μs，单独保存。
每次恰好包含一个simpler_aicpu_kernel_exec及一个aicore_kernel_mode_0，
完整跨度同时覆盖两者，未将重叠时间相加。

固定初态的后端图直接重放，与同一图经compiled callable调用的输出、Top-K、
六份cache/state逐bit相同，metadata/未写区保护检查通过。该验证是原分task
固定地址图的内部一致性，不代表新做了Native精度或动态单SPMD边界验收。

本次原分task源码与目标目录一致；服务入口确认3次PTO派发，没有Native fallback。
最新独立PyPTO/Simpler位置、CANN和源码快照在environment.json/report.json。
该轮没有与L1或Native同时采样，不能将1357.75减去历史1312.22当作纯AICPU耗时。
完整七档表及逐项版本说明见上级LATEST_SEVEN_CASE_PERFORMANCE_20261018.md。
