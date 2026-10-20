# CANN聚合PMU独立复核

2026-10-20。仅读取 `pmu-8k-b16-v1-evidence.tar.gz`、当前PMU助手及分析JSON，CPU重算CSV；未修改SDK或提交任务。

## 结论

外层task级PMU记录有效，`PMU_CANN_ANALYSIS.json` 中各operator的记录数及全部中位数与原 `op_summary` CSV一致。
候选确有3条 `aicore_kernel_mode_0`，均为MIX_AIC、24 AIC／48 AIV、device0、stream74。
该结果仅授予整个Simpler executor的聚合管线计数，不能归因到内部QK/PV、Norm、O-proj等函数。

归档SHA256：`966b8c7854c975ea178a7db9142337ae068984d85747f6ead47b82f2fd557e1d`，17个文本载荷。
两个arm导出的CSV存在且非空；实际目标PMU列非空、有数值，不是只凭成功退出或导出了普通CSV。

## 数值门禁与协议

- Native/PTO A、B八边界的具体比较记录均逐bit通过，A/B/A、compiled/raw、TopK毒化恢复及guards通过。
- 六源、编译收据、保存的Native输入／初始cache／八边界SHA与先前SK0基线完全相同；实际PMU助手文件SHA也与结果身份一致。
- 实际为Level1＋PipeUtilization，每arm3次回放；没有DFX／dep，没有正式性能计时，`timing_executed=false`。
- 实际助手每arm profile后再次检查全部边界和guards，并在结束时检查只读packet。后两者以源码断言和完整成功结束覆盖，没有额外的逐项post-PMU packet JSON字段，不应虚构该字段。
- 本次没有重新运行跨bucket或整模型8K／128K；不得把此前冻结源码通过的范围重标为本次新验收。

## 三条executor记录重算

| 回放 | Task Duration μs | aicore_time μs | AIV scalar | AIV vec | AIC MTE2 | AIV MTE2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1311.580 | 1306.257 | 0.509 | 0.117 | 0.276 | 0.085 |
| 2 | 1268.100 | 1262.209 | 0.501 | 0.121 | 0.283 | 0.080 |
| 3 | 1262.020 | 1256.716 | 0.497 | 0.121 | 0.281 | 0.082 |
| 中位 | 1268.100 | 1262.209 | **50.1%** | **12.1%** | **28.1%** | **8.2%** |

另AIC scalar中位37.1%。原CSV `Task Duration(us)` 对应汇总JSON的 `Duration(us)`，已按同一列语义核对。
三条Task ID相同是同一捕获节点的三次回放，不是重复读取一条记录；CSV三条duration／指标确实不同。

## 解释限制

1. Scalar、Vector、MTE2等指标可以重叠，各角色和多核聚合的分母也不是一个可分割的串行时间轴。
   **不能把50.1%、12.1%、28.1%和8.2%视为互斥耗时，更不能乘executor时长后称为可独立消除的成本。**
2. 这些是外层executor聚合值，不证明某个内部函数是Scalar瓶颈或某段KV读取是MTE2瓶颈；搬运busy比例也不是搬运字节数或stall比例。
3. host-flow关联失败限制了host调用／内部算子归属；它没有使原 `op_summary` 的device task计数凭空消失。
   可描述这3条外层task的计数，但不能据此建立内部func PMU归因、关键路径或优化收益上界。
4. CSV外层 `Task Wait Time=0` 不表示核内同步、MTE等待或标量轮询为0。CANN任务等待与Simpler内部等待不是同一观测层次。
5. 与Native某个单独attention／matmul的比例直接相比会混淆范围：候选记录覆盖多个内部函数，Native记录是分散的外层算子。
6. 这是受profiler影响的诊断任务，不替代无DFX 12×100基准，不能用1268.1 μs宣称本轮比先前1333.7 μs优化了。

## SDK能力边界

已只读核对保存的Simpler `host/kernel_static_config.h:40`：kernel-mode路径对 `enable_pmu!=0` 明确返回unsupported。
该限制与本次CANN外层profiling并不矛盾。当前不应绕过validate、修改SDK或声称已经取得内部per-task PMU。

本轮计数可以作为后续分离测量的线索；内部归因仍须有效且支持的独立观测协议，本报告不授予未经测量的优化结论。
