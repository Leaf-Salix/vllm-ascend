# 关闭Indexer Q反量化生产者的预派发：本体未改善，撤回

基底为Q_A先行版6cfc737d；上一项Score开关已恢复。本项只关闭idx_qr_dequant_rope的allow_early_resolve，
使其Query Hadamard消费者等待Vector结果完成后再占AIC。核内、任务数量和数据依赖不改。
动机：原B40四窗口Query Hadamard平均local_setup9.50–44.10 μs，部分等待期间Q_B仍在执行。
这是潜在资源竞争，不预先断言为本体慢的唯一成因。

任务task_20260927_174025_266828031875退出0，B40/H8K、S6/TP1/mode2/atomic1/deterministic0，无EPLB，
正式layer4权重加合成输入/历史；5次预热、20次无profiler计时，另4个DFX窗口。

| 指标 μs | Q_A先行基底 | Query Hadamard不预派发 |
| --- | ---: | ---: |
| CSA本体均值 | 1327.86 | 1333.59 |
| CSA本体p50 / p95 | 1322.03 / 1367.88 | 1327.00 / 1372.64 |
| 完整PTO均值 | 1674.93 | 1662.00 |
| Query Hadamard平均local_setup四窗口范围 | 9.50–44.10 | 0.53–0.56 |
| Query Hadamard启动分散范围 | 0.50–43.38 | 5.42–36.12 |
| Q_B启动分散范围 | 71.52–93.58 | 77.84–98.68 |
| Q_B最后完成位置范围 | 422.00–433.08 | 398.14–449.06 |

提前占位消除，但Q_B仍分散；无profiler本体没有获益（+0.43%），p50/p95也未改善，撤回。
完整路径下降12.93 μs与本体方向不同；split/body/writeback/full是独立计时，不能将完整路径变化归给本体调度。
仅此代表档，不扩测；两项关闭预派发的试验分别失败，不说明所有预派发都应保持，也不支持全关。
后续继续围绕真实关键链与资源利用，不以泳道外观更整齐作为性能验收。
保护区、Top-K结构通过，输出/状态非有限值0；Native输出max_abs0.03125，零容差仍FAIL；未做新的token/DSpark验收。

[完整计时/数值与原始泳道](report.json)、[单变量补丁](candidate.patch)、[复现](run.sh)。
