# 2026-10-11：B16/S6 旧版 L1 首尾 52 μs 核实

所有有效性能数字均为真实单卡、L1 的 incore 起点到终点；没有使用模拟器数据。
计算口径：按 launch_epoch 分组，从原始 aicore_tasks 计算
`(max(end_cycles) - min(start_cycles)) / 50`，单位 μs。
不计 Worker receive→start、本次 incore 之前/之后的 Simpler 调度及 Orchestrator 头开销。

| 条件 | 三次 incore 首尾 μs | 中位数 μs |
| --- | --- | --- |
| 历史：同时开启 PTO PyTorch profiling | 33.12 / 34.00 / 33.88 | 33.88 |
| 今天：未开启 PTO PyTorch profiling | 55.06 / 52.02 / 35.24 | 52.02 |
| 今天第二轮：同上 | 54.62 / 52.18 / 33.98 | 52.18 |
| 今天恢复历史 profiling 开关 | 29.92 / 32.52 / 34.22 | 32.52 |
| 今天不开 PTO profiler，Native profiler 完成后补 5 次 PTO 预热 | 35.06 / 34.42 / 37.14 | 35.06 |

结论：之前把不同采样条件下、前两次未稳定的 52 μs 作为基线不合适。
旧版并没有经本轮证据确认发生从约 34 μs 到约 52 μs 的稳定性能退化。
原有预热在 Native profiler 之前；其退出和同步解析耗时数秒后，直接开始 PTO L1 三次采样。
恢复历史 profiling 开关或在 Native profiler 结束后重新预热，都消除了约 52 μs 的现象。
这里验证的是采样流程与预热状态的影响；没有设备频率或缓存计数器证据，不能进一步断言是锁频、升频或某级缓存造成。

## 证据

- `timing_audit.json`：从 L1 原始时钟重算，逐项与导出泳道一致。每个有效 step 有 24 AIC + 48 AIV 共 72 条记录。
- `baseline_b16_profile_audit/`：任务 `task_20261011_143516_3426621586`；恢复 `--profile`，完整数值、graph/eager、同址更新和保护区检查 PASS。
- `baseline_b16_warmup_audit/`：任务 `task_20261011_143713_4370976851`；DFX 窗口前追加 5 次预热，原有完整检查 PASS。
- `baseline_b16_warmup_audit/warmup_audit.json` 的 `before_begin_warmup` 是有效的三个 step。
- 该任务附带尝试窗口内 5 次预热 + 3 次测量；后四次记录数不足 72，已明确标为无效，不参与上表和性能结论。原始 8 次保留用于排查采集完整性。
- `bench_before_warmup_fix.py`、`audit_warmup.py`、`run_warmup_audit.sh` 固定了诊断逻辑。
- 历史和今天的 `hc_medium96.cpp`、AIC/AIV 包装代码：去掉注释和自动生成的 inline 编号后逐字一致。没有做 hash 校验。

## 后续采样约束

正式脚本已在 Native profiler 完成后、DFX 窗口开始前补足 `args.warmup` 次 PTO 重放，保持窗口只采三个 step。
报告显式记录 `pto_pytorch_profile` 和 `pto_warmup_after_native_profile`。
新旧优化版本必须使用一致的 profiling 开关和预热流程，不能用今天的 52 μs 与历史 34 μs 跨条件计算收益，也不能只选三次中最快的一次。
本轮未修改 HC_pre 算子执行逻辑或依赖仓库。
