# 当前性能版单次 PIPE_UTILIZATION 诊断

2026-09-26，`task_20260926_201418_26222942858`，physical device 0，completed/exit=0。
生产算术与任务图仍为保留版；B16/S6/H8192、正式第 2 层权重及合成状态、mode=2、atomic=1、
metadata 复用。PyPTO 88297437、Simpler a54c05095、PTOAS 0.66、PTO-ISA 327cd586、CANN 9。

- `pmu.csv`：本次原始 1131 条记录，425 AIC + 706 AIV，52 个 func_id。
- `kernel_config_source.py`：实际编译生成的名称表，不能拿旧泳道的 func_id 直接解释本次计数。
- `summary.json`：各任务原始计数求和；比例为 `100 * sum(busy_cycles) / sum(total_cycles)`。
- `report.json`：原始回放报告。有限值检查通过，没有跨实现参考，状态为 MEASURED。
- `failed_attempts.json`：此前输入设备、运行配置与重复 PMU 注册失败，未用其计数作性能结论。

O-A MTE2 busy/total 为 84.4%，Cube 为 31.2%。通道可以重叠，busy 不等于实际带宽利用率。
这份数据用于寻找核内方向：没有上游同口径 PMU，不能由此解释全部上游差距。
program 模式将原始 CPU 快照复制到设备，本次没有 NZ 转换；PMU 强制 single-issue 派发，
因此它与默认 kernel 图重放的运行条件不同，不替代原有泳道、Event 或整模型性能结果。

需要再次诊断时从仓库根通过队列运行，避免裸跑占卡：

```bash
task-submit --device 0 --max-time 180 'bash tests/pypto_test/results/csa_baseline_20260926/upstream_gap/pmu_pipe/run.sh'
```

用返回的任务号查询 `task-submit --status <id>`。脚本复用现有 schema=2 输入，
临时编译与输出位于 `results/csa_baseline_20260926/pmu_pipe_current/`，不覆盖本归档。
PMU 只执行一次；如需其他事件组另起进程，勿循环注册采集器。原始报告里的绝对路径
保留当时运行位置，轻量原始 CSV 已复制到本目录，重复编译与运行目录可清理。
