# Sparse Attention：16行UB双缓冲先导，未保留

2026-09-27，基于 `21d99f8a`（V10＋性能版B40 KV投影优化）。
只修改性能版 compressed KV 搬运：每个AIV原来收齐64行再写GM，
改为每16行一块，通过 `pl.pipeline(stage=2)` 使用两个UB缓冲。
参考Native `SASVectorBlock::ProcessVec0L` 的16行分批流水。
源码没有实现Native动态行距的成对DMA，不能称为完整复制Native搬运。
候选顺序、softmax、BF16量化和PV算术保持原实现，跨任务调度未改。

CPU ABI/PTOAS/AICPU编译通过，生成C++确认两个16×512 BF16缓冲地址不同。
单卡任务 `task_20260927_153712_16465687755` 退出0；固定layer4权重、8K/B40、S6/TP1、
mode2/atomic1/确定性0/无EPLB，5次预热、20次无profiler计时及4个独立DFX窗口。

| 指标 | 保留基底 | 16行候选 |
| --- | ---: | ---: |
| CSA本体均值 μs | 1386.00 | 1386.11 |
| 本体p50 / p95 μs | 1378.45 / 1431.02 | 1385.99 / 1412.58 |
| qk_pv AIC四窗口block均值范围 μs | 319.81–334.24 | 322.44–329.31 |
| 同轮Native均值 μs | 1410.61 | 1417.81 |
| 完整PTO均值 μs | 1736.41 | 1724.72 |

核内范围重叠、本体均值未改善；单次p95或完整路径变化不足以证明该核内改动获益。
候选已撤回，没有扩测七档。不能将“搬运改成双缓冲”写成已验证性能优化。

保护区失败0、Top-K结构错误0、输出非有限值0；max_abs=0.03125、RMSE=0.003291306，
Top-K集合替换900。浮点零容差FAIL，未做整模型token/DSpark验收。

- [候选补丁](candidate.patch)
- [统计及四窗口原始路径](report.json)
- [基底独立结果](../kv240_splitk_only/README.md)

原始目录：`../../csa_split_optimization_20260927/sparse_gather16_pipeline/h8192_b40/`。
下一步用固定Native输入的PMU诊断搬运、计算及等待，避免继续只按逻辑字节量选候选。
