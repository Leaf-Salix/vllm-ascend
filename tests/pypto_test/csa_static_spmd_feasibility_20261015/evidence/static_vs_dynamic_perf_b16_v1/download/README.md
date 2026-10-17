# CSA 128K/B16/S6：原动态 PTO 与单 SPMD

同卡正式权重测试，仅比较 PTO，task_20261016_183457_19106108116 已完成。
两侧输出、Top-K、六份 cache/state 精确相同，保护区和30代际进度检查通过。
每步恢复初态，这不是连续decode轨迹。

| 口径（μs，中位数） | 动态 PTO | 单 SPMD | 耗时变化 |
| --- | ---: | ---: | ---: |
| L1 incore 首尾（各3次） | 897.70 | 1380.22 | +53.75% |
| 图外 Event（各20次） | 942.62 | 1467.75 | +55.71% |

六份 JSON 是 Perfetto/Chrome tracing 泳道；只看 incore 起止，不把接收任务时间当成计算开始。
summary.json 保留全部样本和原始路径。

- [CSA_128K_B16_S6_dynamic_pto_L1_step1.json](CSA_128K_B16_S6_dynamic_pto_L1_step1.json)
- [CSA_128K_B16_S6_dynamic_pto_L1_step2.json](CSA_128K_B16_S6_dynamic_pto_L1_step2.json)
- [CSA_128K_B16_S6_dynamic_pto_L1_step3.json](CSA_128K_B16_S6_dynamic_pto_L1_step3.json)
- [CSA_128K_B16_S6_single_spmd_L1_step1.json](CSA_128K_B16_S6_single_spmd_L1_step1.json)
- [CSA_128K_B16_S6_single_spmd_L1_step2.json](CSA_128K_B16_S6_single_spmd_L1_step2.json)
- [CSA_128K_B16_S6_single_spmd_L1_step3.json](CSA_128K_B16_S6_single_spmd_L1_step3.json)

[完整报告](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012/tests/pypto_test/results/csa_static_spmd_feasibility_20261015/static_vs_dynamic_perf_b16_v1/REPORT.md)
