# HC_pre 第二阶段48个AIV的性能候选（2026-10-11）

本目录全部性能来自真实设备 L1：完整 HC_pre 的最早 incore 开始至最晚 incore 结束；每场景固定三个 step。没有运行 Native、PyTorch profiler 或新增模拟器测试。

B16/S6 是观察基准，目标仍为完整 HC_pre 小于28 μs；优化方案必须泛化。代表规格为 B=2/8/16/28、S=6，分别覆盖当前四个编译分档。有效方案最终仍需验收原来的十四个场景。当前没有候选证明已达到目标，正式入口仍为上一轮已验收的48核分工版本。

每个任务由 task-submit --device auto 分配设备；自动分配用过物理卡1和2。不同卡、不同轮次的结果不直接当作优化百分比；应优先看同一任务中的基线与候选。所有三次采样、设备号、精度门禁和路径保留在 report.json 及汇总 CSV 中。

| 实测快照 | 卡 | T12 | T48 | T96 | T168 |
| --- | ---: | ---: | ---: | ---: | ---: |
| run_aligned_sink_blocks_representative | 1 | 30.72 | 30.22 | 35.72 | 41.48 |
| run_baseline48_control_representative | 2 | 33.7 | 31.86 | 35.36 | 43.08 |
| run_compact_materialize_representative | 1 | — | — | — | — |
| run_compact_tree_recip_representative | 1 | 33.22 | 32.66 | 37.88 | 40.16 |
| run_compact_tree_representative | 1 | 32.66 | 31.48 | 34.6 | 39.62 |
| run_constant_full_d_representative | 1 | 30.88 | 35.02 | 39.12 | 43.12 |
| run_mixed_compact64_d1024_representative | 0 | 34.46 | 29.32 | 38.18 | 41.66 |
| run_mixed_fused_compact_representative | 2 | 30.66 | 35.84 | 35.74 | 45.72 |
| run_mixed_fused_load_representative | 2 | 30.98 | 31.16 | 37.94 | 43.24 |
| run_mixed_repeat64_representative | 1 | 32.98 | 33.82 | 36.38 | 41.9 |
| run_reduce_all_then_pad_representative | 0 | 33.82 | 35.42 | 35.24 | 40.44 |
| run_reduce_pad_copy_representative | 1 | 34.74 | 29.88 | 35.5 | 40.02 |
| run_role_balance_post_sink_representative | 1 | 32.16 | 32.12 | 37.48 | 38.36 |
| run_scalar_indices_representative | 1 | 33.9 | 31.16 | 37.82 | 45.2 |
| run_sink_block_gather_representative | 1 | 32.1 | 31.0 | 36.24 | 40.36 |
| run_static_compact_tree_representative | 1 | 33.04 | 29.88 | 36.4 | 43.62 |
| run_static_full_tail_representative | 1 | 37.54 | 32.9 | 38.16 | 42.78 |
| run_tree_reduce_representative | 0 | 32.7 | 31.0 | 34.7 | 42.88 |

## 本轮排除及限制

- 缩小部分和 tile 后用 assemble 扩展的候选未过数值门禁，不采用。改为 fillpad_expand 后的一版通过代表规格，但没有稳定改善 T96。
- 动态有效列的树形 col_sum 被当前 A2/A3 编译校验拒绝；保持静态输入的树形版本已验证。没有修改依赖仓库。
- fillpad 的 PadZero 元信息在后续 transpose 遇到类型不匹配；实验快照采用单行 col_sum 拷贝恢复普通 tile。未把这个候选强行落入生产。
- 直接增大 D 到1024/2048且保留8行双缓冲会超出188416字节 Vec 限额；降低物理行数、合并四路读取的版本可以编译，但收益不足。
- 20轮 Sinkhorn、正式 checkpoint、BF16 mixed 与FP32 post/comb接口以及原精度阈值都保留。每个 PASS 还包括图/急切一致、同址换输入和输出保护区。
- 先前逐核探针证明的是 baseline48.py 的十四场景。分工有变化的候选若被采用，需要重新验证其实际逐核工作覆盖，不能直接沿用旧版的48核证据。

## 文件

- baseline48.py：开始本轮时的生产源码快照。
- run_candidates*.sh：排队执行脚本；独立快照由 --kernel-file 选择。
- build_*/report.json 和 build_*.log：CPU 编译结果，不算真机验证。
- run_*/report.json：精度结果、三个 L1 样本和泳道路径。
- summarize_candidates.py：重新生成 candidate_l1.csv 与 candidate_status.json；不选择快样本。

## 本次最终收尾（2026-10-11）

小档整数索引候选 `integer_small_tail_v2.py` 已完成十四档同进程L1对照，原输入和换输入后的三个输出全部逐bit一致。性能没有证明全档稳定收益，候选不采用；正式 `hc_pre.py` 保持 `baseline48.py`。28 μs目标未达成，按用户要求停止优化。

最终结果见 `run_paired_integer_small_hot14/report.json`、`comparison.csv`、`decision.json`，以及其中记录的全部真实L1泳道路径。完整解释见 `../../HC_PRE_OPTIMIZATION_LOG.md` 的“本次收尾”小节。
