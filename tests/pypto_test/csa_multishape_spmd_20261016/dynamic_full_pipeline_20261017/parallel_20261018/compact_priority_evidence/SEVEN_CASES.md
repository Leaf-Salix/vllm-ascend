# compact 优先调度：同源码七档完整验收记录

2026-10-18。七档均完成 exit=0，当前 fixture 的完整精度、图重放、代际及保护区检查通过。每档同进程三方各五窗，合计 105 窗。七档均比 593 旧 SPMD 快，仍全部慢于同轮原动态分 task；相对原 task 降低至少 30% 的门槛为 0/7，不能标记该性能目标完成。

本记录直接复用已完成的 `remaining_summary.json`；[SEVEN_CASES.json](SEVEN_CASES.json) 是其完整原文副本。归档期间未重新提交、编译或执行设备测试，未修改原始结果。

## 测量方法与验收范围

- 每档原动态分 task、593 旧 SPMD、候选在同一进程、相同根地址下比较，候选与旧 SPMD 的策略相同，workspace 和信号独立。每方交替顺序采五窗，以五个原始样本的中位数比较。各档之间不计算因果收益，也不叠加其它候选结果。
- 完整 L1 指该次 incore 最早开始至最晚结束，包含内部等待及阶段间隙，不包含外围 orchestrator；每窗 72 核、时钟频率 50 MHz，两个 SPMD 实现均为单 MIX task。未使用 phase_clock 插桩数字报收益。
- 七档均为 S6、正式第 4 层 W8A8 权重、合成历史。当前 fixture 的 output/TopK/score/cache-state 与原动态 task 做 exact 检查；三代 A→B→A、同址图、历史及页表重放、代际协议、原生内存保护区与 scratch 保护区均通过，包含 L1 后检查。
- 七档设备执行均使用 `--wait-backoff --l1-windows 5`；既有失败轮询 100 tick 退避 hook 对两份 SPMD 同时开启。本结果不将该测试 hook 解释为正式默认变化。
- Native attention 正确性和性能未运行，其继承差异未被本记录验收；`native_memory_guards` 只表示对应内存保护区检查。token 生成、整模型端到端、全部动态 metadata 边界、整层 mixed Q/空 rank、连续 decode 状态轨迹均不在验收范围。七档 S6 通过不能替代 S1/S5 或这些其它范围。
- 性能门槛逐档为 `候选中位数 <= 同轮原 task 中位数 × 0.70`。相对 593 的改善不能替代此门槛。

候选冻结源码：

`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/ultra_compact_priority_20261018/source_snapshot`

593 参考源码：

`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/baseline_source`

正式包只迁入 `cube_schedule.py`、`vector_schedule.py` 的 compact 阶段前移。33 份源码的既有逐字比较记录保留在 [formal_migration.json](formal_migration.json)；本次只更新其七档结果状态，没有再次改源码或叠加 L0/物理 plan 候选。

## 完整 L1 中位数与门槛

单位 μs。降低百分比为 `100 × (1 - 候选 / 对照)`；负数表示候选更慢。表中四舍五入，完整精度保留在 JSON。

| 档位 | 原 task | 593 旧 SPMD | 候选 | 对 593 降低 | 对原 task 降低 | 30% 目标上限 | 达标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128K/B4/S6 | 513.88 | 559.82 | 531.92 | 4.984% | -3.511% | 359.716 | 未达 |
| 128K/B8/S6 | 645.16 | 712.98 | 684.44 | 4.003% | -6.088% | 451.612 | 未达 |
| 128K/B16/S6 | 861.48 | 913.06 | 888.38 | 2.703% | -3.123% | 603.036 | 未达 |
| 128K/B24/S6 | 1155.24 | 1193.86 | 1163.08 | 2.578% | -0.679% | 808.668 | 未达 |
| 128K/B28/S6 | 1281.36 | 1357.70 | 1332.30 | 1.871% | -3.975% | 896.952 | 未达 |
| 8K/B16/S6 | 668.74 | 712.50 | 700.14 | 1.735% | -4.695% | 468.118 | 未达 |
| 8K/B32/S6 | 998.48 | 1024.52 | 1001.56 | 2.241% | -0.308% | 698.936 | 未达 |

## 全部原始五窗样本

每行按该实现窗口编号 00→04 列出五个 incore span，单位 μs；未筛选窗口或去除离群点。实际三方交替顺序由 JSON 的 `windows` 列表保留。

| 档位 | 实现 | 窗 00 | 窗 01 | 窗 02 | 窗 03 | 窗 04 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4/S6 | 原动态分 task | 504.76 | 516.70 | 485.42 | 513.88 | 551.14 |
| 128K/B4/S6 | 593 旧 SPMD | 559.82 | 545.54 | 559.36 | 564.56 | 585.00 |
| 128K/B4/S6 | compact 优先候选 | 531.92 | 546.24 | 521.74 | 537.84 | 530.30 |
| 128K/B8/S6 | 原动态分 task | 619.28 | 628.80 | 648.44 | 647.32 | 645.16 |
| 128K/B8/S6 | 593 旧 SPMD | 712.98 | 718.90 | 706.60 | 709.00 | 729.30 |
| 128K/B8/S6 | compact 优先候选 | 682.84 | 684.44 | 685.70 | 691.68 | 684.44 |
| 128K/B16/S6 | 原动态分 task | 852.68 | 847.48 | 863.38 | 861.48 | 881.46 |
| 128K/B16/S6 | 593 旧 SPMD | 913.06 | 917.82 | 901.38 | 902.34 | 917.62 |
| 128K/B16/S6 | compact 优先候选 | 888.22 | 890.76 | 889.94 | 885.92 | 888.38 |
| 128K/B24/S6 | 原动态分 task | 1169.84 | 1155.24 | 1169.28 | 1136.32 | 1153.78 |
| 128K/B24/S6 | 593 旧 SPMD | 1193.86 | 1183.22 | 1194.22 | 1187.26 | 1228.46 |
| 128K/B24/S6 | compact 优先候选 | 1159.56 | 1184.54 | 1163.08 | 1161.14 | 1181.12 |
| 128K/B28/S6 | 原动态分 task | 1281.36 | 1279.44 | 1305.34 | 1281.08 | 1319.12 |
| 128K/B28/S6 | 593 旧 SPMD | 1357.70 | 1342.26 | 1352.32 | 1361.66 | 1366.16 |
| 128K/B28/S6 | compact 优先候选 | 1340.16 | 1331.58 | 1332.12 | 1332.30 | 1344.34 |
| 8K/B16/S6 | 原动态分 task | 664.30 | 676.46 | 668.74 | 661.82 | 672.14 |
| 8K/B16/S6 | 593 旧 SPMD | 709.96 | 706.16 | 721.40 | 712.50 | 719.60 |
| 8K/B16/S6 | compact 优先候选 | 698.34 | 704.86 | 694.70 | 707.54 | 700.14 |
| 8K/B32/S6 | 原动态分 task | 998.48 | 1041.28 | 1010.42 | 985.76 | 983.26 |
| 8K/B32/S6 | 593 旧 SPMD | 1024.52 | 1024.26 | 1027.80 | 1010.24 | 1029.14 |
| 8K/B32/S6 | compact 优先候选 | 1001.86 | 999.22 | 1005.68 | 989.46 | 1001.56 |

## 任务与原始证据索引

128K/B16、8K/B32 使用先前已经完成的两个代表任务，没有为补齐七档而重测；其余五档各提交一次。所有任务经 `task-submit --device auto --max-time 1200` 运行。下表的两个 JSON 是各档原始完整报告和三方同进程报告；[SEVEN_CASES.json](SEVEN_CASES.json) 逐窗保留全部 105 个 `raw` 与 105 个 `trace` 的绝对路径、incore span、时钟、核数和 task 数。

| 档位 | task ID | 原始报告 |
| --- | --- | --- |
| 128K/B4/S6 | `task_20261018_015905_72747510107` | [result](../compact_priority_h131072_b4_v1/result.json) / [source_matrix](../compact_priority_h131072_b4_v1/source_matrix/result.json) |
| 128K/B8/S6 | `task_20261018_015905_7275183064` | [result](../compact_priority_h131072_b8_v1/result.json) / [source_matrix](../compact_priority_h131072_b8_v1/source_matrix/result.json) |
| 128K/B16/S6 | `task_20261018_013723_19521519782` | [result](../compact_priority_h131072_b16_v1/result.json) / [source_matrix](../compact_priority_h131072_b16_v1/source_matrix/result.json) |
| 128K/B24/S6 | `task_20261018_015906_7275582901` | [result](../compact_priority_h131072_b24_v1/result.json) / [source_matrix](../compact_priority_h131072_b24_v1/source_matrix/result.json) |
| 128K/B28/S6 | `task_20261018_015906_7275999294` | [result](../compact_priority_h131072_b28_v1/result.json) / [source_matrix](../compact_priority_h131072_b28_v1/source_matrix/result.json) |
| 8K/B16/S6 | `task_20261018_015906_7276406980` | [result](../compact_priority_h8192_b16_v1/result.json) / [source_matrix](../compact_priority_h8192_b16_v1/source_matrix/result.json) |
| 8K/B32/S6 | `task_20261018_014358_34138622015` | [result](../compact_priority_h8192_b32_v1/result.json) / [source_matrix](../compact_priority_h8192_b32_v1/source_matrix/result.json) |

七档均为 `completed (exit=0)`；三方五窗、同进程和同根地址条件以及逐项 exact/保护区/协议检查的收取方法，保留在 [remaining_five/summarize.py](remaining_five/summarize.py)。本次只复制该脚本，没有重新执行。

## 同源码下的策略

| 档位 | Query group | QK panel | Score tile | Prefetch | Balanced | Roots | OA cols | OB rows |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 128K/B4/S6 | 6 | 64 | 1024 | 1 | true | 1 | 128 | 32 |
| 128K/B8/S6 | 6 | 64 | 1024 | 1 | true | 1 | 128 | 96 |
| 128K/B16/S6 | 6 | 64 | 1024 | 1 | true | 1 | 128 | 96 |
| 128K/B24/S6 | 6 | 64 | 1024 | 1 | true | 1 | 256 | 128 |
| 128K/B28/S6 | 6 | 64 | 1024 | 1 | true | 1 | 256 | 128 |
| 8K/B16/S6 | 2 | 128 | 768 | 0 | false | 2 | 128 | 96 |
| 8K/B32/S6 | 2 | 128 | 768 | 0 | false | 2 | 256 | 128 |

七档均开启 `TO_L0`、`OUTPUT_GROUP_PIPELINE`、`HC_ROW_DEPENDENCIES`，关闭 `ATTENTION_QUERY_METADATA`、`HC_CONSUMER_REDUCE`。完整策略逐档保留在 JSON；这是动态策略选择，不是冻结运行时 Q/L/position/page。

## 提交、收取文本与本地编译产物

`remaining_five/` 直接复制原私有目录的 `submit_remaining.py`、`summarize.py`，以及五档各一份 `submit_*.log`、`status_*.log`、`task_*.log`，并复制 [compact_priority_remaining_jobs.json](remaining_five/compact_priority_remaining_jobs.json)。其中 submit 日志为提交 stdout/stderr，status 日志为队列终态，task 日志为完整设备任务 stdout/stderr；原始文件全部保留。登记表保留五条精确命令、输出路径和 task ID。没有单独保存过收取脚本的终端 stdout，因此这里不将重建文本冒充原始 stdout；完整收取结果由 SEVEN_CASES.json 保留。

两个先完成代表任务的原始日志仍为 [device_long_v1.log](device_long_v1.log)、[device_medium_v1.log](device_medium_v1.log)，提交记录仍为 `device_job.json`、`device_medium_job.json`。

[LOCAL_ARTIFACTS_REMAINING.json](LOCAL_ARTIFACTS_REMAINING.json) 登记七档 `build_output` 总文件数/总字节、各编译单元目录、全部 `.bin` 和至少 1 MiB 的 `.cpp`/`.pto` 的绝对路径及字节数。这些文件只在原结果目录本地保留，未复制编译二进制或大生成物。统计仅访问路径和文件大小，没有计算 hash。

原收取记录时间（UTC）：`2026-10-17T18:06:13.776263+00:00`。
