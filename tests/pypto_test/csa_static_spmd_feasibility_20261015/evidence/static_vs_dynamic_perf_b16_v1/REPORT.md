# 128K/B16：单 SPMD 与原动态 PTO 性能对比

用户于 2026-10-16 要求只比较单 SPMD 与原动态 PTO，不测 Native。

## 被测范围与口径

- history=131072、batch=16、S=6、T=96；正式 W8A8 权重，第 4 层。
- 原动态 PTO 使用 `results/hc_pre_layer_integration_20261012/baseline_ops` 冻结实现，
  与静态排程提取、上一轮精确比较同源。
- 单 SPMD 使用 `static_local_signal/h131072_b16/entry_jit.py` 与已验证的
  `local_signal_adapter.py`；notify 前包含本核 PIPE_ALL。
- 两边在同卡同进程、同一流串行执行，共用权重、输入和缓存地址，AB/BA 交替。
- 每次调用前恢复完全相同的缓存初态。恢复和输出毒化不进入计时区间；进度不清零。
- 先编译及 eager 比较，再捕获只含 PTO 调用的 ACLGraph；重放结果先对动态 eager
  精确比较。该图仅用于固定输入计时，不代表完整请求生命周期或连续 decode 验收。
- 设备 Event：每侧预热 5 次、采样 20 次。Event 在图外记录，不复用图内旧时间戳。
- L1：每侧独立 3 个窗口，一窗口只运行一侧一次图重放。
  主指标为原始 `aicore_tasks` 的 `max(end)-min(start)`，按记录中的设备时钟换算。
  不把收任务时间当成 incore 开始，不累加各核时长；保留内部等待及任务间隙。
- 两边使用同一独立 PyPTO/Simpler 调试安装、TMR runtime 和 L1 构建配置。
  Event 采样时 DFX 窗口关闭；L1 与 Event 分开报告，不混合样本。
- 不执行 Native attention forward，不使用此前 Native 性能作为本次对照。

## 首轮任务

- task ID：`task_20261016_183457_19106108116`
- 输出目录：`../results/csa_static_spmd_feasibility_20261015/static_vs_dynamic_perf_b16_v1`
- 启动：`run_static_vs_dynamic_performance.sh <全新输出目录>`，必须通过 `task-submit --device auto`。
- 最终状态：completed (exit=0)，输出、状态、信号与保护区检查通过。

测试入口为 `benchmark_static_vs_dynamic.py`，未修改生产 CSA、数学计算体或依赖组件。

## 最终结果

主口径为 L1 incore 首尾跨度，单位 μs。动态 PTO 共 66 个计算 task、973 个物理核执行记录；
单 SPMD 为 1 个 MIX task、72 个物理核执行记录。两边均覆盖 24 AIC + 48 AIV。

| 采样 | 原动态 PTO | 单 SPMD |
| --- | ---: | ---: |
| L1 第 1 次 | 897.70 | 1519.30 |
| L1 第 2 次 | 900.14 | 1380.22 |
| L1 第 3 次 | 877.42 | 1350.06 |
| **L1 中位数** | **897.70** | **1380.22** |
| Event 中位数（各 20 次） | 942.62 | 1467.75 |

单 SPMD 的 L1 中位数增加 482.52 μs，耗时增加 **53.75%**，动态/静态比为 **0.6504×**。
设备 Event 中位数增加 525.13 μs，耗时增加 **55.71%**。当前静态化版本没有取得加速。

L1 动态范围 877.42～900.14 μs；静态范围 1350.06～1519.30 μs。
Event 动态范围 926.12～967.02 μs；静态范围 1442.78～1819.22 μs，所有样本保留，未剔除离群值。
L1 与 Event 是不同采样窗口，不能把两者中位数直接相减当作某个阶段的精确开销。

退步已经发生在 incore 首尾区间内，不能只归因于 Simpler/orchestrator 的外围首尾调度。
单 SPMD 的 L1 仅显示每核整段融合计算，没有内部阶段时间戳；本次结果尚不能分别量化
notify/wait、缓存维护、PIPE_ALL、静态排程或融合后编译结果各自的影响。未继续做优化。

## 正确性和范围

- 单 SPMD eager、两侧图捕获后重放、两侧最终 L1 重放的输出、Top-K、六份 cache/state
  均与动态 eager 参考逐元素完全一致。无 Native forward 比较。
- 单 SPMD 累计执行 30 个代际，进度区不清零；72 核末次进度和工作区保护通过。
- Native 格式缓存的未写区域、只读 metadata 保护检查通过。
- 这是固定输入、固定地址的计时图；每次恢复缓存，不代表连续 decode 轨迹、动态 B/S
  或完整整网性能。

## 数据与复现

- 原始结果：`../results/csa_static_spmd_feasibility_20261015/static_vs_dynamic_perf_b16_v1/result.json`。
- 汇总：同目录 `summary.json`，含全部 20 次 Event 及 3 次 L1 数据。
- 泳道：同目录 `l1/dynamic_pto/step_{1,2,3}/swimlane.json` 和
  `l1/single_spmd/step_{1,2,3}/swimlane.json`，各目录同时保留原始记录与名称映射。
- `download/` 汇总六份重命名的泳道、报告与 JSON；可以直接用于 Perfetto/Chrome tracing。
- 归档副本：本目录 `evidence/static_vs_dynamic_perf_b16_v1/`。

正式权重、大张量、动态库及设备二进制不入归档；只记录本地路径，没有计算 hash。
