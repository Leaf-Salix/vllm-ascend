# 当前局部交接方案的七档 CSA 对比

后续解除直接WAIT生产者限制后的复测见[新SDK七档报告](../superkernel_csa_seven_unrelated_pending_20261011/README.md)。
本目录保留Simpler `d00f8f421`下的历史结果，两轮不构成旧/新SDK交错因果对照。

2026-10-11，单卡任务 `task_20261011_154838_375836631548`，设备 2，退出 0 并释放设备。
本轮按用户最新要求，将此前只测 128K/B16 的限制扩展为七档。

## 对照口径

- 两版均使用当前私有 SDK：PyPTO `1f8206d09`、Simpler `d00f8f421`、PTOAS `4a5ce22a`。
  CSA 使用隔离环境的冻结源码，私有 vLLM-Ascend HEAD 为 `2cd33a9a`。
- 普通版使用 `deepseek_v4_flash_csa_scalar_overlap_demo`，不插入局部 Wait/Set；
  early_start 版使用 `deepseek_v4_flash_csa_all_boundaries`，启用此前已接入的局部边界。
  这是同 SDK 下的边界对照，不是 Native 对比，也不是改造前 SDK 与当前 SDK 的比较。
- 63 处 `allow_early_resolve` 设置逐项一致，保持原任务数学、分块和调度表达。
  [版本及设置记录](manifest.json)。
- CANN 9.2.0-beta.2、A3、TMR、NZ mode=2、atomic_add=0、deterministic_level=0。
  第 2 层真实权重，随机种子 1024 的合成历史，单请求验 6 token。
- 区间为单层 **HC_pre + norm + CSA + HC_post**。图捕获后每版 warmup 5 次，
  两版轮换先后顺序，各记录 30 次单次 NPUGraph replay 的 NPU Event 时间。
  缓存恢复在计时区间外；结果包含运行时调度，不是独立 InCore 计算时间。
- profiler 关闭；当前 runtime 的 profiler 路径会禁止实际提前交接，不能用其耗时替代本轮测量。
- 七档的形状特化保留同一份动态程序，因此每版只编译一次，再执行不同形状。
  [动态形状检查](dynamic_shape_checks.json)。

## 七档结果

单位 μs。最后一列为 `(普通 mean − early_start mean) / 普通 mean`，正数表示耗时减少。

| 档位 | 普通 min | 普通 mean | 普通 max | early_start min | early_start mean | early_start max | mean 耗时减少 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 638.20 | 653.38 | 673.54 | 636.78 | 656.31 | 683.98 | -0.45% |
| 128K/B8 | 745.80 | 760.03 | 783.62 | 748.22 | 758.13 | 772.84 | +0.25% |
| 128K/B16 | 970.28 | 999.46 | 1052.56 | 965.32 | 984.49 | 1008.92 | +1.50% |
| 128K/B24 | 1199.06 | 1248.08 | 1283.14 | 1203.12 | 1260.67 | 1294.46 | -1.01% |
| 8K/B16 | 726.90 | 746.48 | 768.92 | 715.98 | 737.04 | 758.92 | +1.26% |
| 8K/B24 | 873.24 | 923.16 | 991.34 | 884.72 | 928.95 | 962.28 | -0.63% |
| 8K/B32 | 1013.96 | 1075.10 | 1116.78 | 1018.04 | 1066.00 | 1113.32 | +0.85% |

本轮均值四档改善、三档回退，最大值五档下降、两档增加。
128K 各档百分比等权平均为 +0.0726%，8K 为 +0.4944%；按长短档 8:2 加权为 **+0.1569%**。
这是对本轮七档百分比的汇总，不是整模型加速，也不足以认定已有稳定的整体净收益。

| 档位 | 配对更快次数 / 30 | 配对耗时差中位数 μs（early_start − 普通） |
| --- | ---: | ---: |
| 128K/B4 | 13 | +3.31 |
| 128K/B8 | 16 | -0.67 |
| 128K/B16 | 25 | -12.44 |
| 128K/B24 | 9 | +16.95 |
| 8K/B16 | 22 | -10.11 |
| 8K/B24 | 12 | +7.33 |
| 8K/B32 | 20 | -6.77 |

128K/B16 的 min/mean/max 同时改善；128K/B24 的三项同时回退，不能用总加权值掩盖。
8K/B24 均值回退但最大值下降，两项都保留。没有仅凭这些总耗时认定任何 CSA task 为 Scalar-bound，
也没有将该差值全部归因于前一轮测得的 67 ns 标记开销。本轮不据此自动修改公共默认路径。

## 功能与证据

七档均通过普通版与 early_start 版的 9 项输出/缓存/状态逐 bit 对照，
改变输入后的图重放也逐 bit 一致；每档两版各 31 项保护区检查通过。
这是两条 PTO 路径之间的检查，不是 Native 精度或整模型 token 验收。

- [128K/B4 全部样本与功能结果](h131072_b4/results.json)
- [128K/B8 全部样本与功能结果](h131072_b8/results.json)
- [128K/B16 全部样本与功能结果](h131072_b16/results.json)
- [128K/B24 全部样本与功能结果](h131072_b24/results.json)
- [8K/B16 全部样本与功能结果](h8192_b16/results.json)
- [8K/B24 全部样本与功能结果](h8192_b24/results.json)
- [8K/B32 全部样本与功能结果](h8192_b32/results.json)
- [汇总 JSON](summary.json)、[设备任务日志](device_task.txt)。

此前约 67 ns 来自单block短任务中“有标记但无后继”相对同 SDK 普通 task 的均值差。
不插入 Wait/Set 可以避开那一项标记开销，本表普通版即走这条路径；
67 ns 不是每个 CSA task 的固定常数，也不能证明当前 SDK 的公共路径与旧 SDK 完全等价。

## 复现

私有根目录为工作区下 `.worktrees/scalar-overlap-demo-20261009`。
本目录保存扩展后的两个 Python 驱动和编译、执行、汇总脚本；驱动补充 batch/history 参数，
CSA kernel 与运行时源码在本轮测量中未改动。
将两个 `tmr_csa_full_128k_b16_*.py` 放入私有 vLLM-Ascend 的 `tests/pypto_test/`，
其余脚本放入私有根目录的 `artifacts/`，然后在私有根目录运行：

```bash
bash artifacts/compile_csa_seven_handoff_cleanup.sh
source env.sh
simpler/.claude/skills/onboard-arch-precheck/check.sh a2a3
task-submit --device auto --device-num 1 --max-time 1200 \
  --run "bash $SCALAR_OVERLAP_ROOT/artifacts/run_csa_seven_handoff_cleanup.sh"
python artifacts/summarize_csa_seven_handoff_cleanup.py \
  artifacts/csa_seven_handoff_cleanup_20261011
```

预编译产物和逐档完整 stdout 保留在上述私有 `artifacts/csa_seven_handoff_cleanup_20261011/`。
本轮完整保留 420 次计时样本，未按性能好坏删样本或追加筛选轮次。

后续只读审查见[early_dispatch到early_start的静态覆盖分析](early_dispatch_coverage.md)：
128K/B16展开为66组、973个单核block调用；入口及sync_start限制后最多剩62组、778个调用有后继资格，
实际early_dispatch/handoff命中数尚未记录。直接WAIT生产者限制、MIX和sync_start覆盖缺口需要单独审视。
