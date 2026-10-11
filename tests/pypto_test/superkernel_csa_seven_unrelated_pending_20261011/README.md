# 去除直接生产者限制后的七档 CSA 性能复测

2026-10-11，单卡任务`task_20261011_170750_320296621605`，设备2，退出0并释放设备。

## 对照口径

- 两版均为PTO，使用同一私有SDK：PyPTO `fdac65f66`、Simpler `5c8418a02`、PTOAS `4a5ce22a`。
  私有vLLM-Ascend基点`2cd33a9a`，CSA沿用上轮冻结包。
- 普通版为`deepseek_v4_flash_csa_scalar_overlap_demo`，无局部Wait/Set；
  early_start版为`deepseek_v4_flash_csa_all_boundaries`，保留既有局部边界。
  新SDK已移除gated交接要求本核前驱必须是直接WAIT生产者的额外限制。
- 63处allow_early_resolve设置逐项一致；未修改任务数学、分块、核选择、ACK后派发或sync_start。
- CANN9.2.0-beta.2、TMR、NZ mode=2、atomic_add=0、deterministic_level=0。
  第2层真实权重、seed=1024合成历史、单请求验6 token；区间含**HC_pre + norm + CSA + HC_post**。
- profiler关闭；每版warmup 5次，轮换顺序各30次单次NPUGraph重放，以外层NPU Event计时。
  缓存恢复在计时之外，耗时包含运行时调度。七档每版各复用一份动态编译产物。
- [版本及调度设置](manifest.json)、[动态形状检查](dynamic_shape_checks.json)。

## 七档结果

单位μs。最后一列为`(普通mean − early_start mean) / 普通mean`，正数表示耗时减少。

| 档位 | 普通 min | 普通 mean | 普通 max | early_start min | early_start mean | early_start max | mean 耗时减少 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 633.16 | 653.30 | 678.54 | 625.40 | 645.70 | 668.24 | +1.16% |
| 128K/B8 | 749.06 | 772.00 | 827.60 | 745.92 | 755.68 | 764.04 | +2.11% |
| 128K/B16 | 962.98 | 983.60 | 1001.54 | 958.50 | 984.32 | 1003.54 | -0.07% |
| 128K/B24 | 1201.76 | 1251.98 | 1290.02 | 1187.38 | 1241.17 | 1286.16 | +0.86% |
| 8K/B16 | 733.74 | 757.86 | 788.38 | 718.48 | 740.16 | 760.32 | +2.34% |
| 8K/B24 | 886.82 | 919.19 | 969.92 | 906.46 | 930.81 | 954.94 | -1.26% |
| 8K/B32 | 1027.30 | 1063.97 | 1108.66 | 1020.38 | 1067.32 | 1115.72 | -0.31% |

均值四档改善、三档回退；最大值五档下降、两档增加。
128K组内各档百分比等权平均减少**1.0164%**，
8K为**0.2525%**，长短档8:2加权减少**0.8636%**。
这是单层七档的统计口径，不是整模型加速。128K/B16均值略慢0.07%、max增加2.00μs；
8K/B24均值慢1.26%但max下降14.98μs；8K/B32均值慢0.31%且max增加7.06μs，不能只报告加权正收益。

| 档位 | 配对更快次数 / 30 | 配对耗时差中位数 μs（early_start − 普通） |
| --- | ---: | ---: |
| 128K/B4 | 23 | -7.64 |
| 128K/B8 | 27 | -14.84 |
| 128K/B16 | 13 | +1.87 |
| 128K/B24 | 20 | -12.54 |
| 8K/B16 | 25 | -22.97 |
| 8K/B24 | 11 | +13.57 |
| 8K/B32 | 11 | +4.87 |

上轮同SDK内部对照的加权耗时减少0.1569%，本轮为0.8636%。
两轮SDK不同、运行时间不同，普通版基线本身也波动，因此不能将两轮相减直接解释成
删除此条件的精确因果收益；以本轮同SDK、同卡交错对照表作为本轮结论。
[上一轮逐档效果对照](previous_run_comparison.json)。

## 功能与覆盖范围

七档均通过9项输出/缓存/状态逐bit对照，改变输入后的图重放也逐bit一致，
每档两版各31项保护区检查通过。这验证两条PTO边界路径，未新增整模型token或Native精度测试。

当前仍未接入idle算子前缀、MIX和sync_start后继；需要前驱异步尾部与经检查的后继Wait边界，
并且只在前驱返回时探测一次后继。泳道/PMU/参数dump路径仍禁用实际交接。
这些覆盖限制与必须保留的输入数据及本核流水等待分开处理；本轮未更改这些条件。
实际CSA的early_dispatch/handoff命中数尚未采集，不能将剩余覆盖缺口或总耗时当作其计数。

- [128K/B4 全部样本与功能检查](h131072_b4/results.json)
- [128K/B8 全部样本与功能检查](h131072_b8/results.json)
- [128K/B16 全部样本与功能检查](h131072_b16/results.json)
- [128K/B24 全部样本与功能检查](h131072_b24/results.json)
- [8K/B16 全部样本与功能检查](h8192_b16/results.json)
- [8K/B24 全部样本与功能检查](h8192_b24/results.json)
- [8K/B32 全部样本与功能检查](h8192_b32/results.json)
- [汇总JSON](summary.json)、[设备任务日志](device_task.txt)。

## 复现与保存

私有根目录为工作区下`.worktrees/scalar-overlap-demo-20261009`。
本目录保存两个Python驱动及编译、执行、汇总脚本；驱动与上轮相同。
将驱动放入私有vLLM-Ascend的`tests/pypto_test/`，脚本放入私有根目录`artifacts/`后执行：

```bash
bash artifacts/compile_csa_seven_unrelated_pending.sh
source env.sh
simpler/.claude/skills/onboard-arch-precheck/check.sh a2a3
task-submit --device 2 --timeout 1800 --max-time 1200 \
  --run "bash $SCALAR_OVERLAP_ROOT/artifacts/run_csa_seven_unrelated_pending.sh"
python artifacts/summarize_csa_seven_handoff_cleanup.py \
  artifacts/csa_seven_unrelated_pending_20261011
```

预编译产物与逐档完整stdout保留在私有`artifacts/csa_seven_unrelated_pending_20261011/`。
420次样本全部保留，未按结果筛样本或追加挑选轮次。基准SDK的对应源码没有在本轮继续修改。
