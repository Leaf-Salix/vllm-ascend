# O_A复用最新上游的行块×列块网格

基底c7a52af5；源码参考pypto-lib官方main `216456332c2a74d89cca23b7824dab264ce34bff`，2026-09-27浅拉取确认。
上游O_A按行块×N块创建SPMD，原接入ND相同，但NZ因早期非负偏移校验而只按N分配任务、行块在核内串行。
现NZ也采用二维网格，余数通过`max(nf,0)`显式标明非负；合法block下值不变。Native物理NZ权重直接复用。
每个输出tile的K256遍历、stage2流水、FP32累加/写回不变；量化仍等待本组所有O_A块，组间不加屏障。
原O_A→quant→O_B登记顺序与上游保持一致。没有改精度版。

## B40主先导

A3 device0，8K/B40/S6/TP1，正式第4层权重与合成输入历史，mode2/atomic1/deterministic0、EPLB关闭。
第二CSA层metadata复用；5次预热、20次无profiler计时，另4个DFX窗口。
完整CPU根/PTOAS/AICPU编译通过，任务`task_20260927_175801_275821911865`退出0。

| 指标 μs | Q_A先行基底 | 行列并行 |
| --- | ---: | ---: |
| CSA本体均值 | 1327.859 | 1318.049 |
| 本体p50 | 1322.030 | 1319.180 |
| 本体p95 | 1367.880 | 1364.580 |
| 同轮Native均值 | 1427.994 | 1419.105 |
| 拆分+本体+写回完整均值 | 1674.927 | 1674.572 |
| O_A Worker数量 | 32 | 64 |
| O_A启动分散 | 99.92–109.58 | 91.58–93.74 |
| O_A首kernel开始→末kernel结束 | 164.06–174.10 | 128.28–130.08 |
| merge结束→HC_post结束 | 305.16–312.86 | 275.96–285.62 |

本体均值−0.74%，完整路径基本不变；同轮Native也变化约−0.62%，所以不宣称本体已有大幅加速。
保留依据是同段O_A窗口明确缩短34–46 μs、尾段缩短约20–37 μs，且本体均值/p50/p95未退化。
64个任务各做一个行块，单任务核内均值从86.60–91.74降到41.74–42.09 μs不能称计算加速两倍。
B40实际240行，覆盖第二行块112行尾部；保护区/Top-K结构通过、非有限值0。
Native输出max_abs0.03125、RMSE0.003292595、Top-K替换901，零容差仍FAIL。

## B24短尾行块验证

任务`task_20260927_180312_278526728381`退出0，同口径5/20及4个DFX窗口；
B24/S6为144行，O_A第二块只有16有效行。64 Worker正常运行，O_A整组窗口101.22–106.42 μs。
保护区/Top-K结构通过、非有限值0；Native输出max_abs0.03125、RMSE0.003298233、Top-K替换545，零容差仍FAIL。
本体均值983.49/p50 985.42/p95 998.98 μs，Native1146.39，完整PTO1317.73 μs。
未重跑B24旧基底，因此不把当前与历史da2七档的差值归因于本项。
[B24原始timing](../../csa_split_optimization_20260927/o_a_row_parallel/h8192_b24/timing/report.json)、
[B24原始泳道](../../csa_split_optimization_20260927/o_a_row_parallel/h8192_b24/swimlane/dfx/merged_swimlane.json)。

## 对725 μs参考的适用范围

历史上游实际Worker窗口727.98 μs，其B16记录O_A为64 Worker，但缺完整配置；不能拿它与B40绝对耗时比较。
当前B16只有一个O_A行块，本次不会改变任务数量；本次从最新源码吸收的是大batch的二维并行策略。
B16当前对上游仍有O_A核内差距及前段调度差异，未被本次解决；[完整上游对照](../upstream_725/README.md)。

## 证据

- [汇总数据](report.json)、[统计脚本](summarize.py)、[可复建补丁](candidate.patch)、[隔离编译脚本](compile_candidate.py)。
- [B40设备脚本](run.sh)、[B24尾行块脚本](run_b24.sh)。
- [B40原始timing](../../csa_split_optimization_20260927/o_a_row_parallel/h8192_b40/timing/report.json)。
- [B40原始泳道](../../csa_split_optimization_20260927/o_a_row_parallel/h8192_b40/swimlane/dfx/merged_swimlane.json)。

未追加七档全测或16卡；七档同源码及新算术路径token/DSpark仍须在本阶段收尾补齐。
