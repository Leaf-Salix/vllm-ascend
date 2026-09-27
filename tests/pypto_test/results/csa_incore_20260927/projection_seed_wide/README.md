# Q_A/KV连续整行清零：保留

第二项核内优化：性能版Q_A的ND/NZ、KV投影seed由16×128窄块反复清零，改成复用16×1024/16×512零tile，整行连续写出。
保持每个投影一个seed任务、有效/padding行覆盖、split-K、atomic规则和跨任务依赖，精度版不动。
完整CPU根编译通过；以下两个设备任务退出0，每档四个DFX窗口加现有单层诊断，无新增本体计时。
配置沿用layer4正式权重、合成输入/历史、S6/TP1/mode2/atomic1/deterministic0，无EPLB。

| H / B | 任务 | 改前 μs | 改后 μs |
| --- | --- | ---: | ---: |
| 8K / 40 | Q_A seed | 24.54–28.04 | 6.94–8.50 |
| 8K / 40 | KV seed | 11.96–13.14 | 4.96–5.20 |
| 128K / 4 | Q_A seed | 3.68–3.90 | 1.46–1.76 |
| 128K / 4 | KV seed | 2.12–2.36 | 1.24–1.36 |

范围为四窗口block核内均值。B40基底f5482bb5，B4使用da2e2368已有seed对照（该段算术未改）；
B4只用于小档/padding检查，不冒充同轮全链路差分。B4真实T24且清零到32行；B40 T240。
Q_A/KV matmul耗时基本重叠，没有把seed收益转算为未经测量的CSA收益。
保护区/Top-K结构通过，所有输出/状态非有限值0；Native输出max_abs两档均0.03125。
B40 RMSE 0.003292715→0.003292750，B4为0.004322217；Top-K替换分别901、184不变。
Native零容差仍FAIL，atomic重放存在既有差异；未做新的整模型token/DSpark验收。

任务：task_20260927_172455_25647421651、task_20260927_172622_257559010067。
复现：通过task-submit调用 `results/csa_split_optimization_20260927/run_case.sh projection_seed_wide H B swimlane`。
[统计与原始泳道](report.json)、[补丁](candidate.patch)、[隔离CPU编译脚本](compile_candidate.py)。
