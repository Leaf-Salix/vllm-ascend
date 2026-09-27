# 核内复评：全有效compressed KV块跳过清零，撤回

基底ec12e0e9（已保留PV N128/K128），只评估性能版，未纳入生产。
与此前[kv_valid_nozero](../kv_valid_nozero/README.md)是同一优化方向；本次在跨query流水、提前KV发布和PV分块后的新基底复评，不能称为新策略。
Native `SASVectorBlock::ProcessVec0L/CopyInKv` 直接覆盖有效KV行，PTO每半块先清零64×512 BF16。
候选让plan额外row_min标记0空/1部分/2全有效，压缩全有效块跳过清零；SWA和部分块仍原地清零。
最新pypto-lib2164563的Sparse实现仍先初始化gather缓冲；这里是吸收Native的覆盖写策略，未照抄上游PTO。

## 两档结果

四个独立DFX窗口，表中核内值是每窗口同名block均值的平均；含核内同步等待，不是纯算术。
基底引用已保留PV候选的同口径结果；任务数量/依赖未改，Native也未改。

| μs | 8K/B16基底 | 候选 | 128K/B16基底 | 候选 |
| --- | ---: | ---: | ---: | ---: |
| qk_pv AIC均值 | 126.72 | 121.42 | 163.35 | 173.70 |
| qk_pv AIC四窗口范围 | 116.66–132.71 | 117.60–124.88 | 161.54–164.25 | 170.83–176.06 |
| qk_pv AIV均值 | 128.66 | 123.45 | 165.24 | 175.64 |
| plan核内均值 | 6.64 | 6.96 | 7.16 | 6.84 |
| 无profiler本体均值 | 786.57 | 772.71 | 1278.32 | 1245.35 |
| 本体p50 | 788.71 | 775.48 | 1197.83 | 1198.17 |
| 本体p95 | 803.64 | 787.78 | 1541.40 | 1529.18 |
| 同轮Native均值 | 935.92 | 933.55 | 1309.96 | 1307.78 |

短档AIC−4.18%，范围重叠；长档AIC+6.34%，范围不重叠。plan额外工作已经计入。
长档本体均值下降但p50持平、长尾仍在，不拿它掩盖核内退化；不保留、不扩测七档。
尚未证明退化具体来自新增控制/同步还是新的流水关系，不按逻辑清零字节减少推断硬件收益。

## 功能与编译过程

CPU全编译和生成代码检查后，task_20260927_200537_381384626131退出0。
B3短历史/部分有效块/18token尾部解析检查589824元素零差；两档保护区/Top-K结构通过、非有限值0。
Native零容差仍FAIL；8K/128K输出max_abs=0.03125/0.0390625，RMSE=0.0033201356/0.0041770661，Top-K替换366/670。
仅单卡layer4正式权重＋合成输入/历史，第二CSA metadata复用；S6/TP1/mode2/atomic1/deterministic0，无EPLB。
两档各5预热/20计时＋4窗口DFX，未进行新整模型token/DSpark验收。

条件分配两个tile会超过UB预算；复用同一tile的fillpad返回值需要保持类型一致。
初版没有消费fillpad返回值，最终生成C++缺少必要清零；B3虽通过也不能证明安全。
旧任务task_20260927_195846_37833505425已主动终止，退出130，任何部分数据均排除。
最终候选两分支都消费PadValue.zero结果；生成C++确认同一UB地址、SWA/部分块有效行0触发TFILLPAD，
全有效块有效行64不写padding。新数据独立使用`incore_sparse_full_valid_no_zero_v2`标签，没有覆盖旧数据。

[补丁](candidate.patch)、[CPU重建](compile_candidate.py)、[运行](run.sh)、[原始路径及四窗口](report.json)、[B3解析结果](tail_b3_v2/report.json)。
统计：`python tests/pypto_test/results/csa_incore_20260927/compare_candidate.py incore_sparse_full_valid_no_zero_v2 tests/pypto_test/results/csa_incore_20260927/sparse_full_valid_no_zero/report.json`。
