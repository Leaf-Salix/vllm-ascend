# PTO 内直接更新 Native Indexer slot

2026-09-27。基底为3d1f0f65算子（16cec785只新增文档/证据）。
按用户最新要求，只改4个PTO性能版源码文件，另调整单卡诊断脚本；
Native算子、分配器、页表、slot、调度、精度版保持原样。

## 改动

- adapter把已有Native物理页描述符作为一个额外InOut参数传给PTO，复用原内存，无转换。
- 已有Compressor key写任务把同一份INT8量化结果写到逻辑连续缓存和Native物理slot。
- 已有串行scale任务把同一FP16数值写到两侧；对Native的64B scale区做读改写，保留邻居。
- 删除CSA外的Torch slot定位、取行和两次scatter。无额外PTO task，不改变Score算术/调度策略。
- 入口仍按页表复制完整历史；不把key/scale分离等同于请求历史连续，也不宣称已完全零拷贝。

## 单卡两档结果

正式第4层权重＋合成历史，第二CSA metadata复用，TP1/S6/mode2/atomic1/deterministic0，无EPLB。
5预热20次无profiler计时，每档另外4个DFX窗口；单位μs。

| 档位 | 原完整PTO | 新完整PTO | 改善 | 同轮Native | 新PTO对Native | 新完整PTO p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K/B16 | 1079.16 | 888.64 | -17.65% | 926.94 | -4.13% | 887.46 / 897.10 |
| 128K/B16 | 1645.76 | 1382.39 | -16.00% | 1301.29 | +6.23% | 1353.10 / 1653.08 |

| 档位 | 原本体 | 新本体（含核内直接提交） | 新本体 p50 / p95 | 新入口load | 原外部commit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K/B16 | 790.78 | 801.55 | 798.77 / 818.04 | 98.35 | 216.26 |
| 128K/B16 | 1234.06 | 1261.10 | 1180.34 / 1466.68 | 184.49 | 220.43 |

外部commit调用已删除，其新开销记为已消除；不是测一张空图来声称0μs。
分阶段独立图计时不严格相加为完整图，不能只拿本体与Native比较。
scale提交核内四窗口：8K从5.40～5.64增至9.32～11.44，128K当前10.88～11.74μs。
长档Score AIC当前312.97～323.83，对基底310.49～313.52，未证明此项核内改善。
本次保留依据是**两档完整路径均明显加速**，而不是把写回移入本体后隐藏成本。
长档完整路径仍慢于Native，750μs目标和长尾尚未解决。

## 验证与证据

- 完整PTOAS＋AICPU CPU编译通过，生成kernel确认每key写128B、scale串行读改写64B。
- 两档任务task_20260927_205352_412770127178退出0；Native slot内容与连续缓存精确一致，
  保护区、metadata、Top-K结构及有限值检查通过。
- Native零容差输出仍FAIL，8K/128K max_abs为0.03125/0.0390625；
  Top-K替换366/670，不能据此声称token/DSpark或逐bit精度验收通过。
- B4/H4095补位图检查使用atomic0/deterministic1，task_20260927_205353_412802212678退出0；
  同一图4→3→1→4有效请求的输出、cache/state、compact metadata及保护区检查全部通过。
- 未重跑七档及16卡；最近完整七档仍是[3d1f0f65基底](../csa_incore_20260927/final_3d1f0f65/README.md)。

[数值、所有泳道路径与检查](report.json)，[只读汇总](summarize.py)，
[CPU编译脚本](compile_candidate.py)，[两档命令](run.sh)，[补位命令](padding.sh)。
原始计时及泳道目录：
[8K/B16](../csa_split_optimization_20260927/cache_direct_commit/h8192_b16/)、
[128K/B16](../csa_split_optimization_20260927/cache_direct_commit/h131072_b16/)。

下一项只在PTO中评估按物理页表直接读取，消除入口load，同时保留现有Cube/量化策略；
非连续页、尾页和补位必须有边界处理，不改Native缓存生命周期。
