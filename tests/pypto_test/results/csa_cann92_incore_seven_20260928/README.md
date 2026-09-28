# CANN 9.2：已保留核内优化的七档单卡收口

状态：task_20260928_224619_23037581103已完成，退出0；auto分配card0，七档检查和采集齐全。
这是当前阶段完整覆盖及Native核内资料补采，不是新候选或EP16验收。

## 冻结范围

- 同一生产实现e33d842a，快照`.cache/csa-cann92-baseline-e33d842a`。
  含Key L1预取、均衡长leaf、2560/3072局部排序、四路UB根、QR/HC已保留优化，以及WO_A ND和arena配置。
  不含已否定的最大长度复用、query排序循环；精度版不在本轮。
- 公共CANN9.2.0-beta.2，PyPTO88f60598、Simplera54c05095、PTOAS0.66、PTO-ISA327cd586。
  Native QLI/Sparse仍为release custom包，本轮不能冒充最新ops-transformer二进制测试。
- 第二个CSA层layer4正式权重，checkpoint DeepSeek-V4-Flash-0731-w8a8；合成历史、反序物理页、逐行变化scale。
  TP1、query6、mode2、atomic0、det0、EPLB关闭；固定seed1024。
- 七档：128K B4/B8/B16；8K B16/B24/B32/B40。一个auto队列任务在同一卡顺序执行。
  每档5次预热、20次无profiler图计时；计时完成后另采Native/PTO PyTorch profile，另进程采四个PTO DFX窗口。
- 检查PTO八类状态自重放、计时图对eager、A→B→A、metadata/保护区、Top-K结构及有限值。
  不含跨版本八类状态精确比较，Native算术差异单列；不代表整模型token/DSpark验收。

## 结果口径

每档完整CSA均值、P50、P95、最大值及20个原始样本；分别报告Native/PTO。
七三变化率先在128K三档与8K四档内等权平均，再按0.7/0.3合成；不混用旧9.0成绩。
核内均值、最慢核、Worker包络、启动分散及同核重复派发分别记录，不能把包络差额叫纯调度时间。
Native的独立kernel profile与PTO四DFX属于不同运行，保留各自测量范围；不把所有并行核时相加。
单层整图时间不能与独立profile kernel直接相减作因果分摊。

## 当前结论与后续

完整CSA七档均值/P95/max均低于同轮Native，七三均值变化−14.161%；128K/B16为
Native1279.55 / PTO1078.85μs，PTO P95为1087.12μs。全部原始样本保留，Native短B32的单个慢样本未剔除。
单卡自重放/图状态/保护区通过；不代表新源码的整网token/DSpark通过，也不代表旧EP16长尾已关闭。

128K Score B4/B8的PTO AIC核时134.262/166.674μs，仍高于Native PMU参考66.855/124.422；
B16 AIC242.765接近Native240.925，但PTO AIV263.622、独立归并10.943仍需优化。
两侧工作范围和测量方法不同，不能直接相减声称可回收相同数量的μs。定义见[METRICS.md](METRICS.md)。
8K B16/B32/B40四窗口中仍观察到Score同核两份；本轮正式P95未异常不等于调度问题已解决。

采集期间主线新增3b27c7fd，将WO_A布局从ND改为跟随Native的NZ29。本矩阵冻结e33d842a，
**不包含该修正**，不得改标签为最新HEAD。先补该修正的长短B16集成/核内对照，再决定后续关键链。
[WO_A两档验证](../csa_wo_a_native_nz_20260928/README.md)。无新证据不重跑两个已否定候选，不追加EP16。

[完整七档及核内表](RESULTS.md)、[全部机器可读数据](matrix.json)、[21份JSON汇集](download/README.md)。

[设备入口](run.sh)、[队列句柄](task.txt)、[主机汇总](collect.py)。
汇总入口要求任务completed(exit=0)，检查七档配置/自重放/保护区及完整profile/四窗口后才生成RESULTS.md和matrix.json。
