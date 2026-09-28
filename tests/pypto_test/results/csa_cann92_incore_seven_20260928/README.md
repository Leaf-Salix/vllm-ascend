# CANN 9.2：已保留核内优化的七档单卡收口

状态：task_20260928_224619_23037581103已提交，结果尚未完成；不提前填性能或正确性结论。
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

结果出来后更新七档Native/PTO差距，按128K优先选下一项调度改动，8K及异常P95作为约束。
没有新证据不重跑已否定的两个候选，也不把这个矩阵称为新七档整模型结论。

[设备入口](run.sh)、[队列句柄](task.txt)、[主机汇总](collect.py)。
汇总入口要求任务completed(exit=0)，检查七档配置/自重放/保护区及完整profile/四窗口后才生成RESULTS.md和matrix.json。
