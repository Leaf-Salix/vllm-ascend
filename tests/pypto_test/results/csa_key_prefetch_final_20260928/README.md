# 8e176285长B4/B8预取组合定向模型验收

冻结源码`.cache/csa-key-prefetch-final-8e176285`，仅含已提交实现，不含其他会话未提交改动。
新增长双query Key L1/L0B及长三query Key L1-only均已完成各自单卡状态、图重放和计时，
组合CPU完整编译/load通过；[B4依据](../csa_score_key_l1_pair_20260928/README.md)、
[B8依据](../csa_score_key_l1_only_20260928/README.md)。

本轮只补受影响的128K/B4、B8，加8K/B16控制，不重复长B16及短B24/32/40。
以各档本轮Native为基线；长两档内部等权后与短B16按七三计算定向指标，不能称完整七档结果。
最新完整七档仍为[554b3bca](../csa_key_l1_seven_20260928/README.md)，不混拼不同版本。

固定正式W8A8权重/bank、TP1 DP=EP16、DSpark出5验6、mode2/atomic0/det0、EPLB关闭。
两侧保留相同batch capacity40、六档capture sizes；预热8步后连续10步纯decode forward，
另采各档rank0独立三步PyTorch profile。长档PTO→Native、短档Native→PTO。
原有主机/builder观察保留，不新增同步、改GC或扣除慢rank等待。

[模型入口](run_model.sh)只通过auto分配16卡执行；作业内读取四份单卡PASS及组合CPU证据，
不在守护进程内嵌套查询队列。提交前确认前置任务均完成。
[正式收集](collect_model.py)、[入场分析](analyze.py)、[离线导出](export_profiles.sh)、
[模型区间/下载汇集](profile_report.py)复用当前已执行的七档流程。

20:01已提交`task_20260928_200111_19871068850`，auto请求16卡、最长7200秒；
提交前确认七档模型、B4和B8单卡前置任务均已退出0，三个Python入口Ruff、两个shell语法检查通过。
三档任务已完成、退出0。B4 Native/PTO forward均值45.645/43.617ms（−4.44%），
B8为55.147/55.188ms（+0.07%），短B16为65.234/60.865ms（−6.70%）。
共114688个输出token零差异，48组rank DSpark一致；位置及配置门禁通过。
B4 P95 48.736/47.475ms，B8为55.831/56.129ms，短B16为66.362/61.907ms。
长两档内部等权后−2.185%，与短B16按七三为−3.538%；这是定向三档，不是新版完整七档。
[已完成档位及全部正式样本](model/RESULTS.md)。

新增离线主机分析保留持续的rank偏移：设备跨时钟居中法会把稳定的迟到与时钟差一起消去，
不适合据“未发现>2ms异常”判断入场一致。本机monotonic_ns显示B8 PTO rank5/9十步中位
分别晚于同一步rank中位2.231/1.852ms，自身forward中位53.257/53.501ms，rank0为55.412ms。
metadata墙钟/线程CPU分别6.165/4.064和5.771/4.095ms，rank0为4.090/3.908ms；
多个builder都有墙钟与CPU差距，没有定位为单一算子、GC或CPU抢占。
该模式与跨rank等待相符，但不据主机入口差额扣减正式forward或宣布唯一根因。
[持续偏移及逐rank证据](HOST_SPREAD.md)。

复读上一轮554b3bca的同一主机记录，B8 PTO rank5也持续晚1.841ms；
该轮Native/PTO平均forward主机入口跨度0.957/2.151ms，本轮为0.894/2.581ms。
两侧当前rank5/9的CPU绑定计划相同，尚无实际线程调度或锁等待证据，不直接改绑定策略。
原计划除六份rank0，再离线解析已经采集的B8两侧rank5/9，用于定位等待范围。
按用户最新“128K incore及CSA内调度优先、整网后置”，此项暂缓；没有启动profile导出或追加EP16。
上述主机分项仅已分析128K两档，原始三档profile及正式结果完整保留，待后续整网阶段恢复。
