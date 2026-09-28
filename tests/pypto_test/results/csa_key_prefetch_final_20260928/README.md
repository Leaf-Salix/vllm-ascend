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
尚无本轮设备结果；不得把单卡核内收益直接写成组合模型收益。
