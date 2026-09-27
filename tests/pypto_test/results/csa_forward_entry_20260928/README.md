# 正式forward入场尾部诊断

30f2b228统一七档中，128K/B8的PTO step11 rank4相对进入异常3.980ms，
128K/B16 step16 rank6异常14.060ms；晚进入rank自身forward正常，其余rank分别增加3.631/13.892ms。
[原样正式样本与相对时钟方法](../csa_qa_matrix_20260928/ARRIVAL.md)。不减掉这些点或修改验收边界。

只补128K/B16，Native先、PTO后；算子、cache、配置均不变，预热8步后10步forward。
新增可选主机时间/线程CPU时间与GC回调，在execute入场、forward入场、forward提交、execute返回各留时间戳；
GC只记时间/代数，不扫描对象、不打印、不修改GC配置、不加入设备同步。
这用于区分CPU侧准备、进程未获调度、GC或设备队列延迟，不预设根因。

CPU回归6项通过：前后处理不计入forward、窗口/缺失forward被拒绝、诊断开关与GC回调均恢复。
EP等待不能在单卡完整复现，因此后续只补受影响的一个16卡档位，不重跑七档。
源码冻结于 `.cache/csa-entry-diagnostic-30f2b228`，与主矩阵/单卡候选隔离。

任务：task_20260928_074528_18466049204，排队中；CPU测试日志见[cpu_test.log](cpu_test.log)。
