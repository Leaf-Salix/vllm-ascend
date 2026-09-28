# 连续长档入场诊断

只重复同进程128K/B4→B8→B16顺序，以复现统一七档中换档后的入场尾部。
算子30f2b228未改；PTO先、Native后，mode2/atomic0/det0/EPLB关、原物理cache。
两侧增加同样的主机/GC观测，正式计时边界与8步预热后连续10步保持；不清GC、不增加同步、不去掉异常点。
容量40，capture 24/48/96/144/192/240，固定权重/bank/budget256。
该复测针对原B8/B16的尾部，B4是复现前置状态所需；不是新七档成绩。
源码复用已冻结的`.cache/csa-entry-diagnostic-30f2b228`。

依据：[统一七档尾部](../csa_qa_matrix_20260928/ARRIVAL.md)、
[直接B16未复现](../csa_forward_entry_20260928/README.md)。

任务task_20260928_081648_96094015347已提交；没有据假设修改生产代码。

PTO侧已完成：B16 step11 rank3设备相对入场异常约3.147ms，主机forward入场相对中位约3.143ms。
其execute入场还早于中位0.809ms，准备墙钟59.778ms/线程CPU55.625ms，forward提交仅0.319ms。
正式B16无GC；B8的约9ms GC发生在execute之间，没有对应的>2ms设备入场异常。
因此目前定位到主机准备区间，但仍不能将CPU时间解释为纯计算（可能含已有设备同步忙等）。
待Native完成后统一收集正式对照；不靠这个诊断点修改生产策略。

为下一次必要模型验证扩展可选主机观测：原有input sync、_prepare_inputs、DP档位协调、
attention metadata、_preprocess各留起止时间，既有同步调用不变。
CPU回归6项通过，设备forward仍100μs的模拟区间，钩子/GC回调均恢复。
这项扩展未放入本轮冻结源码，因此本轮数据不冒充细分观测。[CPU验证](cpu_phase_test.log)。
