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

任务：task_20260928_074528_18466049204，退出0；CPU测试日志见[cpu_test.log](cpu_test.log)。

本次128K/B16直接入场：Native/PTO均值72.943/69.519ms（−4.69%），P95为73.859/70.297ms；
逐步最慢rank均值73.430/69.735ms（−5.03%），10/10步更快。
65536输出token零差异，16组DSpark与请求位置一致。
[正式结果](model/RESULTS.md)、[主机与GC观测](HOST.md)、[相对设备入场](ARRIVAL.md)。

两侧Worker均未冻结GC堆，但正式10步之间未出现GC，也未出现大于2ms的设备入场异常。
约15～21ms的GC均在预热期，不能据此确定旧长尾的根因。
生产算子未改，不宣称长尾已解决，不用本次结果覆盖统一七档中的异常。

本次直接运行B16；旧异常来自同进程B4→B8→B16连续换档。下一步仅复现该长档顺序，
两侧相同启用上述观测，保留正式10步与独立profile，不扩到短档或完整七档。
上游vLLM GPU Worker在compile_or_warm_up_model末尾调用freeze_gc_heap，当前Ascend Worker没有；
这是只读源码差异，尚无正式窗口GC证据，因此不修改GC策略。
