# 复用已有最大cache长度：CANN 9.2下的Indexer候选

两侧统一冻结`e33d842a`，使用已切换的公共CANN 9.2.0-beta.2环境；
不再用8e176285旧快照对照当前WO_A ND/arena改动，也不拼接9.0历史计时。
基线`.cache/csa-cann92-baseline-e33d842a`，候选`.cache/csa-maxlen-reuse-e33d842a`。
只修改候选性能版`decode_indexer.py`，尚未保留到生产。

## 来源与待解决差异

ops-transformer 28f40354的A3 QLI V2 `SplitCore()`从metadata读取每核分片起止信息，
`GetS1S2ActualSeqLen()`读取当前请求长度；以预计算计划减少每核重复规划是可参考方向。
同文件仍定义扫描batch的`GetTotalBaseBlockNum()`，不能据辅助函数存在或某一入口就声称Native所有路径都不扫描。
当前PTO已有编排扫描，用于长短Score分支，却在Score所有worker和长merge所有worker再扫描一遍，
末尾编排也再次扫描同一组只读长度。上一轮生成C++已确认核内扫描未被自动移除。

候选复用首轮`max_topk_cache_len`：给五种Cube Score策略和merge传现有INDEX标量，
删除两处核内全batch扫描及第二次编排扫描。每个query仍按自身长度和position裁剪可见范围。
Native adapter已要求tokens=batch×6和五组request容量一致，原两个最大值的扫描范围相同；
没有把batch最大长度当作每个query自己的可见长度。

未改变Score算术、叶片分配、排序规则、cache布局、任务数或调度标志；精度版保持。
与pypto-lib的差异仍是直接使用Native分页cache及已有query分组，复用当前编排能力，
不引入新metadata张量、executor或CPU内容读取。参数传递也有成本，收益必须看设备核时而非仅按少读估算。

## 必要验证

1. 新9.2环境完整CSA两侧编译/load；核对生成码的核内扫描消失且标量由当次编排传入。
2. 小型[图探针](length_graph_probe.py)调用真正候选merge，同地址长度32769→16385→32769，
   对应12→6→12半leaf根；预置有序且分数唯一的root，逐元素检查输出和保护行。
   用于排除图捕获把长度标量固定住，不代替完整Score或模型精度。
3. 128K/B16＋8K/B16完整层对照：八类PTO状态零容差、保护区/metadata、A→B→A，
   每侧20次无profiler计时及四个独立DFX窗口，分别记录incore、包络、CSA/P95。
4. 只有这些必要检查完成后才判断保留；长短七三权重，不以一项均值掩盖异常P95。
   暂不追加EP16或全七档；阶段性策略影响范围明确后再补缺口。

[补丁](candidate.patch)、[CPU入口](compile_all.sh)、[单卡入口](run_layer.sh)。
目前代码和静态检查完成，CPU编译待已运行的16卡任务`task_20260928_215247_8667695589`结束后开始。
由现有主机guard观察具体队列句柄，避免与其正式计时争用CPU；尚无本候选设备收益结论。
