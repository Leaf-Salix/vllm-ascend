# 复用已有最大cache长度：CANN 9.2下的Indexer候选

两侧统一冻结`e33d842a`，使用已切换的公共CANN 9.2.0-beta.2环境；
不再用8e176285旧快照对照当前WO_A ND/arena改动，也不拼接9.0历史计时。
基线`.cache/csa-cann92-baseline-e33d842a`，候选`.cache/csa-maxlen-reuse-e33d842a`。
只修改候选性能版`decode_indexer.py`。两档检查通过，但没有明确核内收益，**本候选不合入生产，不扩测**。

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
CPU编译在既有16卡任务`task_20260928_215247_8667695589`结束后执行，guard记录见[compile_guard.jsonl](compile_guard.jsonl)。
两侧完整CSA及小探针lowering/PTOAS/CCE/链接/load通过，见[compile.log](compile.log)。
PyPTO为88f60598、Simpler为a54c05095、PTOAS为0.66、PTO-ISA为327cd586；两侧使用相同工具链。

## 完成结果与取舍

`task_20260928_221801_183162810930`在card9完成退出0；20次无profiler图计时、预热5次，
另采每侧四个DFX窗口。长档先候选后基线，短档反向；正式第2个CSA层（layer4）权重、合成历史，
mode2/atomic0/det0、seed1024、出5验6、EPLB关闭。没有重新开展EP16或七档测试。

两档PTO八类状态零容差、metadata/保护区和A→B→A均通过，八类状态不包含idx_topk_scores。
[真实merge长度变化探针](length_probe/report.json)同地址32769→16385→32769，对应12→6→12半leaf根，
每次输出分数、索引和保护行精确通过；这证明该标量随图重放更新，不代替完整Score或模型精度验收。
[生成码记录](codegen.json)及代表性S6/长merge人工检查确认核内全batch扫描消失、编排扫描2→1，
7处标量传参、75个生成函数列表保持。少读GM不等于可见核内加速。

每格为基线→候选，单位μs；全部无profiler样本及四窗口保留在各档evidence.json。

| 档位 | CSA均值 | 变化 | P95 | 最大值 | Native控制均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 1090.455→1085.091 | −0.492% | 1098.52→1099.66 | 1108.94→1108.20 | 1291.503→1273.579 |
| 8K/B16 | 809.297→802.132 | −0.885% | 831.10→816.38 | 837.80→830.72 | 911.110→907.007 |

CSA七三变化−0.610%；未按Native控制漂移归一化，不将低于1%的整层均值变化断言为标量复用的稳定收益。
长档P95增加1.14μs、最大值略降；本轮没有明显异常尾部，不代替旧EP16长尾问题的解决证据。

| 档位 | Score AIC均值 | Score AIV均值 | merge均值 | Score AIV包络 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B16 | 244.088→244.295 | 265.101→265.136 | 10.337→11.480 | 274.545→274.545 |
| 8K/B16 | 38.093→41.245 | 42.582→45.673 | 9.042→8.577 | 60.865→63.420 |

长档Score四窗口分布重叠，AIC/AIV分别+0.085%/+0.013%，merge+11.1%；
短档AIC/AIV分别+8.276%/+7.259%。核内七三分别+2.542%/+2.187%，没有达到减少核内耗时的目标。
独立DFX不能与无profiler某个快慢样本一一对应；参数传递、编排和缓存影响尚未分离，
不把短档变慢直接归因某条指令。Sparse算术未改，其核内波动不记作候选收益或回退的确定原因。
16个DFX窗口的Score/Sparse均每核一份，没有该轮同核重复分配；不证明所有场景调度已稳定。

因此保留现有生产实现，记录这个反例；无新依据不再扩测该候选。
本轮同时补齐e33d842a组合在9.2上的长短PTO编译/图重放检查，包含WO_A ND和arena配置，
但不覆盖HCA设备或新整模型token/DSpark验收。

核内最慢任务、包络及所有窗口见[INCORE.md](INCORE.md)、[incore.json](incore.json)。
[长档状态/计时/泳道路径](h131072_b16/evidence.json)、[短档状态/计时/泳道路径](h8192_b16/evidence.json)、
[七三汇总](weighted.json)、[主机汇总入口](summarize.sh)。
