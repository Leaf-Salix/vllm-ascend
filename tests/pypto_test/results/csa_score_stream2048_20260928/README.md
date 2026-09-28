# Indexer：2048候选分段排序与Cube重叠

状态：两侧排序探针及两档CSA状态/图检查通过，计时与四DFX窗口已收齐。
task_20260929_002109_36305829117完成exit=0，auto分配card0；[同轮结果](RESULTS.md)。
长B16 CSA−3.066%、短B24+0.452%，8:2−2.362%；长Score AIC+3.965%、AIV−1.973%，
不能称为全面核内改善。本轮候选证据保留；2026-09-29已与[UB中间根](../csa_stream_root_ub_20260929/README.md)组合纳入性能版，尚未完成受影响全档或整网验收。
基线19d93a5b（生产等同3b27c7fd），两侧CANN9.2、mode2/atomic0/det0、正式layer4权重/合成历史。

## Native依据及当前差异

最新ops-transformer28f40354 A3 QLI V2的ProcessVec1每段最多2048候选，立即排序并累计根，
随后继续处理下段。该源码BASE_TOPK=2048，内部保留2048对，最后按sparseCount输出；
PTO按当前模型需求保留Top-512，借鉴分段时序，不照搬Native缓冲大小。QK/WS在Cube侧可继续推进。当前PTO在AIV接收完一整个half-leaf后统一排序，
2560/3072/4096候选的排序集中在最后一次Score_READY之后，不能与同leaf后续Cube流水重叠。

本候选仅长档（balance_leaves=True）：收到第四个512候选块就先排序前2048并发布临时Top-512，
剩余分数继续按原缓冲接收；结束后只排尾段512/1024/2048，与先前根合并。
保持后2048段优先的tie顺序、候选集合、量化/规约、分页cache、24MIX任务及调度参数。
短档Score路径不改，通过8K/B24检查完整区间及P95。

这是核内流水候选，尚未等同Native完整UB驻留：现有score GM中转保留，临时Top-512额外写读一次GM，
用于单独判断排序前移是否产生收益。历史§236整块4096 Score UB驻留因物理tile/gather开销失败，
本次没有重试该内存布局；不能把数据流简化或源码顺序当成已验证的加速。
pypto-lib原Score/TopK函数边界不同，本候选源自AscendC融合流水，非机械照抄上游PTO调度。

## 必要验证

两侧使用同名私有pkg、不同冻结工作树，公共依赖同时冻结；排队前解析生产/测试根_get_dep_graph()并编译/load。
先独立十个排序边界/tie case精确比较原算法和值/索引参考，再128K/B16与8K/B24同卡：
八类状态零容差、A→B→A、metadata/保护区，各5预热/20无profiler图计时及四DFX窗口。
按2026-09-29用户新口径，长短8:2分别算Score核时与完整CSA；P95/max单列。无明确核内收益则撤回，不扩测整矩阵或EP16。
若有收益，再按受影响分组补长B4/B8/B24，阶段出口覆盖新七档。

首次task_20260929_000225_403316119769完成两侧十个排序边界/tie用例，值、索引和保护区全部通过；
随后runner误将队列追加的`--device`当作case参数，在启动CSA前argparse失败，exit=2。
修正runner为固定case数组，续跑复用已通过探针，不重复占卡测试。两份task分别保留，不能将exit=2称为整任务通过。
续跑八类状态零容差、A→B→A、metadata/保护区通过；不是Native或整模型token/DSpark验收。

[源码差异](candidate.patch)、[CPU准备](compile.py)、[边界/tie](sort_case.py)、[单卡入口](run_layer.sh)、[冻结路径](source.txt)。

编译生成代码确认：长S6 AIV接收循环中的第四步出现前2048排序，末尾保留尾段排序及两根归并。
两侧都使用同名pkg:dsv4_csa_stream2048_19d93a5b、不同冻结工作树；源码在排队后不再改动。
ring heap=[256,128,256,32]MiB/task_window4096，两侧相同。
[基线编译证明](compile_baseline.json)、[候选编译证明](compile_candidate.json)、[设备任务](task.txt)、
[CPU汇总入口](collect.py)。本轮跨入2026-09-29，按最新8:2取舍；不是旧7:3结论。
