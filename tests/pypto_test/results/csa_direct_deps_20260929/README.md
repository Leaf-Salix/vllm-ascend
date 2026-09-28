# CSA直接任务依赖实验

用户要求去除AICPU上的dummy中转，保留多任务直接依赖。生产基线d3adbe04，算子等同3b27c7fd。
本轮不叠加尚未合入的stream2048候选；两侧均为冻结整包及公共依赖。

## 修改

| 原依赖 | 候选 |
| --- | --- |
| RoPE → dummy → KV / inner Compressor | 消费者直接依赖RoPE |
| RoPE与Q_A → 汇合dummy → Attention Compressor | Compressor显式deps=[rope_tid, qa_tid] |
| 空dummy → Q_B | 传无前置任务哨兵；量化张量自动依赖仍保留 |
| 空dummy → Indexer weights | 传无前置任务哨兵；输入张量自动依赖仍保留 |

`task_invalid()`仅生成TaskId::invalid()，不会提交替代dummy。未移除scope或自动读写依赖，
未改变kernel算术、cache布局、sync_start与allow_early_resolve。
生产/测试根_get_dep_graph()和CPU完整编译/load通过；调度C++的rt_submit_dummy_task从4处变0，
直接RoPE/Q_A边可见，AIC/AIV提交点数量不变。[lowering.json](lowering.json)。

## 对照范围

128K/B16、8K/B24，第二个CSA层layer4正式权重/合成历史，CANN9.2、mode2/atomic0/det0，EPLB关闭。
两侧ring=[256,128,256,32]MiB/task_window4096。每侧5预热/20次无profiler图计时，另采四个DFX窗口。
八类PTO状态精确对照、图A→B→A与保护区检查复用既有入口，不重复整矩阵或整模型。
判定按长短8:2，同时单列P95/max；DFX首窗口核查4→0 dummy和消费者直接依赖。

本轮只判定PTO自身调度变化。Native控制列仍是手工NPUGraph，不作为对齐decode模板后的新基线。
不能把dummy自身处理时间与多线程重叠区间直接相加，声称等量CSA收益。

[候选补丁](candidate.patch)、[冻结路径](source.txt)、[编译入口](compile.py)、
[单卡入口](run_layer.sh)、[设备任务](task.txt)、[收集入口](collect.py)。
任务task_20260929_005243_82229510692已完成exit=0，auto分配单卡。
[完整结果](RESULTS.md)、[原始计时与调度证据](evidence.json)。

## 结论

长B16均值1046.853→1045.684μs（−0.112%），短B24为976.313→975.806μs（−0.052%）；
8:2为−0.100%，没有形成明确可归因的性能收益。P95长档1057.100→1055.840、短档992.400→992.760μs，
长档max1057.480→1072.560μs略升，不能称为长尾修复。

四窗口均值里，长档weights首次接收提前29.745μs，但Q_A启动分散0.470→22.555μs，
Compressor首次接收延后7.515μs，Score首次接收延后13.975μs；稀疏注意力也没有提前。
短档Score提前12.750μs，但其包络和后续核时上升，完整CSA几乎不变。
这些是调度时序与资源竞争的观测，不能从中断言weights单独造成了所有回退。
核函数算术未变，核时变化包含等待与重叠影响，不记作新incore算法收益。

两档八类状态零容差、图A→B→A、metadata/保护区通过；DFX确认dummy 4→0与原前置任务直接边。
本轮保留候选补丁和证据，不合入生产，不为约0.1%的差异扩跑七档或16卡。
后续若调整整组准入或任务优先次序，可基于此无dummy版本继续做定向对照。
