# 过程103：固定链AICPU派发与AICore启动关联，已编译并排队

2026-10-12。只诊断同一17×48任务链，5/50μs两档各一次，保持256轮Scalar、原数据计算和依赖，
保留early_start、原派发/ACK/选核/early-resolve。没有启用会串行回退的chip swimlane、PMU或参数dump。
此程序不是正式计时用例，不用于宣称min/mean/max改善。

每组trace声明为INT64[96,8]：前48行由AICore的既有trace写入，后48行由AICPU按逻辑block写入。
两侧使用不同cache line；CPU行保存准备开始/结束、原寄存器写操作前/后、context、代次、逻辑核和标志。
AICore行保留物理核、前缀入口、函数返回、最终尾部记录等信息。按stage/block一一关联，
再检查逻辑核到物理核的映射一致，避免假定两侧核号相等。另保留一整行trace保护区，计算输出仍逐元素精确检查。

新增读钟、记录和cache clean会扰动派发；发布后采样包含wmb，仍不等于核上寄存器可见时刻。
分析将区分CPU准备、批量准备后等发布、发布到函数入口的间隔，且不把这些诊断时间替代正式性能。

## 构建及状态

运行时基于保留实现1c56bcd3e另存到本次目录，不替换活动SDK。
生成代码仍使用已冻结的3d0b9c820 ABI，runtime对外布局未改变；设备兼容性仍待实测，不能由离线构建代替。
完整补丁包含该保留实现与诊断；脚本构建时从私有Git对象取保留实现，临时应用后恢复活动源码。

离线构建已通过，Python检查通过。初次C++构建的整数窄化错误已修正；
两次离线编译的SDK提交号不一致已通过恢复冻结SDK解决，原失败日志保留。未因此占卡或重提原队列任务。

任务`task_20261012_140105_357792024852`当前pending，单卡，尚无设备记录或性能结论。
私有输出：`artifacts/boundary_dispatch_trace_20261012/`。
[完整补丁](candidate.patch)、[仅插桩补丁](instrumentation_only.patch)、[驱动](repro/task_boundary_dispatch_trace.py)、[排队脚本](repro/run.sh)。

驱动放到私有PyPTO的`tests/st/runtime/kernel/`，依赖既有`task_boundary_acceptance.py`、
`task_boundary_chain_trace.py`和`task_boundary_normal_state_probe.py`；归档不是独立安装包。
构建、编译分别用`build/compile --root <本次目录>`，设备阶段用归档run.sh，经task-submit分配TASK_DEVICE。
不要使用这个带插桩的runtime做正式性能对比，也不要重编已有的冻结对照产物。
