# AICPU异常内部状态诊断：本轮未复现，根因未定位

2026-10-12。确认轮缓存版对照返回507018，CANN报告retCode=0x2a；现有设备日志没有明确的
Simpler heap、fanin或stall错误，不能据此判断死锁或归因于另一进程的本核初始化候选。
失败对照的原二进制复查 `task_20261012_131102_174586322769` 仍在队列，不重复提交。

当前公共入口将consume_kernel_task的非零内部状态统一转为CANN返回值2，丢失分类。
准备的隔离诊断版只在status非零时输出BOUNDARY_DISPATCH_FAILURE logical_status，
正常返回保持，不新增运行行为开关。它用于区分InvalidArgs、InvalidBinding、ExecutionFailed等返回，
不能覆盖异常退出导致根本没有返回的路径，也不把CANN的0x2a当作同一个枚举值。
SDK编译通过，尚未提交设备任务；仅在原二进制复查仍失败且需要更多信息时运行。
[补丁](candidate.patch)、[构建记录](build.log)、[准备脚本](repro/run.sh)。
运行需同目录放置过程097的复现依赖；计时数据仅作诊断，不纳入正常性能结论。

## 设备复查

缓存097对照在160613卡9的换输入重放再次异常后，使用已构建的错误返回诊断版本，
任务`task_20261012_160821_17859016145`卡8退出0；5/50μs、1/17级、plain/early各组合精确通过。
此次只开ERROR日志，没有打印BOUNDARY_DISPATCH_FAILURE，只能说明本次未复现，不能宣称根因修复。
不把这轮诊断数据与另一张卡的失败对照混成性能夹测。
[错误诊断完整输出](device/error_only/)、[运行脚本](repro/run_error_only.sh)。
