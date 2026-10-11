# 无直接数据依赖的本核 Scalar 交接验证

2026-10-11，CANN 9.2/TMR，仅单卡。Simpler `5c8418a02`、PyPTO `fdac65f66`；
PTOAS维持`4a5ce22a`。本轮删除交接资格中的直接WAIT生产者限制，未修改CSA算术或重测CSA性能。

构造数据图`AIC A → AIV B`，另一组AIV X与A、B无数据依赖。
核分配仍由原调度器决定。两轮改变输入后，均有48/48个B的安全Scalar与本核X的尾部重叠；
全部前缀在A的计算完成前结束，全部B在A真实完成后才越过Wait。输出精确一致、保护区不变。

- [摘要与版本](summary.json)
- [两轮原始时序](remote_dependency_results.json)
- [Simpler修改、C++回归和协议说明](simpler.patch)
- [10项调度单测](scheduler_ut.xml)
- [跨核依赖单卡结果](remote_functional.xml)与[任务日志](remote_task.txt)
- [首次任务结果](initial_functional.xml)与[日志](initial_task.txt)：既有3项回归通过；
  新用例因单block SPMD遗漏block索引而解析失败，修正后单独重跑通过。

时序单位为原始device timer tick；字段定义沿用局部交接测试。
前驱字段为core/start/returned/completed/event_id/wait_start/wait_ticks/compute_completed，
消费者字段为core/prefix_start/prefix_end/after_wait/scalar_result及三个未写保护槽。
48/48是该构造用例的命中数，不能代表CSA命中率。

测试源码位于上述PyPTO提交的
`tests/st/runtime/scheduling/test_local_task_boundary_remote_dependency.py`。
[设备脚本](run_remote_validation.sh)和[重建脚本](build_private_runtime.sh)为私有工作区
`artifacts/`下的原样副本，其父目录须包含私有`env.sh`、`pypto`、`simpler`，不能直接在本归档目录执行。

归档文本日志仅清理行尾空白；补丁采用零上下文格式，应用时使用`git apply --unidiff-zero`。
