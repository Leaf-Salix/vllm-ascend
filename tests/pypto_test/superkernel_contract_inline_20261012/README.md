# 过程098：缓存交接初始化内联，设备结果待回收

2026-10-12。在过程097上只将 initialize_overlap_context 强制内联，消除每个标记 block
的辅助调用。保持分支判定、字段值和原派发语义；没有改变 AICore、算子计算量或计时方式。
19项CPU检查通过，完整SDK隔离编译成功，源码已恢复。

任务 `task_20261012_125133_105743132639` 排队中，尚无性能结论。
为区分内联效果与设备漂移，同卡顺序为旧SDK→缓存版→内联版→缓存版→旧SDK；
各候选内普通/标记两路轮换，仍测试固定的1/17级和5/50μs负载。
AICore和字段语义与097一致，复用CPU交接检查；不额外重复三项设备诊断。

| CPU机器码 | 缓存版 | 内联版 |
| --- | ---: | ---: |
| prepare_subtask_to_core 字节 | 1136 | 1268 |
| initialize_overlap_context 字节 | 180 | 无独立函数 |
| build_payload 外部入口字节 | 284 | 284 |

热函数本身变长，所以不预判净收益。保留与否只看设备对照。
[候选补丁](candidate.patch)、[CPU结果](cpu_results.xml)、[CPU日志](cpu.log)、
[构建日志](build.log)、[复现脚本](repro/run.sh)、[提交记录](submission.json)。
