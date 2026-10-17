# 2026-10-17：N80 WS K128分段与Key Right预取

旧N80候选完整精度通过但中位879.10μs，未采用。它的WS Right为
384×80 FP16=60KiB，无法同时容纳10KiB Key，限制了预取。

新候选`indexer_n80_ws128.py`保留QK M384/N80、13个panel、原叶分配、
FP16量化/ReLU、FFTS、AIV TopK、根参数及GM工作区。WS按0/128/256三段累加，
每个Right块降为20KiB，提前把下一Key搬到Right。此变换增加matmul指令数量，
也改变K的指令分段，不能仅凭理论容量与数学等价断言更快或逐bit相同。

开关`--indexer-n80-ws128`默认关闭，首轮只验证128K/B16/S6，未与其他新候选叠加。
prepare/cpu_v1完整通过。实际IR有13个Key Right和39个WS Right：
WS地址为0/20480，后续Key为40960/51200，峰值末端61440。
检查了12组“下一Key extract在当前QK matmul之前”，对应Key地址区间不重叠。
这是实际分配与提交顺序证据，还不是硬件重叠测量。审计在
`indexer_n80_ws128_codegen_audit.json`。

任务`task_20261017_071758_23254658630`已提交，结果目录
`b16_row_deps_indexer_n80_ws128_v1/`。先完整exact/TopK/六状态/图/30代/保护区，
通过后测三窗无插桩L1；同轮对照仅原动态PTO，不运行Native forward。
当前尚无设备结论，未改变现选七档。

## 终态：完整通过，性能不采用

任务exit0，输出/TopK/六状态、eager/图、30代信号与保护区均通过。
单SPMD三窗844.52/855.10/858.08μs，中位855.10；同轮原动态906.78μs。
Event单独见result，不混入L1计时。未做同进程旧N80/现选配对，不能把跨运行
879.10→855.10当作精确收益。该候选仍慢于现选，未采用、不扩七档。
这证明K分段本fixture通过exact；不代表任意输入均已验证，也不是连续decode轨迹。
证据`indexer_n80_ws128_result.json`及原任务目录。
