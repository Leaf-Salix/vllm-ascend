# B16 向量索引路径的提前 KV 通知（2026-10-17）

完整验证通过，但未观察到优于现选的性能，不采用、不扩档、不重复测量。

生产`EARLY_KV_MIN_TOKENS=144`有旧实测依据。本次不是修正所谓错误阈值，而是核对
新增向量索引路径后，T96是否适合已有的提前通知策略。
`qk_early_kv.py`仅把三个条件中的144改为96，恢复条件后整个AST必须与原始一致。
保留数学、三槽、pre-launch=2、全部FFTS事件和GM访问；开关默认关闭。

`qk_early_kv_prepare_v1/`与`qk_early_kv_cpu_v1/`完整CPU编译PASS。
集成`--qk-early-kv`后的`qk_early_kv_integrated_v1/`入口文本与已编译源码完全一致，
没有重复CPU编译。`qk_early_kv_codegen_audit.py`在所有编译扩展启用后检查实际artifact，
同时保留100tick等待策略源码门禁。

真实AIV C++阶段内顺序：1192等待前块Score → 1203通知当前KV → 1330 row-max →
1357 row-sum → 1424通知Prob → 1464等待PV。两个wait、两个notify位点和9个PIPE_ALL
与现选一致。静态位点/顺序本身不证明运行时重叠或性能收益。

任务`task_20261017_114240_273482232688`，目录`b16_row_deps_qk_early_kv_v1/`，exit0。
正式权重、128K/B16/S6、PTO对PTO；eager/graph/final的输出、TopK和六cache/state均零差异，
30代进度通知、输入/写入范围/工作区保护区全部PASS。每次恢复fixture，不是连续decode轨迹。

| 实现 | 完整无插桩L1三窗（μs） | 中位数（μs） |
|---|---|---:|
| 本次提前通知单SPMD | 823.38 / 816.34 / 833.82 | 823.38 |
| 同轮原动态PTO | 905.06 / 907.42 / 900.04 | 905.06 |

现选B16历史807.16μs，最近另一轮同进程现选807.50μs。本次没有加入现选SPMD配对，
不能把跨轮约16μs差额精确归因，但没有理由用该候选替换现选。
只计完整incore首尾，保留等待/空隙，排除外围调度。摘要`qk_early_kv_result_v1.json`。
