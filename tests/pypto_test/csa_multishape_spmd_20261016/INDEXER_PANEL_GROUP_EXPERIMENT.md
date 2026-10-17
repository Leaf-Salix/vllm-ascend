# Indexer N64两面板循环：未采用

2026-10-17，保留M384/N64/K128 QK与M16/N64/K384 WS，16个展开面板改8组、每组2面板。
组末完成WS，避免Key Right/score跨迭代携带。运行时数学量、FP16 FIX/ReLU、叶分片、
每1024分数交接和AIV Top-K不变。代价是组边界可能打断跨面板重叠。

`indexer_group2_prepare_v1`与`indexer_group2_cpu_v1`通过。
实际分配及生成循环见`indexer_group2_codegen_audit.json`、`indexer_group2_generated_loop.cpp.txt`。
仅group2经过编译和设备测试；group4/8只有源码生成/列覆盖核对。

任务`task_20261017_080841_317060310914`终态exit0，输出、Top-K、六份cache/state、
图、30代信号、保护区均PASS。完整无插桩L1三窗920.58/924.02/918.76μs，
中位920.58μs；同轮原动态PTO中位909.50μs。证据`indexer_group2_result.json`。
没有收益，未采用、不扩档位。现选B16历史807.16μs（不是本任务同进程对照）。

未测到I-cache miss，不能把代码缩短视为已识别的性能收益。
每次调用恢复fixture，不代表连续decode轨迹；未运行Native forward。
