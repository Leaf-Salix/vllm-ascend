# 固定 K 规约下按 M 行分组 KV 投影：独立候选

基底45412ba3（核内71153bb3），工作树`.cache/csa-kv-mgroups-45412ba3`。
不包含正在EP16验证的免seed候选；原清零、QR、Indexer、Sparse和调度标志均保持。
本轮七档DFX中8K/B16的KV投影仅4个Cube block，每block均值54.77μs；
历史上游对应8个block、29.18μs，但旧图配置不完整，只作方向参考。

pypto-lib main2164563已有KV split-M框架；原规则要每个组至少容纳两个M64块，T96只有一组。
当前部署atomic0又没有split-K，并行度变成仅N维4块。
候选在atomic0时改为M32，并按完整M32块数启用最多3个M组；T96为3组×4个N块。
每个输出仍由一个核完整执行K4096的K256顺序累加，不需要atomic或跨核归并。
atomic1保留原M64及组数规则；精度版不改。Native QA已有无跨核split-K的M/N分工，
但此处不声称完全复刻Native KV tiling，需用本模型实测判断。

代价是不同M组重复读取权重，增加总搬运，必须同时报告block数量、核内均值/累计核时间和本体/P95，
不能只因为单block处理行数减少就宣称整体优化。
具体到本轮T96，旧M64+两个M16尾块、新三个M32都各需三次完整K权重遍历，总读取轮数相同；
T48从三次降到两次，但T144/192/240分别由3/3/6变为5/6/8次。
因此该先导不能直接说明大batch也应该使用M32；若T96先导有效，下一步在算子内部选择M32/M64，
大档先保留M64并只增加M组，避免不必要的重复读取，再按受影响项补测。
有效dense行和tail行仍由原两段循环互斥覆盖，K顺序和下游量化没有改；逐元素一致必须实测。

[补丁](candidate.patch)、[单卡入口](run_layer.sh)。先128K/B16与8K/B16两档，mode2/atomic0/det1，
layer4真实权重、可变物理行scale的人工历史。双方20次独立图计时、8类状态精确比较、候选A→B→A；
核内另外各4个DFX窗口，不与计时混采。未通过前不合入，也不启动该候选整网测试。

完整CPU lowering/PTOAS/CCE/链接通过，编译期间8K模型仍在初始化；未改编译器。
`task_20260928_060059_4594592998`已排队，在免seed两档EP16任务之后使用一张卡。

本轮只读已有七档Native profile，按QA归一化消费者和辅助stream的MatMul顺序匹配，
441个CSA区间均通过计数与stream检查。8K/B16第4层Native KV三步均值22.01μs，
当前PTO KV单窗口block均值54.77μs；128K/B16分别13.89/45.67μs。
Native是完整kernel，PTO是单block，输入/轮次也不同，不计算直接加速比。
[七档投影对照与映射依据](../csa_atomic_matrix_20260928/NATIVE_PROJECTION.md)。
