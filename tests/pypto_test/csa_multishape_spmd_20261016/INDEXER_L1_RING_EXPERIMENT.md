# Indexer N64 L1 循环流水实验（2026-10-17）

> 2026-10-17更正：本次设备产物提前JIT编译，未应用请求的cached_wait/100tick。精度结果保留，不能据此判断数学核改动的性能回退；见[编译上下文修正](COMPILE_CONTEXT_CORRECTION.md)。


目标：在原单 SPMD 的 Indexer AIC 内减少长展开体，保持原 M384/N64/K128
INT8 QK、FP16 WS K384、FIX 量化 2^-10 与 ReLU，以及原叶计划、1024 分数交接、
AIV Top-K 和任务依赖。只有 `indexer_score_topk_native_pair_aic` 的面板计算体改变。
以下版本都没有改生产代码、冻结的静态目录或依赖仓库。

## 尝试和边界

| 版本 | 写法 | CPU 结果 | 是否允许上卡 |
|---|---|---|---|
| v1 | Key 三槽、score 双槽；循环最后一轮跳过预取 | 同地址 188416 的 Mat→Mat 移动无法生成 | 否 |
| v2 | 剥离最后一轮，循环 1..14 总是预取下一 Key | 完整 codegen 通过，但物理地址错误 | 否 |
| v3 | 两个 Native 页在同一个分支叶子中加载 | PTOAS 拒绝不支持的 Mat→Mat 地址空间组合 | 否 |
| v4 | Key 每轮局部创建，只让 score 双槽跨循环 | 完整编译与物理审计通过 | 完整精度通过，性能变慢，不采用 |

v1～v3 都不是设备精度失败，也没有性能数据；没有把编译通过当作可运行依据。
旧源码、准备产物、passes_dump 和错误均保留。独立的小 FP16 原位循环探针成功，
不能证明包含 Native 页分支与循环携带值的完整 CSA 也正确。

## v2 的实际错误

`indexer_l1_ring_cpu_v2/kernel/passes_dump/38_after_AllocateMemoryAddr.py`
和 `kernel/ptoas/csa_static_group.pto` 已显示不一致：

- 循环携带的 Key 池以及后续 Key 提取使用 Mat 地址 **188416**。
- 第一个 Native 页加载后，中间 `stage_16_v71__phi_v17` 被分到 Mat 地址 **0**。
- 第二个 Native 页的 `pto.subview` / `pto.tload` 实际写入这个地址 0 的池。
- 最终合流值重新指向 188416；中间没有把第二页搬到该地址的有效操作。

因此，两页中的后半页不会按源程序含义进入下一次 QK 的输入池。
这是物理数据流错误，不是性能波动。`indexer_l1_ring_v2_misaligned_pool_evidence.pto.txt`
保留了关键分配与使用点；`indexer_l1_ring_v2_rejected_audit.json` 记录拒绝结果。
原审计脚本在 `source_snapshots/indexer_l1_ring_codegen_audit_v2.py`。

只读检查当前隔离 PyPTO 的 `src/ir/op/tile_ops/paged_gather.cpp` 可见，
`tile.gather_row` 已声明 `ReadWrite` 与 `output_reuses_input(0)`。
错误在 PTOAS 前的 PyPTO 分配产物已经出现；本次尚未进一步确定到具体的
MemoryReuse / 分支合流修复代码行，不能宣称已修复编译器。

## v4 的取舍

首块生成 QK0 并将其 FIX 到 score 槽 0；循环 p=1..15 中加载本轮 Key，
执行 QK(p) 与 WS(p-1)，将 QK(p) 写入槽 p%2；循环结束后排空 WS15。
Key 和 Right 均不携带到下一迭代，只有 score 池采用 DPS 原位循环更新。

源模型检查确认 32 个 Native 页、16 个 QK/WS 面板和 1024 列输出各出现一次。
记录在 `indexer_l1_ring_slot_contract_v4.json`。这仍不是设备正确性的证明。
取消 Key 提前预取可能带来代价，必须由完整无插桩 L1 首尾决定是否采用。

完整编译命令使用冻结工具 `../csa_static_spmd_feasibility_20261015/compile_jit_entry.py`，
输入为各版 `indexer_l1_ring_prepare_v*/variants/h131072_b16/entry_jit.py`，
参数 `--memory-planner PYPTO --kernel-mode`，输出保存在对应 `indexer_l1_ring_cpu_v*`。
物理审计还要核对 Mat/Right 地址、Acc 不重叠、FIX 量化与 ReLU，以及循环指令。

当前七档策略不因这些候选而改变；128K/B16/S6 的 700 μs 目标尚未达到。

## v4 物理审计与设备任务

Key 两个局部池均分配在 Mat 163840，分支合流地址一致；score 双槽分配在65536。
循环 QK/WS Acc 分别0/98304，Key/WS Right 分别49152/0。实际循环包含两个
TMATMUL、一个带2^-10/ReLU的TINSERT、一次TSTORE，没有Mat→Mat或Right→Right，
也没有PIPE_ALL。完整审计和循环C++见 `indexer_l1_ring_codegen_audit_v4*`。

已提交 `task_20261017_094749_7744056502`，结果目录 `b16_row_deps_indexer_l1_ring_v4`。
新增设备驱动开关 `--indexer-l1-ring`；驱动在执行前对真实artifact再次执行地址审计。
终态为 exit=0；完整精度与图重放通过，性能结果如下，现选策略不变。

## 缩减探针的范围

`probe_loop_gather_phi.py` 保留循环携带、两次条件gather与前后Key读取，CPU完整编译通过。
`probe_loop_gather_phi_cpu_v1/allocation_audit.json` 显示各池与视图均为地址0，
未复现完整CSA v2的不一致。完整CSA的外层分支与全图存储复用可能是触发条件，
目前仅为待定位范围，不能据此确定根因。缩减探针没有设备调用。

## v4 设备结果（不采用）

完整输出、Top-K、六份cache/state的eager/图/最终比较逐bit一致，30代信号、
未写区与workspace保护通过。正式第4层权重；每次恢复fixture，不代表连续decode轨迹。

| 指标 | 原动态 PTO | L1 score环 v4 |
|---|---:|---:|
| 完整L1三窗 μs | 905.34 / 908.00 / 899.98 | 984.56 / 1034.82 / 1008.86 |
| 完整L1中位 μs | 905.34 | 1008.86 |
| 图重放Event中位 μs | 951.86 | 1133.55 |

候选明显退步，不采用、不扩档、不重复测；现选历史807.16μs，最新配对测量807.50μs
属于此前不同任务，不能写成这轮同进程对照。结果见 `indexer_l1_ring_result_v4.json`。
缺少候选内部阶段插桩，不能把全部退步定量归因于Key预取取消；无需为已否决版本加测。

## 修正等待配置后的v5

`task_20261017_110544_21223701219`完成exit0，全门槛PASS，实际AIC/AIV 100tick等待门禁通过。
完整L1三窗804.14/813.50/806.38μs，中位806.38；同轮动态885.92。
与此前现选807.16/807.50接近，未证明稳定收益，不替换七档选择。
机器记录indexer_l1_ring_result_v5.json。旧v4精度结果保留，性能退步归因撤回。
