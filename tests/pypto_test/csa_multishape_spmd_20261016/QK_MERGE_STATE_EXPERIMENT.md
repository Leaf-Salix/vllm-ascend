# QK/PV循环状态分块（2026-10-17）

> 2026-10-17更正：本次设备产物提前JIT编译，未应用请求的cached_wait/100tick。精度结果保留，不能据此判断数学核改动的性能回退；见[编译上下文修正](COMPILE_CONTEXT_CORRECTION.md)。


**未采用。** B16/128K/S6完整精度通过，但完整无插桩L1中位1033.44μs；同轮原动态PTO为912.20μs。
现选单SPMD此前约807μs，两者不是同轮配对。七档选择保持，不扩档，不重复本候选。

## 改动与编译记录

- 最早16-head及后续8-head版只缩小PV incoming Tile，仍保留32×256整状态并assemble。
  v1/v3/v5均因196608B Vec超过188416B失败；v4仅准备阶段受组合保护拦截。
  失败代码及日志保留，没有扩大硬件容量或改编译器。
- v5 MemoryReuse转储显示四个32768B状态基址和一个65536B共享arena。
  缩小incoming不会自动消除整状态双份分配。证据：`qk_merge_heads_v5_memory_allocation.json`。
- v6把每侧循环状态本身分成两个16×256 FP32值，完整展开初始化、分支/循环yield和reset；
  最终16-head输出直接读取相应状态。每元素old*alpha+incoming*beta及m/l、五块合并次序不变。
  保持所有GM写表达式、scalar metadata读、显式同步和其他函数AST。
- `qk_merge_heads_prepare_v6` / `qk_merge_heads_cpu_v6`完整codegen通过。
  AIV分配65536+32768+32768+32768+32=163872B。
  四个状态块地址65536/81920/98304/114688，每块16384B；初始化/更新/reset地址一致、互不重叠。
  `qk_merge_heads_codegen_audit.py`对CPU及设备实际artifact均通过，设备首次执行前检查。

## 设备结果

任务`task_20261017_103218_149245329068`，exit=0。
正式第4层权重，合成128K历史，固定输入图重放；两侧均PTO，没有Native forward。

| 项目 | 原动态PTO | 单SPMD状态分块 |
|---|---:|---:|
| L1窗口1（μs） | 910.98 | 1033.44 |
| L1窗口2（μs） | 914.34 | 1351.48 |
| L1窗口3（μs） | 912.20 | 1029.72 |
| L1中位（μs） | 912.20 | 1033.44 |
| 图Event中位（μs） | 959.95 | 1118.08 |

输出、Top-K及六份cache/state的eager、图重放、最终结果全部零差异；30代通知、所有未写区域、
metadata与workspace保护通过。每次恢复fixture，不代表连续decode轨迹。
三个窗口全部原样保留，没有删除较慢样本。

## 已核对的生成代码影响

与现选产物比较全部AIC/AIV阶段的静态指令位点，只有QK AIV第30阶段这些计数改变：
PIPE_ALL 9→8、set/wait各80→82、TLOAD 22→24、row-expand-mul 4→8。
其他阶段这些计数没有变化；这不是所有指令或地址完全相同的证明，也不是回退的定量归因。
PV内部局部V屏障9→13、set/wait各6→10，完整流水并未因为UB缩小就自然提速。
没有为负结果追加探针或复测。

## 文件

- 实现：`qk_merge_heads.py`；旧8-head源码：`source_snapshots/qk_merge_heads_8_full_state_v5.py`。
- 生成代码审计：`qk_merge_heads_codegen_audit_v6.json`及同名merge.cpp.txt。
- 全阶段静态比较：`qk_merge_heads_full_codegen_comparison_v6.json`。
- 完整结果/泳道：`b16_row_deps_qk_merge_state_v6/`；摘要：`qk_merge_heads_result_v6.json`。
- 大产物只保留本地，过程镜像中的LOCAL_ARTIFACTS.json记录路径，不做hash。

## 修正等待配置后的v7

`task_20261017_110518_211373122343`完成exit0，全门槛PASS，实际AIC/AIV 100tick等待门禁通过。
完整L1三窗820.58/812.90/820.24μs，中位820.24；同轮动态909.56。
没有证明优于现选807.16/807.50，未采用；没有同进程现选配对，不给出精确回退比例。
机器记录qk_merge_heads_result_v7.json。上文v6数值保留，但其性能归因已由编译上下文更正撤回。
