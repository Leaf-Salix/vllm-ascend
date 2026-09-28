# Native / PTO 核时的可比范围

本轮统一CANN9.2、相同输入和同一卡，仍需区分计时边界。公式已核对本机同版本profiler实现及Simpler a54c05095。
本文只解析已有源码/采样，没有新增设备试验。

## Native profile

[CANN官方op_summary定义](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0067.html)
区分Task Duration、aicore_time/aiv_time及各流水线指令时间。

- `Duration(us)`来自设备任务的完整开始/结束边界，包含向计算单元调度、执行和结束响应。
  不能把它当成纯核内计算时间。
- `aicore_time(us)`、`aiv_time(us)`由总周期、频率、block数和可用core数计算；
  是假设同波同时启动、工作均衡后的理论执行时间，不是最慢核，也不是只统计Cube/Vector算术。
- `total_cycles`为所有block执行周期之和。若block数等于core数，折算后的时间就是平均每block执行时间；
  多波时还乘以波次数。不能不看block数便和PTO某一个worker直接相减。

本机9.2 profiler的链路：

1. `analysis/common_func/msvp_common.py:add_aicore_units`将`aic_total_time/aiv_total_time`映射到这两个导出字段。
2. `analysis/mscalculate/stars/ffts_pmu_calculator.py:calculate_total_time`取得当前频率、主/从core的block数和core数。
3. `analysis/common_func/utils.py:cal_total_time`计算
   `total_cycles / freq_MHz / block_num × ceil(block_num/core_num)`，结果单位μs。

这些文件位于公共入口使用的
`/data/pyptouser/qinchuanyu/pto-eager/.cache/dsv4-toolchain/cann92-profiler/tools/profiler/profiler_tool/`。
官方文档同时指出频率变化会影响指标准确性；本轮保留原始周期与block数，不用未经测量的固定频率自行修正结果。

本轮128K三档QLI均为24个主block、48个从block。原始周期和时长全部保留在matrix.json/profile csv。
例如B4的Native QLI任务Duration235.640μs，而aicore_time66.855μs；不能把差额168.785μs全部认作某一个调度模块，
也不能用235.640μs代替Native的核内指标，从而声称PTO Score134μs已经在核内明显领先。
任务整体边界、各核启动时刻、工作均衡及退出边界均需另有证据才能分摊差额。

## PTO DFX

Simpler `src/a2a3/runtime/host_build_graph/aicore/aicore_executor.cpp`在`execute_task(exec_payload)`
前后读取syscnt。`simpler_setup/tools/swimlane_converter.py`把这个差值写为`kernel-duration-us`。

- `kernel_mean_us`：该任务所有实际worker调用区间的平均值，包含函数内部DMA、流水/跨核等待与算术。
  ACK前的local_setup和接收/传播时间不在该值中；不能解释成纯Cube或纯Vector时间。
- `kernel_max_us`：当前窗口最慢的一个调用。本报告再对四个窗口的最大值取平均，原窗口数据仍保留。
- `worker_envelope_us`：本组首个receive到最后一个worker结束的区间，含setup和启动分散。
  它与核时的差值不等于纯AICPU调度开销。
- `start_spread_us`及`max_blocks_per_core`分别检查首尾启动差和同核多份，避免只看平均核时掩盖尾部。

## 如何判断剩余差距

Native QLI内部含系数准备、Score、本地Top-K和必要的全局归并，PTO把其中一部分拆成独立任务。
Native Sparse内部规约与PTO的`qk_pv`、`merge_norm`也不完全同范围。
因此保留Native原始PMU值、任务总时长，以及PTO每个任务的核时/最慢核/包络，分别说明工作量和分工。

当Native block数与硬件core数相同时，Native PMU平均时间可作为接近的每核执行参考，
但仍缺Native逐核分布，且两侧函数范围不同。不能只凭两列平均值给出可回收的精确μs，
更不能将不同运行的重叠任务求和后从完整CSA正式计时中扣减。

完整CSA收益只取同轮20次无profiler图计时；独立Native/PTO profile和四DFX用于解释热点，
不与正式样本逐步配对。不把小量标量/代码组织候选的失败推导成“核内已无差距”。
