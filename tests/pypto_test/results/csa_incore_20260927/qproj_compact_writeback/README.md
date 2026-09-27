# 第三项：Q_B FP16紧凑写回先导，核内退化，撤回

Native `dsa_v1.py::_mla_prolog_multistream` 使用quant_matmul同时应用逐通道weight_scale、pertoken_scale并输出BF16。
当前PyPTO `tile.store(pre_quant=...)` 仅能表达编译期常量缩放，不能将两种动态scale直接移入FIXPIPE。
本先导只试NZ性能路径：沿用Q_B矩阵形状和任务/依赖，INT32 Acc经FIXPIPE乘2^-10写FP16，
Vector转FP32后乘2^10，再用原有row/channel scale、RMS及RoPE。
中间GM读写字节减半；新增FP16舍入发生在动态缩放之前，不声称复刻Native或数值中性。
K1024全范围INT8累加缩小1024倍后处于FP16有限范围。共享精度实现未改。

CPU完整根编译通过；首版dtype全局别名不能被JIT推断、if两分支不同dtype冲突，最终试验直接使用FP16。
**这是仅NZ的隔离先导，不是已实现ND兼容的生产功能。** 若获益才值得补齐独立ND入口；现已全部撤回。
任务task_20260927_172818_25903511079退出0，仅8K/B40四窗口DFX及现有单层诊断；不扩测。
配置：layer4正式权重、合成输入/历史、S6/TP1/mode2/atomic1/deterministic0，无EPLB。

| 项目 | dba5c01f 基底 μs | FP16写回 μs |
| --- | ---: | ---: |
| Q_B matmul | 69.74–75.02 | 75.19–77.19 |
| dequant/RMS/RoPE | 53.38–57.44 | 57.77–63.18 |

四窗口block核内均值，两项均退化；不能把GM字节减半直接当成耗时降低。
没有新增无profiler本体计时，因此不归因总体性能；依据核内退化撤回。
保护区、Top-K结构通过，输出/状态非有限值0；输出对Native max_abs仍0.03125，
RMSE 0.003292750→0.003293042，Top-K替换901不变；Native零容差仍FAIL。
这不等于逐token/DSpark验收。本轮三个核内点到此收尾，保留前两项，直接进入调度阶段。

[统计与泳道路径](report.json)、[试验补丁](candidate.patch)、[隔离编译脚本](compile_candidate.py)。
