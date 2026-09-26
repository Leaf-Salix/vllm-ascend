# B4 forward 回退定位

B4/H131072 的6.67%回退是无profiler稳态forward的实际结果。当前主要差距定位到
FFN/MoE：前置PTO CSA完成时间在rank之间更分散，MoE分发阶段等齐开销增加；
两次专家grouped matmul也明显变慢。后一项是否由路由分布变化造成，尚缺实际专家索引和分组计数证据。

## Event 模式对照

PyPTO runtime在初始化中调用 `rtEventWorkModeSet(1)`，影响整个进程。
单卡探针证实默认0→初始化后1；显式设0则保留0。探针任务
`task_20260926_231800_1664387893` completed/exit=0，没有修改PyPTO/Simpler。
Native硬件模式1对照任务 `task_20260926_232019_2246675419` completed/exit=0，入口源码 `ce7e6067`。
两侧均为正式权重、TP1/DP=EP16、容量40、预算256、S6、mode2、相同捕获档位，EPLB关。

| 实现 / CANN event | 无 profiler forward 均值 ms |
| --- | ---: |
| Native / 原默认 | 48.483 |
| Native / 显式硬件1 | 47.952 |
| PTO性能版 / 初始化后硬件1 | 51.715 |

16rank各10步，预热与窗口同六档主表。Native改硬件模式后未变慢，因此不能把6.67%回退
归咎于两侧event模式不同。当前PTO对Native硬件模式仍慢7.85%。三组逐token、DSpark计数一致。
该对照为独立诊断，不替换原六档默认行为的部署主表。

## 同为硬件 event 的 trace 拆解

以下是各16rank、独立3步Level0诊断窗口的均值，不是无profiler样本的逐项分账。

| 主图区间 | Native ms | PTO ms | PTO增加 ms |
| --- | ---: | ---: | ---: |
| 21个C4 attention半层 | 16.912 | 17.337 | 0.424 |
| 其他attention半层 | 9.167 | 9.502 | 0.335 |
| 43个FFN半层 | 21.488 | 24.154 | 2.666 |
| 半层之间的间隙 | 0.142 | 0.281 | 0.139 |
| 主图首个HC_pre至末个HC_post | 47.709 | 51.274 | 3.564 |

约74.8%的诊断窗口增量在FFN半层中。之前不同默认event模式的trace曾显示Native C4合计
18.873ms、PTO17.337ms；Native切硬件模式后profiling扰动明显减小。因此撤回据旧trace
推断“B4 CSA本体已经更快”的结论。原无profiler forward测量没有混入编译，数值保持有效。

FFN内四类关键kernel的每步时长汇总如下。这是各kernel时长的描述统计，不能把所有并行
kernel求和当成FFN区间；上表区间按首末设备任务计算。

| Kernel | Native μs | PTO μs | PTO增加 μs |
| --- | ---: | ---: | ---: |
| MoeDistributeDispatchV2 | 4415.660 | 5506.804 | 1091.144 |
| GroupedMatmulSwigluQuant | 4060.179 | 5037.628 | 977.449 |
| GroupedMatmul（down） | 1939.074 | 2392.124 | 453.050 |
| MoeDistributeCombineV2 | 4029.012 | 4088.898 | 59.886 |

## MoE 分发中的跨 rank 到达差

各rank的本地profile序号可能错开一个全局迭代，不能把本地“第0步”直接叠在一起。
按同层dispatch结束时间匹配同一轮：要求全16rank到齐，且与rank0结束时间偏差不超过100μs。
实际结束时间跨度约16～18μs；Native有3轮完整交集，PTO有2轮，未匹配轮明确丢出分析。

| 紧接C4 attention的dispatch | Native μs | PTO μs |
| --- | ---: | ---: |
| rank最早到最晚启动的跨度 | 35.933 | 113.293 |
| 各rank相对最后到达者的平均领先时间 | 19.008 | 82.359 |
| dispatch平均持续时间 | 61.797 | 123.630 |
| 最后rank到达后至各rank结束的平均时间 | 42.769 | 41.269 |

最后rank到达后的剩余时间接近，主要增加发生在各rank到达之前；时间关系指向前置CSA完成不齐。
这个额外开销显示在后面的MoE dispatch内部，所以单看CSA中位数会漏掉它。
进一步按同一轮CSA结束时间核对：跨rank的CSA结束跨度平均31.280→112.821μs；CSA结束到dispatch启动的间隔平均87.894→89.653μs，PTO同一轮各rank的CSA结束与dispatch启动时间相关系数平均0.9916。这补充了前置CSA尾延迟传递到MoE的直接时间证据。
这是诊断窗口里的同步时间关系，不能精确折算成无profiler窗口的1.33ms收益承诺。

## Grouped matmul 的后续确认点

正式配置 `num_hash_layers=3`，Native `DeepseekV4MoE` 对0～2层使用token-id hash路由，
第3层起依赖router_logits选择专家。0～2层两次GMM合计431.580→435.439μs，接近；
3～42层则5567.673→6994.313μs，增加1426.640μs。
因此优先核对同一满档step各rank实际专家索引、有效group_list及tile填充工作量。
这与PTO浮点路径引起的路由分布变化相符，但目前是待验证假设；不能直接定为精度或Native算子bug。
逐token与DSpark一致并不保证所有中间路由一致。第二个待确认点是前置CSA对后续访存/调度的影响。

当前不扩大六档重测，不开始新的CSA优化。后续恢复性能工作时，优先检查跨rank尾延迟和
路由/专家分组，而不是只继续压低单卡CSA中位数。

## 证据

- `comparison.json`：三组无profiler原始十步、实际worker配置、token/DSpark核验。
- `dispatch_alignment.json`：全部匹配窗口、每rank实际本地步号、未匹配项及完整主图拆解。
- `ffn_kernel_totals.json`：16rank的逐层FFN kernel时长。
- `rank0_ffn_by_layer.json`：rank0逐层明细。
- `native_hardware_ranks/`：此次Native硬件模式16rank原始profiling及导出JSON。
- 原Native/PTO主矩阵和六档泳道在 `../`。
