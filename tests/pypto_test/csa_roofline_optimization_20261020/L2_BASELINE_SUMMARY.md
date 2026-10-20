# B28 当前基线的真实 L2 计数

任务 `task_20261020_203423_3258805852` 已完成，完整精度、状态与图检查通过。

每方三个完整图回放，实际 MIX 为 24 AIC + 48 AIV。下表只计算工具返回的
`read_cache_hit + read_cache_miss_allocate` 两类事件，不假设覆盖全部访问类别；
不是 HBM 字节量，也不是某个阶段的计数。

| 路径 | 核类型 | miss_allocate 在两类读事件中的中位比例 |
| --- | --- | ---: |
| dynamic_pto | aic | 33.58% |
| dynamic_pto | aiv | 17.21% |
| spmd_current | aic | 33.60% |
| spmd_current | aiv | 16.91% |

当前两条路径的分配型读缺失比例接近，不能据此把单 SPMD 的差距归因于
总 L2 命中率，也不能定位到 O_B。预取候选需要同轮效果与必要的后续计数确认。
PMU 的外层 Duration 不属于完整 L1 验收口径，未加入性能表。

上一轮 MemoryAccess 附带的 `l2_cache.csv` 仅对应夹具 TensorMove，已明确排除。
