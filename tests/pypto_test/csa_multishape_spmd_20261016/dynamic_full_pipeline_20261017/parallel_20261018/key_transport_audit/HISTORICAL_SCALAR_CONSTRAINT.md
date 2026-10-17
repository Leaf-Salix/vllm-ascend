# 对 Key 预取设计的历史证据补充

此补充不改已冻结的 `../key_prefetch_design/` 模型和计数。主会话提供线索后，
本次只读复核主日志第 6860–6877、12960–12969 行，没有重新运行旧探针。

原任务 `task_20261017_022628_11712641926`，输出目录
`C/tests/pypto_test/csa_multishape_spmd_20261016/b16_cached_indexer_clock_v1`，
是**静态旧版** B16/S6，不是当前 593 动态 SPMD。

|24 个 AIC 的中位统计|原值|
|---|---|
|Indexer 阶段|210.23µs|
|页表读取到 Key DMA 提交累计|11.81µs，704 次|
|1024 候选 score_step 区间|193.96µs，22 次|
|FFTS 等待|15.00µs，22 次|

完整输出、TopK、六状态、图、30 代信号和保护区当时均通过。完整插桩 L1
845.64µs 只作诊断，不能用于性能目标。11.81µs 是 Scalar 提交窗口，不是
异步 MTE2 独占忙时；这些区间会重叠且插桩会影响队列，不能做简单减法。

此记录的约束是：**页解析/小 DMA 提交尚未被证明占 AIC 主耗时**。不能将
当前动态长 B16 的 AIV 原位 FFTS 约 107µs 全部归到这 32 页解析；也不能把
当前动态数据与上述静态旧版做差。迁移页解析可能改变供数节奏，但在增加约
132MiB Key 暂存交通的同时，其收益依据仍不足。因此本轮保留设计、停止实现，
未提交任何新编译或设备任务。

C 是 `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1`。
原日志路径：`C/vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`。
