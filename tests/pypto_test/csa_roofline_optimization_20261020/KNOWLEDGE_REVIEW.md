# 用户提供的算子知识库：本轮采用范围

来源根：`/data/pyptouser/qinchuanyu/pto-eager/cannbot-knowledge/knowledge/ops`。
本轮只读，不修改知识仓、全局环境或共享安装。

已读 `knowledge-query/SKILL.md`、CLI、证据工作流及性能模式。结构化
discover 和三个 A3 定向查询均以退出码 2 返回：本地缺少
`artifacts/indexes/knowledge.sqlite3`，不是零命中。本轮停止该检索工具，
不自动构建索引、不扫描全库兜底。下列是此前直接定位并读到的具体卡片，
不是通过有效 query receipt 得到的“已审计覆盖”。沿用用户不做 hash
清单的要求，以路径、已读内容、适用范围和实测结果留证。

## 可以转成当前候选的知识

| 知识卡（相对来源根） | 已读结论与范围 | 对 CSA 的具体用法 |
|---|---|---|
| `ascendc/optimizations/p_p80_vectorized_index_compression_emit_via_gathermask.md` | 草稿，platform 未定位；比较生成位掩码、Vector 生成位置、GatherMask 保序压缩并返回数量 | 与精确阈值筛选直接对应。只借鉴算法结构；不能照抄其中 V4/int64 限制到 A3，需以本机指令和 probe 验证 |
| `ascendc/apis/simd_api/basic_api/memory_vector_compute/sort_and_merge_isasi/sort32.md` | stable，明确列 A3；同分保持原输入位置优先，32 项为独立排序块 | 保序压缩有希望保持真实单调 index 的同分规则，但还须验证跨块 MrgSort 和 512/1024/2048 三种排序树 |
| `ascendc/guides/sorting/topk_pruning_order_and_sort_field_contracts.md` | 草稿，platform 未定位；裁剪要求共同全序，尾部/哨兵/特殊值不能混入有效候选 | 筛选必须保留阈值相等项；空结果直接保留旧根，不能把 pad 的 index 写回根 |
| `ascendc/guides/sorting/topk_threshold_ties_and_bisection_limits.md` | 草稿；找阈值不是输出有序 Top-K，固定次数二分不能解决 ties，压缩别名须按具体 API | 本轮不做近似值域二分。阈值来自已经准确求得的根；有效数与物理容量分开 |
| `ascendc/guides/synchronization/attention_schedule_slots_and_platform_handoffs.md` | 草稿；数据槽、softmax 版本、最后消费者必须分别核对，950 交接能力不能套 A3 | Attention 重构须从生产者、消费者和槽复用一起设计；不能只减少通知或扩大 ring |
| `ascendc/guides/synchronization/resident_softmax_state_slot_recurrence_and_reuse_events.md` | 草稿；状态可以留同核 UB，递推必须取上轮有效版本，MTE3 完成与 Vector 完成不同 | 本轮评估将五块 score 保留在每 AIV 的 80 KiB UB，先完成统一尺度，再减少 PV 交接；容量与编译生命周期另核 |
| `ascendc/guides/operator_practice/simd_operator_optimization/memory_access/l2_cache_mode_config.md` | Frontmatter 未定位草稿，但正文明确 A2/A3/950，引用官方 SDK 指南；一次性输入可旁路，复用数据保留 | 支持按数据复用类型决定策略。CSA 的权重、跨核 GM 环和可写 cache 不能统一 BYPASS，实际运行时 hint 必须验证 |

## 不直接照搬

- `p_p60_ascendc_sort_asc_tie_break_direction_is_reversed.md` 为 950 升序
  路径经验，正文还存在表格与“NOT a fix”表述冲突；不能拿它定义 A3 的
  降序 Sort32/MrgSort。这里只保留“必须实测 tie”的警示。
- `p_p81_runtime_bounded_loop_cap_for_top_k_style_merge_s.md` 为 950 动态
  K 经验。CSA 的 K 固定 512，适合借鉴的是按筛选后的有效候选数减少工作，
  不是缩小实际 Top-K 或设置 tie_margin 后截断。
- `l2_reuse_core_distribution_sparse_varlen_attention.md` 标 950、
  draft，且无公开原始出处；只读了范围及候选设计，未把其效果作为证据。
  当前 Score 的核分配已经有负例，不凭该卡启动核顺序扫描。
- `triton/optimizations/operator_optimization/cube_mte3_decoupling.md`
  的触发点为 backward atomic scatter 与 Cube 交替。本次是 forward，
  没有相同 atomic 工作量。只借鉴批量累加、减少交接及统计重读代价的
  思路，不采用其性能倍率或移植两个独立 kernel。

## 对当前优先级的影响

首要方向仍是精确 Top-K，但压缩要落实成 Vector 掩码与硬件保序收集，
不使用逐候选 Scalar 循环。Attention 的设计补入明确的 score 常驻容量、
最后消费者和 KV 二次读取账本。L2 只做有实际指令及流量证据的策略。
知识卡是候选依据，不能代替 A3 编译、精度和完整 L1 实测。
