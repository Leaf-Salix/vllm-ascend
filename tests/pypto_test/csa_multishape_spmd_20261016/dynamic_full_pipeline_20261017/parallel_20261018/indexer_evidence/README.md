# 动态 CSA Indexer 半叶 UB 候选：未采用（2026-10-18）

本轮不合入主源码。候选只在 8K/B16 得到 2.86% 的完整 CSA 收益，8K/B32
反而慢 0.43%；两点不足以建立工作量分界。另有未启用非 balanced 分支残留
UB 分配的问题，不能把当前候选默认用于全区间。七档 30% 目标仍未达到。

工作树：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018`。
基线为 `593c5d25`，生产候选仅修改 `indexer_score_stages.py`；未改主仓、依赖、
共享安装或测量主驱动，未提交/push。原始完整源码、编译物、日志保留在此私有
目录，本文和 `summary.json` 给出可复核结论；归档清单只记录路径及大小。

## 候选做了什么

既有 G2/768 非 balanced 路径逐轮把同一 AIV 的 scaled score 写入 GM，
半叶结束后再读回做 TopK。候选保留该半叶在 UB，每 AIV 两条 query，逻辑
容量 2×4224（4096 按 384 列步长补齐），物理布局 22×384，合计 33792 B。
A3 每次完整搬运 384 列，最后 reshape 后只排序实际有效的最多 4096 列。

删除这次私有 GM 往返后，query visibility 在最终排序前检查一次。G2/G6
分组、balanced 根划分、六个 sort32/mrgsort 分支及其并列优先级保持原样。
AIC score 数学、FP16 舍入位置、READY/ACK 双槽协议和 S5 尾组物理 M 补齐
修正未改。所有 pl.dynamic、device Q/L/position/page table 和访问保护保留。
该候选按已有数学策略区间选择，没有 B16/T96 或七个验收点的白名单。

`score_arena` 只是同 AIV 私有中间 scratch，本候选不再写它。完整 CSA 的
`idx_topk_scores`、TopK、输出和状态仍须精确一致；旧 `score_tail_probe.py`
直接读取该 scratch，因此新 probe 比较 Cube transfer 和完整 pair arena，
并检查候选 scratch 保持填毒值，未放宽任何可见输出的比较。

## 两档完整三方结果

每档均在同设备、同进程、相同根 Tensor 地址上，先 A→B→A 精度，再图回放，
再以轮换顺序各取五个完整 L1 incore 首尾窗口。中间等待和阶段空隙均计入。
两档 eager/graph/post-L1 的输出、TopK/score、完整 cache/state、保护区和
代际协议全部 PASS。它们不是 Native 整层或生成 token 的新精度结论。

| S6 工作量 | 原动态分 task | 默认 SPMD | UB 候选 | 相对默认 SPMD | 相对原分 task |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K/B16 | 675.50 μs | 712.38 μs | 691.98 μs | 降低 2.86% | 慢 2.44% |
| 8K/B32 | 991.12 μs | 1020.64 μs | 1025.04 μs | 慢 0.43% | 慢 3.42% |

- B16：`task_20261018_003818_31105785398`，归档 `b16_matrix_result.json`。
  候选五窗 689.90/698.80/704.00/690.56/691.98 μs；默认五窗
  700.66/714.10/712.38/709.50/712.52 μs。
- B32：`task_20261018_004709_328630619605`，由主 agent 提交；归档
  `b32_matrix_result.json`。候选五窗 1026.62/1021.02/1024.46/1025.04/1038.78 μs；
  默认五窗 1011.36/1020.64/1022.06/1018.42/1022.68 μs。
- 原始输出在主仓 D 的 `parallel_20261018/indexer_h8192_b16_v1` 与
  `parallel_20261018/indexer_h8192_b32_v1`；D 的完整路径见 `summary.json`。

## 边界和数学证据

- `source_geometry_audit.json`：消除 GM/UB 访问表达式后，六个 TopK 分支
  完整数学 AST 相同，26 个 sort/merge 调用顺序保留。81906 个几何组合覆盖
  全部 1..8192 叶长、两个 lane 和不同 query 可见前缀，缓存均不越界。
- `task_20261018_004914_333028922857`：两个版本的正式 score helper 直接
  同源执行，25 个 visible 边界×随机/全同分×live=2→1→2，共 150 轮同地址
  graph 全 PASS。整个 pair arena 与 Cube transfer 逐 byte 零差异，cache
  未改写；未写的根保留相同填毒值。详见 `helper_result.json`。
- helper 覆盖尾组 score 合同，但不替代完整 S1/S5、mixed query 或 Native
  验证。本轮没有为了本候选重复四档完整 S1/S5，没有扩大七档性能测量。

## CPU 编译经过与未采用原因

完整编译都不传样例 shape，不占 NPU。失败产物与日志保留，没有覆盖旧结果。

| 输出目录 | 结果与修正 |
| --- | --- |
| `cpu_medium_v1` | Tile helper 形参注解在当前 JIT 特化中丢失；改在 AIV stage 展开数学。 |
| `cpu_medium_v2` | 展开变量名与 balanced 分支冲突；为 UB 临时变量统一加前缀。 |
| `cpu_medium_v3` | create 位于独立 if 词法域，SSA 不允许后继循环引用；移入共同词法域。 |
| `cpu_medium_v4` | PTOAS 拒绝窄列 UB TMOV；改为 22×384 整行写，再 reshape。 |
| `cpu_medium_v5` | 完整 PASS，58 动态轴、单 MIX、根 ABI 通过；用于 B16 真机。 |
| `cpu_long_v1` | 前端裁剪前检查未启用分支，512 列源不能写 384 列缓存。 |
| `cpu_medium_v6`、`cpu_long_v2` | 尺寸改为策略常量表达式后完整 PASS；G2/768 两个完整内核 C++ 去注释/空白后与 v5 逐字相同。 |
| `helper_cpu_v1` | 默认/候选两个同源 helper 编译 PASS，随后才提交一次 helper 真机。 |
| `cpu_g6_768_v1`、`cpu_g2_1024_v1` | 两侧完整编译均 PASS，但额外 UB 分配未完全删除，后置审计 FAIL，不能称为默认等价。 |

`medium_codegen_equivalence.json` 保存 v5/v6 直接 C++ 比较。Balanced 长历史
的 `cpu_long_v2/unused_branch_audit.json` 确认新缓存和数学分支都已删除。
然而非 balanced 的未启用策略仍留下 create，影响真实活跃 tile 的地址：

| 未启用策略 | 残留 UB 缓存 | score_sums 地址：默认→候选 | kv_scale 地址：默认→候选 | 全核 Vec 末端：默认→候选 |
| --- | ---: | ---: | ---: | ---: |
| G6/768 | 101376 B | 65536→101376 | 74752→110592 | 163872→166912 |
| G2/1024 | 32768 B | 65536→98304 | 69632→102400 | 163872→163872 |

因此差异不只是未使用 Tile 的 C++ 声明/TASSIGN。即使删除这些死声明，活跃
scale/score tile 的真实地址仍不同，不能宣称生成代码与默认逐字等价。两侧
C++ 路径、UB 末端及地址例子见 `unused_strategy_allocations.json`；完整
`allocation_difference.json` 和 C++ 留在上述私有 CPU 输出目录。

主 agent 要求收束后，未继续编译/设备测试。此前已启动的
`scoped_source_v7` 副本尝试把整个 UB 单叶生命周期放入一个仅含 Tensor/Scalar
参数的有界 helper；`cpu_medium_v7` 只完成 CPU 编译，未进入正式候选、未做
设备或性能验证、未作为采用方案。原候选和所有失败历史仍保留。

## 两档差异：可确认的事实与下一步

`existing_codegen_comparison.json` 直接读取两档实际测量内核。两档 Indexer
均为 G2/N128/768、双根、非 balanced；本次均匀 S6 fixture 的实际组数为
48/96，对应 24 个 AIC 的 2/4 波。每组每 lane 都消费三次 FIFO，删除的
GM 往返和重复 visibility 工作在 B32 并没有减少，因此不能用“B32没有走候选”
解释结果。

两档候选的半叶缓存均为 33792 B、地址 65536..99328；score_sums、kv_scale
及排序临时 tile 的所选分配相同，两侧完整内核 Vec 末端也都是 163872 B。
当前生成物没有显示 B16→B32 的 Indexer UB 容量跳变或溢出。两档其它已选
策略不同：OA_COLS/OB_ROWS 从 128/96 变为 256/128；完整 CSA 关键路径也可能
随工作波数改变。现有 L1 只有整链首尾，不能判断 AIV 省下的工作是否在 B32
被 AIC、后继阶段或其它成本遮蔽。此处只是待验证的机制，不是已归因结论。

没有证据支持把 B16/T96 定为启用阈值。本轮保留候选而不默认采用。若继续，
最有价值的单一定位是：仅对 8K/B32 旧/候选做同轮 phase 32/34/35/36/37
时钟诊断，分别观察 Indexer scale、Score Cube、Score Vector、TopK 与
Attention 计划发布，确定差异发生在 Score 双槽临界路径还是后继等待。
插桩只能诊断，不能拿其耗时替代正式完整 L1，也不在当前阶段自动追加测试。
