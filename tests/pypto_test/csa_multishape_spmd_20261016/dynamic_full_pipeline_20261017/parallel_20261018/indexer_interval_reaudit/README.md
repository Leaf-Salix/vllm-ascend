# Indexer 区间策略独立复审：暂无新增候选

2026-10-18。本轮结论：没有找到超出既有证据、值得推进的七档区间策略；保持正式 query G、历史叶、Score tile 和 owner 分配。此结论表示当前证据不足，不是证明现有实现已最优。本轮只读源码、历史及已有结果，没有编译、上卡、修改实现或重跑工作量模型，没有计算散列。

主仓：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1`。
正式包：`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd`，当前 HEAD 为 `e58169ec`，该包无 tracked 改动。将当前 `policy.py`、`indexer_group_stages.py`、`indexer_score_stages.py`、`indexer_publish_stages.py`、`request_geometry.py` 与前轮 `query_group_audit/evidence_v1/source_snapshot` 逐文件直接文本比较，五份均相同。因此前轮真实 Q/L/position 模型可以原样引用，不必重复 960 个均匀与 790 个 mixed-history 例子。

## 量化依据与固定开销

下列数值来自原模型 `evidence_v1/model_result.json`，仅为同一 G、同一 Score tile 下的工作量；没有新增模型、阶段计时或性能预测。

| S6 场景 | 实际 G / groups | 全体 Score steps | 当前最忙核 steps | 平均工作量向上取整 |
|---|---|---:|---:|---:|
| 128K/B4 | 6 / 4 | 132 | 6 | 6 |
| 128K/B8 | 6 / 8 | 264 | 11 | 11 |
| 128K/B16 | 6 / 16 | 528 | 22 | 22 |
| 128K/B24 | 6 / 24 | 792 | 33 | 33 |
| 128K/B28 | 6 / 28 | 924 | 41 | 39 |
| 8K/B16 | 2 / 48 | 144 | 6 | 6 |
| 8K/B32 | 2 / 96 | 288 | 12 | 12 |

低 B 长档已经用均衡叶占满 24 个 Score 核，不存在“仍按 G2、没有下放 G6”这个缺口。除 B28 外，固定现有数学步数时，单纯重排无法降低最忙核的步数下界。B28 的不可分重叶给出更强下界 40；原审计已算过 41→40，但轻核有效叶最多增到 8，根写回和叶尾 drain 集中，不能把少一步当成整层收益。

固定开销确实存在：每个有效 leaf 重读四项组描述符、加载 Query/coeff、重新开始双槽并在末尾排空；G6 Query/coeff 逻辑合计 60 KiB，G2 为 20 KiB。源见 `indexer_score_stages.py:127–172,174–176,375–378`。组描述符已经在 planner 一次生成（`indexer_group_stages.py:131–162`），不能再把“消除每叶 O(B) 请求扫描”当新候选。减少 leaf 数会同时改变根数、合并树和 tie 次序；现有每叶 16K 试验正是减少这类开销，不能换名重做。

短档 non-balanced 的半叶边界为 `min(ceil(valid/tile)*tile/2,4096)`（`indexer_score_stages.py:604–608`）。只改 768/1024 或合并多个块，可能同时改变半叶候选归属和排序输入顺序；不能仅按 READY/ACK 次数减少判断为等价收益。短 G2 的现有局部缓存、预取结果也不支持按 B16/B32 单点建立策略。G2→G6 的 Key 重用须支付更大物理 QK：前轮模型中短 B16 仅 16 核活跃、最忙核 QK MAC 增加 50%；短 B32 虽有 24 核，最忙核 QK MAC 同样增加 50%。不能假设同步极低或 Key 带宽必然主导来抵销它。

S1/S5/mixed Q 已由原审计覆盖。S1 不存在跨 query Key 复用，S5/mixed Q 的实际组数必须按 `sum(ceil(Q_r/G))` 计算，且 S5 物理 M 补齐是正确性修正，不能删掉它以降低模型工作量。host maxQ 与 T/B 不能证明每个请求都有足够 live query。未找到新的普遍区间条件能绕过这些代价，同时保留当前 exact 合同。

## 精确排重路径与证据层级

下表路径均相对主仓。MASTER 指 `vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`，VALIDATION 指 `tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md`，M 指 `tests/pypto_test/csa_multishape_spmd_20261016`。

| 方向 | 已有依据与本轮处理 |
|---|---|
| G4＋2、heavy/tail 配平与 L0 驻留 | VALIDATION:7154–7176；`tests/pypto_test/results/csa_incore_20260927/indexer_group4_balanced/README.md`。旧 1411.08/1348.57 μs 均未优于 1304.87 μs，不重复。该旧记录的 Native 零容差仍 FAIL，不能表述为当前 exact 已通过。 |
| G3、低 B 长 G6 | VALIDATION:10383–10411 明确旧 B8 两组三 query 改 G6 已采用，B4/B8 叶均衡也是现有策略。MASTER:6495–6539 的 M192/N128、WS K192 两组三 query 883.00 μs 为 G6 内部拆分试验；它不是独立动态 G3 policy 的同进程证明，亦不作为新的 G3 候选重复提出。 |
| G6 N128、双 Acc、N32、N80 | MASTER:6495–6539、13086–13088；`M/INDEXER_PARTITION_EXPERIMENTS.md`、`INDEXER_N32_ACC_EXPERIMENT.md`、`INDEXER_N80_EXPERIMENT.md`、`INDEXER_N80_WS128_EXPERIMENT.md`。已有容量失败或完整设备未采用记录。 |
| 16K 叶、1024→2048 交接 | `M/INDEXER_PARTITION_EXPERIMENTS.md`、`INDEXER_TRANSFER_2048_EXPERIMENT.md`；MASTER:13392–13404。前者减少 Query/coeff 加载，后者减少消息，均未证明整层收益；2048 首轮 root 初始化错误及修正均保留。不改名重试。 |
| N64 roll / 面板分组 / L1 ring | `M/INDEXER_ROLLED_PANELS_EXPERIMENT.md` 记录两版 CPU lowering 失败；`INDEXER_PANEL_GROUP_EXPERIMENT.md` 的 group2 设备 920.58 μs 未采用，group4/8 只有源码覆盖。`INDEXER_L1_RING_EXPERIMENT.md:94` 的修正等待后 v5 为 806.38 μs，未证明稳定收益；旧 v4 未生效退避，不能继续用其回退归因。两 panel Key 预取见 `INDEXER_KEY_PAIRS_EXPERIMENT.md:37`，813.42 μs 未采用。 |
| 小尾块只算首 N64 | `M/INDEXER_SHORT_TAIL_EXPERIMENT.md`，B16/B24 完整 PASS 但未证明收益；B4/B8/B16 最忙核面板数不减少。不是未尝试的“历史尾部特化”。 |
| B28 六叶旋转、原五叶重映射 | `M/dynamic_full_pipeline_20261017/parallel_20261018/indexer_query_group_audit/README.md` 与 `OWNERSHIP_DESIGN.md`。前者有 mixed-history 退化及 tie 风险，后者仅 41→40 并集中交接；既有有限否决结论保持。 |
| 短 G2 本地 score、N128 Key 预取 | MASTER:16223–16237、16555–16594。前者 B16 改善但 B32 慢 0.43%；后者最新同进程 B16 慢 1.555%、B32 仅改善 0.299%。不组合，不建立测试点白名单。 |

历史静态试验、旧分 task 试验、源码/CPU 失败及最新动态同进程结果在此分开列明；没有把跨版本中位数当作当前的精确回退幅度。最新阶段前移也不能自动把旧负候选变成新候选，须有新的临界路径依据；当前没有该证据。

## 归档与停止点

本目录只需复制本 `README.md`。完整前轮模型、源码快照及否决依据已经归档在：

`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/indexer_query_group_audit/`

本轮不提出实施或上卡候选，不扩大模型；不把有限排重结论扩张成所有可能策略均无收益。
