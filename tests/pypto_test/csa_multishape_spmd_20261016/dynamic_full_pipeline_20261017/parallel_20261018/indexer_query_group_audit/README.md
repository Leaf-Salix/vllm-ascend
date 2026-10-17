# 当前动态 CSA 的 query_group 与叶分配有限审计

本轮结论：保持正式 `d56c9eb6` 的 query_group、leaf 边界和 owner 分配。
G4、长小 batch 的 G6、N128、双 Acc、N80 和 16K 叶均已有历史证据，不重试。
新的公约数六叶方案有 mixed history 退化及 TopK 平分顺序风险，否决推进。
保持五叶的 heavy/tail owner 重映射在 B28 只把最大 Score 步数 41 降到 40，
同时集中叶初始化、根写回和叶尾 drain；没有足够收益证据，保存设计而不实现或上卡。
主 agent 已认可这一收口决定。

此次只读正式源码、历史记录及已有证据，在私有目录运行普通 Python 整数模型。
未修改正式包、冻结 v6 候选或依赖，未生成 PyPTO/PTOAS 内核，未提交设备任务。
五份源码快照与 `parallel_20261018/adopted_source_d56c9eb6` 逐文件直接文本相同，
见 `source_reference_comparison.json`。本目录只记录路径、大小、直接对照结果。

## 证据与复现范围

`model.py` 从快照读取正式 Scalar 函数，去掉注解和 decorator，原样解释函数体
AST：`select_policy`、`request_bounds`、group planner、work item、leaf range、
query visibility 和 visible root count。这里的 Python `compile/exec` 仅用于
普通解释器执行 AST，不调用算子编译器。浮点 Score、排序、FFTS、流水和寄存器
分配不在模型中，因此所有 step、MAC、DMA 请求及根数都是工作量，不能当作时间。

模型使用每请求真实 Q 边界、L 和 position，实际组数为
`sum(ceil(actual_query_count / G))`，不把 S6 或 `T/6` 当作动态请求数。
混合 Q 包含 1/5、1/2/3/4/5/6、6/0 和 1/1/1/6；混合历史按各组实际 visibility
计数，并包含单长请求、不同长请求位置及 stale position。范围是有效查询的紧凑
输入加 q=0 失活请求，不覆盖保留完整 T 容量的所有 dummy/padding 入口变体，也不
证明页表或浮点数值合法性。S1/S5 的 tail 物理 M 补齐修正完整保留在源码快照中。

已运行数据：

- `evidence_v1/model_result.json`：七档 × S1/S5/S6、36 个 mixed Q、对应 G2/G4/G6
  理论比较，以及 960 个均匀几何和 790 个 mixed history 叶分配例子。
- `owner_evidence_v1/owner_model_result.json`：2685 个有界例子，其中 1344 个强制 G6
  用于覆盖几何边界，其余沿当前 policy；405 个满足新 owner gate，405 个最大 step
  计数下降，2280 个退回原分配，未出现 step 计数增加。覆盖唯一性及总算术、根、消息
  数保持断言通过。这不是编译、同步时序、精度或设备性能通过。
- 复现脚本和源快照均保留，报告收口后没有增加模型点或重跑内核。

需要离线复现时可使用新的输出目录，以下命令未在收口阶段执行：

```bash
AUDIT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_indexer_20261018/query_group_audit
python "$AUDIT/evidence_v1/model_used.py" --source "$AUDIT/evidence_v1/source_snapshot" --output /tmp/csa_query_geometry_review
python "$AUDIT/owner_evidence_v1/owner_tail_model.py" --source "$AUDIT/evidence_v1/source_snapshot" --output /tmp/csa_owner_geometry_review
```

## 当前真实区间和功能约束

令 `C = host_max_seq_len // 4`。host 描述符只选编译策略，实际有效组、位置、叶及页访问
继续由 device metadata 决定；没有冻结 entry 或读取 host 上的 device 值。

| 候选行数 C | 正式 Score 策略 | query G 选择 |
| --- | --- | --- |
| C < 2048 | 小历史 Vector 尚未接入，明确报错 | 本模型不假装覆盖该入口 |
| 2048 ≤ C ≤ 8192 | tile768，无 Key 预取，双根，non-balanced | 默认 G2；maxQ≥6、B≥24，且 `4*6*ceil(B/24) <= 5*2*ceil(ceil(T/2)/24)` 时 G6 |
| 8192 < C ≤ 262144 | tile1024，Key 预取一 panel | T≥24 且 maxQ≥6 时 G6/单根/balanced，否则 G2/双根 |
| C > 262144 | 超过根工作区合同，报错 | 无可用策略 |

G2 为 N128，G6 为 N64。正式 policy 不选择 G4；模型中的 G4 使用 N64，属于理论资源
对比，未声称当前整链已编译支持。系数固定 16 行，形状为 `[16, G*64]`，先将 scale
与 weight 各自舍入 FP16 再相乘，并按 query 对角填零。G2/G4/G6 的逻辑矩阵都可表达；
改变 G 仍可能改变归约形状及根树，不能据此宣称逐 bit 等价。

Balanced 路径把完整 query 分给两个 AIV，要求偶数 G；当前验证的组合是 G6、1024、
ROOTS=1。Non-balanced 以两条 lane 的部分候选各成一根，ROOTS=2。根容量每 query
64 个槽，最多 32 叶 × 2 根；不能随意增加叶数或把 balanced 标志独立于数学路径切换。

下表是长策略的逻辑同时存活块，单位 KiB；没有新编译，不等同真实 allocator 峰值。
Left 为 Query+coeff，Right 为上一 panel 的 WS 输入+双 Key，Acc 为 QK+WS，Mat 为
Query+coeff+双 FP16 Score+双 Key 池。硬件 L0A/L0B 各 64 KiB、L0C 128 KiB。

| G / N | Left | Right | Acc | Mat 热块和 |
| --- | ---: | ---: | ---: | ---: |
| 2 / 128 | 20 | 64 | 72 | 116 |
| 4 / 128（不采用） | 40 | 96 | 136 | 200 |
| 4 / 64（理论） | 40 | 48 | 68 | 120 |
| 6 / 64 | 60 | 64 | 100 | 172 |

G4/N128 即使不算双 Key，WS Right 64 KiB 加一 Key 16 KiB 也超容量；不能只改 G。
G4/N64 没有这个逻辑硬冲突，但不能推出完整编译、流水或收益成立。Balanced AIV 每 lane
的常驻 root 加当前 segment 约 `6*G KiB`，即 12/24/36 KiB，此外还有 score、scale、
排序与全 MIX 共享峰值；184 KiB 的当前 planner 上限不能只用这部分判断。既有 score
GM arena 为 `1.5*G MiB`，balanced 不使用它也仍分配；本轮未修改工作区。

## 七档及 S1/S5 的工作量事实

以下是均匀 S6 fixture 的实际 device planner 结果；history 指原生 token 历史。
128K 加当前 query 后部分 query 可见 32769 行，因此第五叶不是“所有行都可省略”。

| H / B | G | 实际 groups | 每组叶计划（Score steps） | 总 steps | 最忙核 steps |
| --- | ---: | ---: | --- | ---: | ---: |
| 131072 / 4 | 6 | 4 | 6/6/6/5/5/5 | 132 | 6 |
| 131072 / 8 | 6 | 8 | 6/6/6/5/5/5 | 264 | 11 |
| 131072 / 16 | 6 | 16 | 6/6/6/5/5/5 | 528 | 22 |
| 131072 / 24 | 6 | 24 | 8/8/8/8/1 | 792 | 33 |
| 131072 / 28 | 6 | 28 | 8/8/8/8/1 | 924 | 41 |
| 8192 / 16 | 2 | 48 | 3 | 144 | 6 |
| 8192 / 32 | 2 | 96 | 3 | 288 | 12 |

跨 G 不能只比较 steps：每步 QK 行数为 `64*G`，WS K 也随 G 增长。例子如下，
Key 为包括末页安全重复读取的逻辑 DMA 请求量，不是实测外存流量或带宽耗时。

| 用例 / G2→G4→G6 | 最忙核 steps | 最忙核 QK MAC（百万） | Key MiB |
| --- | --- | --- | --- |
| 长 B16 S1 | 24 / 24 / 22 | 402.65 / 805.31 / 1107.30 | 64 / 64 / 64 |
| 长 B16 S5 | 66 / 49 / 22 | 1107.30 / 1644.17 / 1107.30 | 196 / 132 / 66 |
| 长 B28 S6 | 123 / 82 / 41 | 2063.60 / 2751.46 / 2063.60 | 343 / 231 / 115.5 |
| 短 B16 S6 | 6 / 6 / 3 | 75.50 / 150.99 / 113.25 | 13.5 / 9 / 4.5 |
| 短 B32 S6 | 12 / 9 / 6 | 150.99 / 226.49 / 226.49 | 27 / 18 / 9 |

S1 每请求只有一 query，G6 没有跨 query Key 复用，物理 M padding 为 G2 的三倍。
均匀 S5 的 G6 可减少 Key 重读，但模型中最忙核 QK MAC 与 G2 相同，且 roots 与归约
形状改变；host maxQ=5 或 6 不保证各请求真有五条 query。这个观察不足以支持新的
S5/mixed Q 通用 G6 区间。短 B16 的 G6 仅 16 个 Score 核活跃，G2/G4 为 24；短 B32
虽然 G6 全核活跃，最忙核 QK MAC 仍为 G2 的 1.5 倍。不能用假设同步便宜补出收益。

## 为什么不改变叶边界

正式 `indexer_long_leaf_plan` 仅 groups=4 取 6 的倍数、groups=8/16 取 3 的倍数。
有限候选把此规则推广为 `m=24/gcd(groups,24)`，仅接受 `1<m<=6` 且容量允许的叶数，
并按 group 周期旋转逻辑 leaf 编号，避免 6 叶同号反复落同核。

B28 的 5→6 叶：不旋转时最忙核 42 steps，旋转后 39，原为 41；但是有效叶 140→168，
根写回 840→1008，最终消费根 756→1008，末级 merge 252→336，叶尾 drain 252→336，
Query/coeff 请求量增加 1,720,320 B。总 ACK 等待数仍是 924，增加的 drain 是等待位置
变化，不是净增 84 条 ACK。960 个均匀例子有 14 个 step 退化，790 个 mixed history
有 161 个退化；例如 B28 交替 128K/8K 的最忙核由 27 增到 39。真实混长反例不能被
统一 max_history 上界隐藏。

更关键的是最终 TopK 的 sort/mrgsort 平分顺序依赖叶内切分和根输入顺序。
`indexer_publish_stages.py:45` 明确禁止改变原有 incoming/root 次序。即使选出合法的
Top512 集合，修改叶数或边界也不保证当前原 task/正式默认 exact 合同。这一候选不推进。

## 保留五叶也只有有限空间

112 个 8-step 重叶和 28 个 1-step 尾叶，如果仍固定 `core+24*k` 的 140 个 item，
20 个核各 6 item、4 个核各 5 item。若最大不超过 40，前者最多装 4 重叶，后者最多
装 5 重叶，总容量仅 `20*4+4*5=100<112`；所以固定 item 数的任意置换都不能低于 41。

改为可变 item 数、重叶轮转且尾叶只投给少一个重叶的核，B28 可达到理论下界 40。
叶边界、根槽号和 merge 顺序均可保持，但 core16..19 的有效叶由 6 增到 8、根写回
36→48、尾 drain 10/11→12；core0..3 由 41steps/6叶变为 40steps/5叶，仍承担较多
Score，不能宣称已解除此前 Hadamard owner 与 Score owner 的资源偏斜。

若仅作示意，令每 step 成本 s、每叶额外成本 h，旧慢核为 `41s+6h`，新的轻核可能为
`36s+8h`，要比旧慢核短需 `h<2.5s`。真实 AIC/AIV 成本、FFTS、前序起跑和流水并不
符合这个单参数模型，本轮没有测量证明条件成立。只少一步不能当作整链加速证据，
也不值得现在增加 plan ABI 和分配分支。完整 gate、O(1) 映射及接口代价见
`OWNERSHIP_DESIGN.md`，本轮停在设计层。

## 已有试验，避免重复

主历史文件为主仓 `vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`。
该文件链接/汇总的更早验证记录为 `tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md`。
以下数据仅说明历史，不与 d56c9eb6 做跨版本差值。

| 方向 | 精确历史位置及结果 | 本轮处理 |
| --- | --- | --- |
| 真正 query G4，4+2 分组 | VALIDATION_LOG:7154，§187–188；旧配平和 L0 驻留的 CSA 1411.08/1348.57µs，旧 V10 1304.87µs；均撤回，旧 Native 零容差仍 FAIL | 不重试，也不把它误认成 N64 panel 打包 |
| heavy/tail 配平思想 | 同 §187 及 `results/csa_incore_20260927/indexer_group4_balanced/README.md`；旧 4+2 已先配完整重/轻组再分尾叶 | 本次五叶 G6 owner 是不同映射，思想本身并非全新；旧负证据降低优先级 |
| 长 B4/B8 扩 G6 | VALIDATION_LOG:10383，§377；原 Key 复用扩至长 B≥4，B4/B8 叶取 6/3 倍数，已保留 | 当前正式继承，不重复提出旧 G6 下放 |
| M192/N128 + WS K192 | MASTER:13086，旧完整 883µs，未采用 | 不重试；区分已实现拆 M 版本与直接 M384/N128 超 Acc |
| M192/N64 双 Acc、M384/N80、16K 叶 | MASTER:13088，旧完整 968.92/879.10/829.22µs，均未选 | 不重试；N80 WS Right 60 KiB 挤掉 Key 预取 |
| N64 panel group2/4/8 | MASTER:13390；group2 920.58µs、同轮动态 909.50µs；group4/8 仅 AST 覆盖 | 不混淆为 query G4，不把未实测项说成设备通过 |

历史 G4 的两项设备任务为 `task_20260927_143729_131008131569`、
`task_20260927_144233_134499614331`；原直接等 item 版本的排队任务已因负载模型退化取消。
历史 G6 小 batch 也存在未修改短档 CSA 回退，原报告并未将其全部归为噪声。
因此“减少 Key 字节/step”从来不足以单独证明完整 CSA 收益。

## 归档

本目录为可复制的完整归档；`artifact_manifest.json` 列出应复制文件的相对路径和大小，
排除解释器缓存。`summary.json` 是收口结论，两个 evidence 子目录保留原始结果、复现
脚本及源码。没有删除先前 cache、phase、prefetch 或传输审计资料。
