# 四 Scalar 几何缓存：采用区间的只读预审

**四档仅短 B32 正向，不采用、不构造白名单、不再扩测。** 四档完整精度均 PASS；主正式 `policy.py` 与冻结 668 参考逐字相同。交叉结果到达后已只读补齐原件，本审阅未等待或提交任务、未改源、未编译。

| S6 case | T / 每 AIV query 数 | Indexer 策略 | OA/OB；KV 顺序 | 同轮完整 L1 |
| --- | --- | --- | --- | --- |
| 128K B16 | 96 / 4 | G6、balanced、roots1 | 128/96；晚 | 877.84→882.72 µs，−0.556%，0/5 |
| 8K B32 | 192 / 8 | G2、非 balanced、roots2 | 256/128；早 | 1005.18→982.90 µs，+2.217%，5/5 |
| 128K B24 | 144 / 6 | G6、balanced、roots1 | 256/128；早 | 1163.42→1182.38 µs，−1.630%，1/5 |
| 8K B16 | 96 / 4 | G2、非 balanced、roots2 | 128/96；晚 | 698.60→710.04 µs，−1.638%，0/5 |

源码可省的工作由每条 query 的有效块数 k 决定：每 lane 原 k 次几何扫描变成 1[k>0] 次。全有效请求下，一次扫描约有 3B+2 次源码 Scalar 读取；成本随 B、每核 query 数与有效 mask 连续变化，没有自然的 B16/B24/B32 盈亏门槛。它不是 HBM 事务计数；四 Scalar 跨 tick 的真实寄存器代价与关键路径占比尚未测出。

已有条件的解释力有边界：

- **G2/G6 不是几何算法分界。** Attention 的 CORE_GROUPS 固定 24，`attention_query_geometry` 不使用 QUERY_GROUP。短历史 S6 的 B24/T144 按现有波次规则反而选 G6（24≤30），B32/T192 选 G2；长历史 T<24 也可选 G2。因此不能把 `group==2` 当“短历史”或低几何成本的同义词。
- **已有短历史域**是 `2048<=max_seq_len//4<=8192`（Host 已有上界，测试传 history+6），对应 prefetch0/Score768。短 B16 同样是 G2/非 balanced/roots2，却 0/5 改善，直接否定“G2 就适合”或“短历史就适合”的推断。
- **T>=144**确为原早 KV 顺序边界，OA/OB 则在 T>96 改变。长 B24 与短 B32 同走早 KV、同用 OA256/OB128，却回退；因此早 KV 或大输出 tile 也不能单独解释收益。
- 现成单轴策略均无法支持采用；“短历史且大 T”只能筛出唯一正点。它虽可写成连续布尔条件，仍是结果驱动的交集选择，没有第二个正向区间证据。B/T 与扫描量的连续增长也没有证明净收益单调，不能另拟 B 或 T 阈值。

四档精度/图/保护通过仅说明已测合同正确；五个窗口是同一作业内样本，不是五份独立平台重复。当前不知道短 B32 正向的根因：没有本候选几何独立端点、最终寄存器代价或可分离关键路径证据，不能把 cache、Scalar、MTE 回压或上游策略当成确认原因。源码/CPU 正确性、逻辑少读次数和旧 tick gather 总时长都不能代替区间收益；30% 相对原 task 门槛和 Native/token 其它验收也不能由此推导。保留冻结候选及四档原始结果，正式 policy 不改。

证据：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_scalar_geometry_evidence/source_reference/policy.py:83`、`attention_stream_stages.py:237/325`、`attention_plan_stages.py:310`；四作业各自 `source_matrix/result.json`。本目录 `review.json` 保存四档原始五样本和上述条件边界。
