# QA v2 收窄到既有 G6 区间的只读评估

结论：认可 `group == 6 and 16 < tokens < 32` 作为保守、有限的策略收窄；它沿用既有 Indexer 分界，未使用 batch 白名单。但三点不足以证明 G6 是收益根因，也不足以承诺整个历史/混合 Q 范围都加速。这里的性能结论仍是已测代表点的结果，不新增测试或修改冻结源。

## 条件实际覆盖什么

正式 [policy.py:79](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/policy.py:79) 的 long_history 定义为 max_seq_len//4 > 8192；该路径只有 tokens>=24 且 max_query_tokens>=6 才选 G6。短历史且 16<T<32 时 pair_waves=1、request_waves>=1，G6 判据为 24*request_waves<=10，永不成立。

因此建议条件在当前策略中等价于：

`long_history and max_query_tokens >= 6 and 24 <= tokens < 32`。

应直接使用已经计算出的 group，让它跟随实际 Indexer 策略；不另写 T24/30 或 B4/5 枚举。区间内仍恰好两份原 M16 尾块，新增 QA 核是8..15，权重逻辑总载荷不增，数学/owner/valid tail 证明沿用已冻结 v2。其它区间只选择 QA_M16_PARALLEL=False，constexpr 关闭新数学及等待分支。

## 同轮结果与证据边界

原始五窗在主 PAR 的 `qa_small_m16_v2_h131072_b{3,4,5}_v1/source_matrix/result.json`，三档完整精度/状态/图/保护区通过。本次只读了这些原始结果的角色键，未按字段排列推断。

| T / Indexer | 原分 task μs | 9d 参考 μs | QA v2 μs | 参考到候选 |
| --- | --- | --- | --- | --- |
| 18 / G2 | 548.36 | 572.84 | 586.20 | 回退 2.332% |
| 24 / G6 | 515.94 | 548.26 | 540.98 | 降低 1.328%，4/5 配对改善 |
| 30 / G6 | 582.92 | 615.46 | 604.14 | 降低 1.839%，5/5 配对改善 |

选择现成 G6 边界有源码结构依据；保留 T24、T30 两个正向代表、排除 T18 负例，比采用全部16<T<32更保守。不过 T18→24 同时改变 T、G、panel、roots/balanced 等条件，不能从这组三点隔离“G6导致收益”的因果。

[Cube 调度:258](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/cube_schedule.py:258) 中 QA 后紧接本核 Indexer compressor。新增 QA owner 会推迟它们原可先行的 Indexer 工作；当前结果是完整 L1 净效应，不能用 QA 并行度或权重不增推导净收益。混合 Q、不同请求容量及长历史长度仍会改变后继工作量；没有对此额外泛化的观测。

既有 v2 的 False C++ 与同控 9d 对照完全一致（仅排除整行追踪注释），证明范围是已编译的 G6/OA128/OB96 controls；其它控制组合源码 False 分支保持原 AST，不能扩称逐组合机器码均已比较。G6 收窄只改 Host 布尔选择，不需要改变已验证 True/False 数学与接线。

这是实测支持的有限启发式采用边界，不是达到原分 task 30% 目标的证据；三档候选均仍慢于各自原分 task。Native/token 连续轨迹不在本次验收范围。

