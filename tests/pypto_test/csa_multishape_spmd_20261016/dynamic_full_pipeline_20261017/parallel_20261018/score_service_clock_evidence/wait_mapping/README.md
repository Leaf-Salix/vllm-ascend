# Score j8 的10个局部等待：只读映射

结论：**当前 WS 标记计到的是交错流水里的 WS7 服务区间，不能解释为 WS7 的独占 Cube 计算。** 其 `M→FIX/1` 完成事件位于同一 M 指令流的 QK8、WS7 之后，依赖截至 WS7 的队列前缀；不是只等 WS7，也不是等待未来尚未提交的整个 Score step。本轮不提出新的参数试验。

本次正式668、H131072/B16/S6的三轮完整正确性/图/保护区已由主任务通过。每核22 samples；三轮24核累计量的中位数分别为 page 3.83/3.84/3.87μs、QK+FIX 0.20/0.20/0.18μs、WS 7.44/7.43/7.50μs、nested local wait 6.42/6.28/6.33μs。它们是 Scalar 原位区间，包含插桩与原依赖。这里只读取已有源码/日志，未新增测试、模型、编译或设备任务。

## 真实端点与事件映射

下面行号均指本次实际插桩 [AIC wrapper CPP](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_service_h131072_b16_v1/build_output/decode_csa_spmd_jvavx2fg/kernels/aic/cube_schedule.cpp:15122)，不是历史生成物，也不是 hook 中相对函数起点的行号。每行 set 均由同方向、同 EVENT_ID 的原事件定位，未把不同方向相同 ID 混为一条依赖。

hook明确选择 `qks[8]` 和 `wss[7]`；正式源先预取下一 Key、算当前 QK、算上一 WS。因此 page 区间实际准备 **Key9**，QK区间是 **QK8**，WS区间是 **WS7**。CPP中 Key9逻辑列偏移576，WS7写回列偏移448。当前 Key8 位于 Right57344，Key9复用 Right49152；WS的Right0占48KiB。QK Acc0占96KiB，WS Acc98304占4KiB。

| 序号/归属 | wait行及原事件 | 对应set行、原生产操作 | 等待保护的消费/复用 |
| --- | --- | --- | --- |
| 1 / page | 15253：MTE1→MTE2 / 2 | 15123；15122 `TEXTRACT` Key8到Right | Key池已完成上一提取后才继续GM→L1页载入 |
| 2 / page | 15452：MTE2→MTE1 / 0 | 15433；此前Key9两个32行分页 `TLOAD` | Key9的L1内容就绪后提取到Right |
| 3 / page | 15462：M→MTE1 / 2 | 15160；15159 QK7 `TMATMUL` | QK7读完Right49152后才用Key9覆盖该槽 |
| 4 / WS准备 | 15505：FIX→MTE1 / 7 | 15180；15178 QK7 `TINSERT`（FP16缩放/ReLU） | 将QK7分数从L1提取为WS7的Right |
| 5 / WS准备 | 15515：M→MTE1 / 3 | 15172；15171 WS6 `TMATMUL` | WS6读完Right0后才能装入WS7输入 |
| 6 / QK | 15544：FIX→M / 1 | 15179；15178 QK7 `TINSERT` | FIX读完QK7 Acc0后，QK8才能复用Acc0 |
| 7 / WS发射 | 15572：MTE1→M / 0 | 15524；15523 `TEXTRACT` 分数7到Right0 | WS7读到已提取的FP16输入 |
| 8 / WS发射 | 15582：FIX→M / 2 | 15213；15212 WS6 `TSTORE` | FIX读完WS6 Acc98304后，WS7复用该Acc |
| 9 / QK FIX | 15603：M→FIX / 0 | 15552；15550 QK8 `TMATMUL` | FIX读取QK8 Acc并做FP16/ReLU转换 |
| 10 / WS写回 | 15657：M→FIX / 1 | 15590；15588 WS7 `TMATMUL` | FIX读取WS7 Acc并写回GM；事件生产位置在QK8与WS7之后 |

关键原顺序为：15550 QK8 → 15552 M→FIX/0 → 15588 WS7 → 15590 M→FIX/1 → 15603 wait/0 → 15612 QK8 FIX → 15657 wait/1 → 15663 WS7 STORE。两次MMA使用不同Acc，所以能先后发射；这并不把同一Cube M流水变成两个独立计算引擎。wait/1之前的工作也包括WS输入就绪与Acc复用依赖，不能据源码标签把实际等待成本只归给WS算术。

## 观测能说明与不能说明的范围

WS三个片段包含前一分数提取、Right复用、Acc复用、MMA发射和末尾M→FIX等待；QK8异步执行可以落入这些Scalar区间。QK+FIX仅0.18–0.20μs不证明QK计算很便宜，WS约7.5μs也不是可通过删除WS得到的收益上限。现有local-wait计数把 **page3处＋QK2处＋WS5处** 全部累计在一起，没有逐站点时间；不能认定6.3μs主要来自最后的M→FIX，更不能从WS中直接减去该总量得到“纯WS”。三个类的中位数也不能拼成某一核的精确时间分解。

这些事实支持按 **QK/WS共享M流水、FIX、L0B/Acc复用整条链** 审视瓶颈；不支持直接删除这些RAW/WAR依赖，或由WS标签较大推导“迁移WS到AIV/缩短WS K必然提速”。原位时钟未新增中间硬件fence，最终ISA顺序、发射背压和各异步引擎完成时间没有分别测量。固定j8的22样本不可乘16外推整个Score；本轮不追加测量来填空。

## G3/N128确实做过，须分清两段历史

真正全局G3/N128早已在 [csa_indexer_triple_20260928/README.md](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_indexer_triple_20260928/README.md:7) 实现：每个S6请求两组3query，M192/N128、WS K192，24worker重新领取query-group/leaf。它曾比G2有收益，不是从未试过。随后同轮长B16由G3/N128改G6/N64，完整CSA **1191.899→1129.717μs（G6快5.22%）**，见 [S6报告](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_indexer_six_20260928/README.md:81) 和主验证日志 §267/271/272（8293、8345、8360行）。两种Score核时来自不同作业，不将其当同轮差值；当时的精度检查范围也不等同今日完整exact矩阵。

后续G3/N128的Key L1-only曾保留（主验证日志 §349，9708行）；再到§377（10383行），长B8从已有两组G3改整请求G6并做leaf均衡，同轮CSA **819.977→794.187μs**、Score AIC168.931→137.435μs，八类状态exact/A→B→A通过，见 [小长档结果](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_small_long_s6_20260929/RESULTS.md:3)。该轮同时含leaf均衡，不能只归因于G。

另一个容易混淆的负例是 `csa_multishape_spmd_20261016/indexer_n128.py`：audit明确 `query_group=6`、每N128面板两份M192和两份K192 WS，共用Key、保留原leaf/owner/AIV。其883.00μs结果见 MASTER:6509–6549、`b16_cached_indexer_n128_v2/result.json`；它是G6内部拆分，**不是独立G3分组**。前述全局G3历史是旧分task体系，尚不能冒称“当前668动态单SPMD同源G3/N128已经精确失败”；但足以否定“这个方向没有做过”的理由。当前j8 Scalar观测未提供推翻旧G6选择的新依据，不重新扫描G/N。

本次直接来源还包括 [hook的采样选择与分段](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_service_h131072_b16_v1/score_service_clock.py:129)、[正式Score源](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_service_h131072_b16_v1/sources/spmd_reference/indexer_score_stages.py:242)、同任务 `spmd_reference_codegen/audit.json` 及 `clock_round_{0,1,2}_spmd_reference/score_service_summary.json`。只归档本README即可，原始大文件留主任务目录，不复制或修改。
