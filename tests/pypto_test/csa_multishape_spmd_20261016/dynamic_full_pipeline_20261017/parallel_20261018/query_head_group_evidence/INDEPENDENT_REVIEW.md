# 主 Query 8/32-head 分组的依赖安全审计

结论：**可以把 8/32 作为既有动态 head 分组的补充；保留当前完整等待集合时，不需要修改 cube_schedule/vector_schedule 的 wait 或 publish。** 最小修改范围是 query_projection_stages.py 的选择条件及相关中文注释，完整/尾行数学、pipeline stage=2、同步和 GM 视图保持原样。本次只读正式源码、历史文本及已有生成物，未编译、未上卡、未创建源码候选。

## 可实施的准确规则

正式入口固定 H=64、每块 8 行、48 个 AIV。对 h=64、32、16、8 按降序选第一个满足以下条件的 h，否则保持 4：

```text
period(h) = Q_DEQUANT_WORKERS * Q_ROPE_T_TILE * h // H = 6*h
T >= period(h) and T % period(h) == 0
```

即依次检测 384、192、96、48 的正整倍数。这是按设备实际 T 和 48 worker 推导的工作量规则，不能写成 B/T 样例白名单，也不要改成只有“大于阈值”。新增分支分别放在既有 64 与 16 之间、既有 16 之后。

此时任务数 K=(T/8)*(64/h)=48*(T/(6h))，每核领取 K/48 个 head 组，非空且均匀；h 不改变总 head×八行数学次数。例子仅用于说明规则：

| 实际 T | 旧 h→新 h | 全部准备组数 | 每 AIV 的 head×八行次数 |
| --- | --- | --- | --- |
| 48 | 4→8 | 96→48 | 8，不变 |
| 96 | 16→16 | 48→48 | 16，不变 |
| 144 | 4→8 | 288→144 | 24，不变 |
| 192 | 16→32 | 96→48 | 32，不变 |
| 384 | 64→64 | 48→48 | 64，不变 |

可能节省的是 [query_projection_stages.py:187](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/query_projection_stages.py:187) 到 199 的每组 scale/cos/sin/swap 读取及 gather 索引准备、外层循环开销；每 head 的权重 scale、投影读取、归一化与输出字节数不减少。不能把组数减半直接记为整层收益。上述规则是保持完整工作波次的保守候选边界，不声称其它整八行 T 在数学上不能使用较大 h。

## 生产者与可见性

1. [vector_schedule.py:756](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/vector_schedule.py:756) 的 PHASE_QB 等待参数为 (first=0,count=128,tile=1,workers=24,base=0,stride=1)。[wait_row_owners](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/phase_sync.py:121) 将其化为全部 AIC 0..23；并未按 dequant head owner 缩减。Q_B 列块 owner=(column//N_TILE)%24，24 个发布均在 [cube_schedule.py:312](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/cube_schedule.py:312) 的完整 Q_B helper 之后。因此任意新 h 的所有 Q_B 输入仍被覆盖；ND 的 64 列块也只涉及这 24 个 owner。

2. 每个有效 (row,head) 的唯一写者为
   `(((row//8)*(64/h)) + head//h) % 48`。改变 h 的确改变这个映射，但 [cube_schedule.py:381](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/cube_schedule.py:381) 当前等待全部 AIV 24..71 的 PHASE_QUERY；[vector_schedule.py:768](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/vector_schedule.py:768) 仍由全部 48 个 AIV 在本核所有 dequant 工作结束后发布，包括空 worker。发布/获取的 fence 与 cache 维护不变，新的写者仍必在消费集合中；T=0 时也不依赖旧代空槽。**不要伴随该候选剪 QUERY 发布或缩 QUERY 等待。**

3. QR scale 与 RoPE 准备的获取已在 [vector_schedule.py:562](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/vector_schedule.py:562) 和 574 提前完成：均覆盖 [0,T) 的全部八行 owner，与 h 无关。新增分组没有新前驱、没有提前发布，也没有改变跨阶段 release 传递。q_flat 与 q_heads 的同址视图仍由 [binding.py:238](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/binding.py:238) 建立；新 owner 只重分配不重叠的行/head 输出。

若未来另行缩等待，QB 必须按该 AIV 全部 work 的列区间求 producer 并集，QUERY 必须按 AIC 实际 query 集合和上面的新 h 公式求 AIV 并集；旧静态 headtile 等待表不能直接复用。这不是本候选必要修改。

## 尾块与数学

所有 h>4 的候选 period 都是 8 的倍数，故 T%8!=0 时一定仍选 h=4。这样 [query_projection_stages.py:274](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/query_projection_stages.py:274) 固定四头的尾循环继续与外层 hg 步长一致。若只写 T>=48 就选 h=8，例如 T=49，最后八行块每组只写四头、却按八头跳 hg，将漏掉半数头；完整等待不能补救这种漏算。

完整八行继续逐 head 做 INT32→FP32、先 activation_scale 后 weight_scale、512 列 RMS、rsqrt、NOPE/RoPE、BF16 舍入；没有把多个 head 合并规约。尾块继续 recip(sqrt) 及有效行 store。T=0、非 8 倍数 T 和容量尾部的既有路径保持不变。

## UB 与已有生成物

正式 query_projection_stages.py 与本轮长 B16 N128 的冻结 reference 文件直接文本相同。读取已有 [PTO](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/pv_n256_h131072_b16_v1/build_output/decode_csa_spmd_q1m9wnpj/ptoas/dynamic_csa_full.pto:9456) 可见 h 仍是运行时 Scalar，完整 head pipeline 的主循环按两头展开（9523–9525），并有原 remainder 路径；并未按 h 同时创建 h 份 tile。新增的 8/32 都是偶数，仍沿用同一两头流水。

所有数据 tile 形状与 h 无关：主投影/FP32 临时是 8×512，NOPE 是 8×448，RoPE 是 8×64，准备索引是 1×512。现有两份投影 tile 地址为 65536、81920，各 16384B；完整八行路径所有分配的最大地址末端为 131072B，尾路径为 122944B。对应旧 C++ 运行时 head 上界与两头循环在 [dynamic_csa_full.cpp:47906](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/pv_n256_h131072_b16_v1/build_output/decode_csa_spmd_q1m9wnpj/ptoas/dynamic_csa_full.cpp:47906) 和 [48166](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/pv_n256_h131072_b16_v1/build_output/decode_csa_spmd_q1m9wnpj/ptoas/dynamic_csa_full.cpp:48166)。

源级最坏同时展开的 head 数和 tile 尺寸不随 h 增加；准备 tile 只是跨更多 head 迭代存活。这里的地址末端不是精确并发峰值，也不是新候选 lowering 的保证：若实施，第一次 CPU 核对仍须确认增加 Scalar 选择分支没有意外改变分配或循环展开。本审计没有声称新候选已编译通过。

## 历史排重及边界

[MASTER 第 7865/7871 行](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md:7865)、[旧依赖合同](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/qproj_dequant_dependencies.py:17) 和 [逐成员模型报告](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/observed_member_cost_v1/README.md:34) 已分析过 B16/T96 的 head16→8/4，以缩小旧静态投影/后处理依赖集合。该模型没有上卡；收益很小且容易被额外循环成本抵消。本方案保留动态完整等待，在旧 h=4/16 的合适 T 上扩大至 8/32、减少准备组数，未重复旧模型的“减少 head 分组以删依赖”实现。

已检索 MASTER 及历史 Markdown 中的 headtile/头组/主 Q 后处理记录，未找到这一 384/192/96/48 动态完整波次规则的既有设备失败。Attention 角色拆分的 8/16-head UB 超容记录属于不同 helper，不可套用为本 Query 后处理的失败证据。结论仅为源级依赖与资源形状审查，不代替新候选精度或性能验收。

