# 当前 TND 与 nalinaly 性能版受控对拍（2026-09-29）

## 结论

四组同场 Graph replay 中，nalinaly 性能版均领先当前 TND，也领先各自页布局下的原生。
**最大的长上下文模块差距是 Indexer score/TopK：B16/T96、128K，1092.24 对264.66 μs，
包络差827.58 μs。** 8K的显著差距则包括attention之后的输出尾段：261.62对160.48 μs。

这不是同等数值契约的两实现比较。参考性能版保留按group量化等差异，输出及有效cache
未与原生逐位对齐；不能把其全部速度优势视为当前精度约束下可以原样获得的收益。
另一方面，也不能用精度要求解释全部性能差距：共享Key、任务流水、融合和布局仍值得验证。

## 源码与共同环境

| 项目 | 固定版本或口径 |
| --- | --- |
| 当前分支 | dev/pypto-dsv4-csa-tnd-main-20260924 |
| 当前源码 | HEAD 65b6263b6；实际kernel最近改动1ec096d7a |
| 参考分支 | nalinaly/dsv4-flash-pto-v0.25.1rc1 |
| 参考提交 | 7b2961536bb2ffae13ff1f5a1a849af6032f388c |
| 参考入口 | deepseek_v4_flash_dspark_perf，完整HC attention子层 |
| 共同PyPTO | 3e87a843619aca13af39755700513d26b402e924，已有独立构建 |
| 共同Simpler | a54c0509552b01e13fb0960e23ce409a01b5024f |
| PTOAS / ISA | 0.66 / 327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d |
| CANN / ATB | 9.2.0-beta.2 / 9.2.0-beta.2 cxxabi1 |
| Torch / Torch-NPU | 2.10.0+cpu / 2.10.0.post4 |
| Native | 当前vLLM-Ascend main移植基础，vLLM0.29.0 |
| 确定性 | Native level1；HCCL_DETERMINISTIC=true；参考atomic_add=0 |
| NZ | 配置2；四张参考根权重实际format29；当前kernel保留自身ND绑定 |
| 设备 | 227单卡；每组四臂在同一进程、同一张卡串行交错 |

当前常用PyPTO54957491e不支持参考所需`tile.assemble(pre_quant, pre_relu)`，
因此本轮显式复用另一套已经构建好的完整PyPTO/Simpler/PTOAS；没有覆盖原环境或重装vLLM。
这也意味着不能把本轮绝对时间直接与旧运行库、attention-only的历史时间作优化收益对比。

初次smoke漏设真实model runner的`allow_internal_format=True`，权重仍是format2，
虽执行完成，但从正式NZ2结论中排除。修正后四张根权重均检查为format29。
参考kernel源码未修改；测试侧仅映射配置字段、0.29 metadata对象、存储和权重ABI。
两组compressor norm从原生FP32绑定到参考BF16前，确认再转FP32与原权重逐位一致。

## 测试协议

- 真实DeepSeek-V4-Flash-0731-w8a8 layer2（C4）权重；HC权重同样来自checkpoint。
- 输入为固定种子的合成HC residual `[T,4,4096]` 与合成历史cache，TP1、每请求S6。
- 完整边界：**HC pre → input RMSNorm → attention → HC post**，四臂均不另做residual clone。
- Native128/current128、Native32/nal32分别保留页布局；逻辑cache输入SHA256完全一致。
- start_pos为8186/131066，六行query后达到8192/131072；测试B4/T24和B16/T96。
- 独立cache和output；执行真正Native forward或相应CSA，检查dispatch计数、有限输出。
- 24轮四臂全排列，每臂在每个顺序位置各出现6次；30次graph预热。
- 8K每轮200次replay，128K每轮100次；重置和同步在事件计时外。
- 结束后、再次reset前，比较输出和六类有效cache与首次graph快照；所有臂稳定。
- 无DFX计时与DFX4采集分开；dependency generation另开进程，不与DFX计时同时开启。
- 每实现/上下文采一次内部DFX；func_id依据精确生成的kernel_config，且与dep-gen表一致。

## 完整边界无DFX延迟

单位ms，差值列为current−nalinaly的μs。Native列分别使用128/32槽页。

| 负载 | Native128 | 当前TND | Native32 | nalinaly perf | 差值μs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K / B4 / T24 | 0.628052 | 0.661134 | 0.619417 | 0.497579 | 163.56 |
| 8K / B16 / T96 | 0.945116 | 1.121463 | 0.940574 | 0.779852 | 341.61 |
| 128K / B4 / T24 | 0.831427 | 0.933983 | 0.843585 | 0.568521 | 365.46 |
| 128K / B16 / T96 | 1.233215 | 2.121258 | 1.260826 | 0.981185 | 1140.07 |

以下是逐轮配对的current/nalinaly延迟增幅；四组24轮均为current更慢。

| 负载 | 配对中位数 | 24轮范围 |
| --- | ---: | ---: |
| 8K / B4 / T24 | 32.86% | 32.21%–33.40% |
| 8K / B16 / T96 | 43.84% | 43.57%–44.11% |
| 128K / B4 / T24 | 64.48% | 63.33%–65.17% |
| 128K / B16 / T96 | 116.16% | 115.28%–116.70% |

## 精度与重复性

输出relative L2是**含HC residual的完整子层边界**，不能与旧attention-only误差直接比较。

| 负载 | 当前/Native128 | nalinaly/Native32 |
| --- | ---: | ---: |
| 8K / B4 / T24 | 0.004566% | 0.120011% |
| 8K / B16 / T96 | 0.015077% | 0.121166% |
| 128K / B4 / T24 | 0.000000% | 0.121738% |
| 128K / B16 / T96 | 0.016778% | 0.123577% |

四组Native32与Native128的输出及有效cache逐位一致；当前六类有效cache也全部与Native128
逐位一致。参考cache存在差异：raw KV relative L2约0.282%–0.291%，index key约
0.611%–0.683%，其余state/cache也存在较小差异。因此不能将所有参考误差只归因于O-proj。
本轮没有做参考HC/attention逐阶段数值注入归因，也没有证明整模型生成精度。

## DFX模块差距（B16/T96）

单位μs；模块包络是首个实例开始到最后一个实例结束，包含内部流水和等待。
各包络可能重叠，**不能相加当整层耗时，也不等于纯算术时间**。

| 上下文 | 模块 | 当前 | nalinaly | current−nalinaly |
| --- | --- | ---: | ---: | ---: |
| 8K | Indexer score / TopK | 109.96 | 95.10 | +14.86 |
| 8K | Attention QK / PV | 155.22 | 132.32 | +22.90 |
| 8K | O-A | 94.84 | 98.80 | -3.96 |
| 8K | O-B matmul | 86.10 | 68.66 | +17.44 |
| 128K | Indexer score / TopK | 1092.24 | 264.66 | +827.58 |
| 128K | Attention QK / PV | 170.84 | 150.14 | +20.70 |
| 128K | O-A | 86.56 | 84.08 | +2.48 |
| 128K | O-B matmul | 81.02 | 61.70 | +19.32 |

128K的Indexer核心实例累计执行时间为76850.78对16130.76 core-μs，包络差827.58 μs。
DFX支持把主要差距定位到该阶段内部；尚未用PMU把算术、访存、Cube/Vector内部等待的占比拆开。
不能据此声称827.58 μs全部是多余数据搬运，或全部可在不改数值的情况下收回。

### 按任务完成点划分的连续区间

下面四个区间互不重叠，可解释各自kernel首末跨度；但两个kernel边界不同：
nalinaly含HC pre/post，当前HC和Native KV投影位于PyPTO kernel外，故不能用总和替代上面的完整Graph对拍。

| 上下文 | 区间 | 当前 | nalinaly | 差值 |
| --- | --- | ---: | ---: | ---: |
| 8K | 首个内部任务→TopK结束 | 428.94 | 409.40 | +19.54 |
| 8K | TopK结束→QK/PV开始 | 59.12 | 25.26 | +33.86 |
| 8K | QK/PV | 155.22 | 132.32 | +22.90 |
| 8K | QK/PV结束→kernel输出 | 261.62 | 160.48 | +101.14 |
| 128K | 首个内部任务→TopK结束 | 1424.14 | 563.06 | +861.08 |
| 128K | TopK结束→QK/PV开始 | 60.10 | 26.04 | +34.06 |
| 128K | QK/PV | 170.84 | 150.14 | +20.70 |
| 128K | QK/PV结束→kernel输出 | 246.32 | 156.24 | +90.08 |

8K kernel跨度904.90/727.46 μs，128K为1901.40/895.48 μs。
8K尾段差101.14 μs中，QK/PV结束→首O-A为37.42/4.40 μs，已相差33.02 μs；
首O-A→输出为224.20/156.08 μs，相差68.12 μs。
参考O-A自身包络98.80 μs并不比当前94.84 μs短；其quant/O-B在部分O-A仍执行时已开始。
参考最终任务包含HC post，当前`proj_b_act`不含，二者不能当同一反量化任务单独相减归因。

原生HC pre+norm、HC post、KV projection在8K/B16独立graph诊断分别约
81.77、25.41、17.20 μs；它们是单独启动的诊断，不与带DFX包络直接相加推导收益。

## 实现差异与后续优先级

### 1. 长上下文优先Indexer整段设计

参考长档将一个请求的S6 query组合，共享Key，使用M384/N64的QK块、FIXPIPE量化/ReLU，
并用Cube进行head系数规约；同时调整leaf分配和合并。当前正式路径按query/half-leaf处理。
以前仅减少metadata读、增加小幅预取或缩小N面板，未改变主要重复工作，因而很难收回此量级差距。

应先保留当前原生舍入和TopK语义，验证“请求内多query共享Key”的真实数据复用，
并用score、TopK、最终输出三层对拍确认。参考FP16系数与Cube规约不能未经验证直接移植；
TND还必须按query_start_loc处理不等长请求，不能恢复固定query//6寻址。

### 2. 输出尾段采用保精度的流水与融合

当前量化等待全组O-A完成；参考按group量化可早启动O-B，但改变了数值契约。
可调查按token tile推进“八组O-A齐备→原生全8192维量化→O-B”，
让不同token tile重叠，保留每token的统一scale。是否可行和收益均需新实验，当前尚未实现。
另调查当前独立merge_norm与attention输出消费、O-B反量化与HC post融合，
逐项保留BF16舍入和FP32乘加顺序。

| 数值边界 | Native / 当前 | nalinaly perf |
| --- | --- | --- |
| O-A到量化 | BF16舍入 | 使用FP32累加结果 |
| 激活scale | 全token8192维共享 | 每组1024维独立 |
| O-B跨组规约 | INT32先求和 | 每组转FP32乘scale，再求和 |
| 反量化 | acc × (token_scale × channel_scale) | 各组缩放和 × channel_scale |
| O-B进入HC | BF16 | 同样保留BF16，之后融合HC |

### 3. QK/PV和小型metadata优化后置

本轮QK/PV差约21–23 μs，明显小于128K Indexer的828 μs。
继续优先做个位数μs的小改动不符合当前瓶颈证据。

## 证据、失败记录与限制

- 静态参考NZ2编译通过；原运行库缺FIXPIPE接口的失败单独保留。
- B4长测：task_20260929_151602_379049812136，exit0。
- 后三组矩阵：task_20260929_151946_39267762508，exit0。
- 四份DFX4及四份独立dep-gen：task_20260929_152010_393591910722，exit0。
- 原始result/pre_timing、四份Perfetto、原始chip_swimlane_records、deps、精确kernel_config、
  name_map、逐任务及分组统计保存于实验报告`nal-perf-controlled-20260929/evidence/`。
- 四个Perfetto目录：profile-8186-ours128、profile-8186-nal32、
  profile-131066-ours128、profile-131066-nal32，各自`dfx/perfetto.json`。
- 本轮没有修改正式kernel，只补文档；不宣称已经实现以上优化。
- 仅单层真实权重、合成输入、固定步graph；未测试整模型、DP/EP16或非等长参考性能。
- 无DFX长测有24轮，内部DFX每配置一次；模块数据用于定位，不给出精确可回收收益承诺。
