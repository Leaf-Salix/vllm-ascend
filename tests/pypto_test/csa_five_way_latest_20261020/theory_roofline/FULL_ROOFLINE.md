# CSA 完整计算链的 Cube、Vector 与内存 roofline

日期：2026-10-20。仅离线复算已有七档，不新增设备测试，不修改算子，不push。

**结论：Vector已补齐，先前只有Cube+HBM的结果不能代表完整CSA。** 当前峰值资源模型仍显示明显余量，但这个余量包含必须的数据搬运、标量控制、指令启动及同步，不能全部记成可以兑现的优化收益。

此处“完整”指覆盖HC_pre、输入RMS、Q/KV/两套压缩、Indexer/Top-K、稀疏Attention、O投影及HCpost的计算工作量与主要数据通路；**不是周期精确的端到端时延预测**。未知的L2实际吞吐、HBM命中率和生成代码额外指令保持为未知，不填经验利用率伪装实测。

## 1. 硬件与计量口径

本机为Ascend910_9392，一个逻辑device对应一个die；24 AIC、48 AIV，同轮Profiler记录AIC/AIV均1850MHz，L2为192MiB。未使用双die整卡峰值。

| 资源/操作 | 峰值口径 |
| --- | ---: |
| Cube BF16/FP16 | 363.7248 TFLOP/s |
| Cube INT8 | 727.4496 TOP/s |
| Cube标准FP32 | 90.9312 TFLOP/s；HF32另有模式 |
| Vector FP32 Add/Mul/Min/Max | 单AIV 64元素/cycle；全48核5.6832万亿元素操作/s |
| Vector FP32 Exp/Sqrt/Div | 单AIV 32元素/cycle；全48核2.8416万亿元素/s |
| Vector Sort32/MrgSort | 单AIV 2排序项/cycle；全48核1776亿排序项/s |
| HBM | 每die 1.6 TB/s规格上限，非本机实测持续值 |
| L1→L0A / L1→L0B | 每AIC 256 / 128 B/cycle |
| Score FIX INT32→FP16→L1 | 每AIC 64元素/cycle，即128 B/cycle |

来源：[A3/2201架构](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html)、[Cube官方表](https://asc.gitcode.com/api/appendix/cube_instruction_theoretical_perf_summary.html)、[Vector官方表](https://asc.gitcode.com/api/appendix/vector_instruction_theoretical_perf_summary.html)、[Huawei CloudMatrix384 §3.3.1](https://arxiv.org/html/2506.12708v2#S3.SS3.SSS1)。

不能把FP32向量元素操作、特殊函数和排序项都称作FLOP。FMA若按两个FLOP计，与两条独立Mul/Add的吞吐也不同；本代码保留原有舍入顺序，没有假设自动融合FMA。
FP32 HC投影按标准Mmad模式计算，历史敏感性账本仍保留，未改其精度。

## 2. 如何组合这些不同资源

```text
Cube服务量 = Σ(各精度矩阵运算量 / 对应Cube峰值)
Vector服务量 = Σ(各指令类别处理元素数 / 该类每核吞吐 / 频率)
端口服务量 = 经过该端口的字节数 / 端口峰值
内存服务量 = 明确缓存条件下的HBM字节数 / HBM带宽

完整调用资源下界 ≥ max(最忙AIC的Cube服务量, 最忙AIV的Vector服务量,
                         各片上端口服务量, 对应条件下的HBM服务量)
```

先按真实核号累加再取最大值，不能把各阶段最忙核相加，也不能用全核平均掩盖固定分工。Cube与Vector、计算与DMA可重叠，不能将整层服务时间简单相加。真实数据依赖、单次指令延迟、轮询、FFTS以及有限缓冲会进一步抬高可达时延。

## 3. 七档结果

单位μs；所有S6、TP1、正式W8A8第4层、四主投影权重NZ。实测来自同日五口径数据中已冻结的动态版本13259f92，与37个正式源码文件及共享RMS helper核对一致。其它会话随后改过的多task基线不混入。

Vector范围是理想归约与通用Tile归约/末级merge处理量的模型敏感性，不是实测波动或所有可能实现的严格上下界。冷HBM两端分别是假设最大KV复用的只读冷供给，以及逐query供给并在窗口内完成外部输出写回。

| 档位 | Cube最忙核 | Vector服务模型 | L1→L0B最忙核 | 冷HBM场景 | 串行总时间 | 动态实测L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 52.72 | 45.71～52.36 | 81.37 | 124.42～133.23 | 304.22～319.67 | 476.40 |
| 128K/B8 | 95.90 | 70.70～78.55 | 110.70 | 137.28～154.89 | 414.59～440.04 | 626.50 |
| 128K/B16 | 171.73 | 127.75～138.88 | 164.39 | 163.00～198.22 | 626.87～673.22 | 802.20 |
| 128K/B24 | 276.48 | 182.98～196.68 | 261.81 | 188.72～241.55 | 910.00～976.52 | 1039.82 |
| 128K/B28 | 327.40 | 222.66～238.34 | 309.41 | 201.58～263.21 | 1061.06～1138.37 | 1167.72 |
| 8K/B16 | 86.62 | 54.30～64.05 | 86.62 | 122.99～158.21 | 350.54～395.51 | 644.08 |
| 8K/B32 | 175.46 | 93.50～108.66 | 153.32 | 134.42～204.85 | 556.71～642.30 | 933.58 |

**串行总时间 = Cube + Vector + L1→L0B + 冷HBM**，按表内四项完全不重叠求和，合计使用未取整原值。它是资源串行情景对照，不是当前代码的实测或完整阶段DAG串行模拟；仍未计入Scalar、通知、指令启动等尚未定量的开销，不能称为真实耗时上界。

下面取资源最大值。热起始一列允许起始L2中最多已有192MiB只读输入；仍受Cube、Vector及片上端口限制。冷场景一列增加外部输入首次供给。两者均没有将未知L2聚合带宽、完整依赖图和Scalar延迟伪装成已知，所以都是乐观资源口径。

| 档位 | 热起始资源下界 | 冷场景资源模型 | 实测÷热资源下界 | 实测与热下界之差 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 81.37 | 124.42～133.23 | 5.85× | 395.03 |
| 128K/B8 | 110.70 | 137.28～154.89 | 5.66× | 515.80 |
| 128K/B16 | 171.73 | 171.73～198.22 | 4.67× | 630.47 |
| 128K/B24 | 276.48 | 276.48～276.48 | 3.76× | 763.34 |
| 128K/B28 | 327.40 | 327.40～327.40 | 3.57× | 840.32 |
| 8K/B16 | 86.62 | 122.99～158.21 | 7.44× | 557.46 |
| 8K/B32 | 175.46 | 175.46～204.85 | 5.32× | 758.12 |

**最后一列只是相对乐观下界的差额，不是承诺可省的时间。** 例如不能据此声称802μs能直接优化到170μs；矩阵、排序、KV搬运要按依赖供给，短Tile启动、Scalar寻址、UB/L1冲突和同步都会占用时间。

此前B16/B28的Cube全24核平均为161.77/287.07μs；本表改用实际最忙AIC，所以更高。QB实际分给20核且logical=physical%20，KV另有4核循环置换；不是新增计算，也不是性能实测变了。15个矩阵阶段由独立循环模型交叉核对，总运算量完全相符。

## 4. Vector具体算了什么

共26个业务工作量分组，覆盖实际调用链的全部计算阶段：

- HC加宽、归约、sigmoid门控、20轮Sinkhorn、四路混合、输入RMS。
- QR归一化量化，主Query反量化/RMS/RoPE，原始KV归一化/RoPE。
- 两套压缩pool的max/exp/加权合并、state更新、norm/RoPE、compact量化。
- Indexer query反量化/RoPE/Hadamard后量化，head系数、scale处理、叶内/跨叶Top-K。
- 页连续性证明、选中页物理地址与有效块计划、softmax、在线归并及逆RoPE。
- O_A输出量化、八组O_B反量化左折叠、四路HCpost。

注意三项容易漏算的成本：归约不能直接按Add64元素/cycle算；带tmp的高精度TRSQRT在当前A3实现为vsqrt+vdiv；Sort32/MrgSort按(score,index)项计数，不能把1024个FP32暂存误当1024个排序项。短8K采用G2候选半叶分工，已单独建模，没有复用长G6按query分工的公式。
排序计量单位同时参照[官方Sort32/MrgSort接口说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0003.html)及[CANN DeepSeekV4的A3 Top-K理论周期示例](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/deepseek-v4/deepseek_v4_ascendc_operator_guide.md)。

以下是B16/B28各阶段最忙AIV的通用归约服务模型；各行可能是不同核且阶段可重叠，**不能相加成整层时延**。

| Vector阶段 | 128K/B16 | 128K/B28 |
| --- | ---: | ---: |
| HC加宽与RMS | 4.154 | 4.154 |
| HC投影归约 | 0.013 | 0.013 |
| HC门控 | 0.010 | 0.010 |
| HC混合与输入RMS | 4.603 | 4.603 |
| HC_Sinkhorn | 2.664 | 2.664 |
| QR归一化量化 | 0.944 | 0.944 |
| 公共RoPE准备 | 0.123 | 0.123 |
| 主Query反量化RMS_RoPE | 5.428 | 9.704 |
| 原始KV归一化RoPE | 1.110 | 1.110 |
| 主压缩pool_state | 0.480 | 0.480 |
| 主压缩norm_RoPE_cache | 0.317 | 0.317 |
| Indexer压缩pool_state | 0.120 | 0.120 |
| Indexer压缩norm_RoPE | 0.239 | 0.239 |
| Indexer压缩量化 | 0.013 | 0.013 |
| IndexerQuery反量化RoPE | 1.754 | 3.027 |
| IndexerQuery量化 | 1.876 | 3.751 |
| IndexerHead系数 | 0.069 | 0.069 |
| IndexerScale与排序准备 | 2.880 | 5.990 |
| Indexer叶内排序归并 | 79.706 | 146.543 |
| TopK末级归并发布 | 0.623 | 1.799 |
| 前置metadata与页证明 | 0.001 | 11.990 |
| Attention有效块计划 | 2.729 | 2.729 |
| AttentionSoftmax | 9.686 | 16.951 |
| Attention在线归并与逆RoPE | 11.346 | 19.736 |
| O_A输出量化 | 1.628 | 1.628 |
| O_B反量化与HCpost | 4.982 | 9.963 |

## 5. 内存不是只有HBM

大矩阵权重合计178.29MB，已接近192MiB L2容量；长历史key/scale、KV以及中间结果会进一步争用缓存。GM地址访问可能命中L2，不能全部除以1.6TB/s；也不能假定全部命中。
HBM账本已补上norm/scale/APE、旧state、RoPE与metadata；只统计实际访问区域，不按整张cache/state allocation大小收费。本轮生成的新KV不当作外部冷读。

语义GM账本统计当前必需的物化和直接消费，B16/B28分别约781.54/1231.50MB。它未包含后端重复Tile搬入、缓存行放大、信号和临时溢出，不能当精确动态事务量。
没有取得本次整层实际HBM/L2流量，因此对GM总服务率做以下敏感性分析：

| 档位 | 若GM总服务率2TB/s | 4TB/s | 7TB/s | 当前完整L1 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 164.71 | 82.35 | 81.37 | 476.40 |
| 128K/B8 | 240.40 | 120.20 | 110.70 | 626.50 |
| 128K/B16 | 390.77 | 195.38 | 171.73 | 802.20 |
| 128K/B24 | 540.82 | 276.48 | 276.48 | 1039.82 |
| 128K/B28 | 615.75 | 327.40 | 327.40 | 1167.72 |
| 8K/B16 | 343.42 | 171.71 | 98.12 | 644.08 |
| 8K/B32 | 598.12 | 299.06 | 175.46 | 933.58 |

表中已与Cube/Vector/片上端口取max；这些带宽不是本机实测值。7TB/s仅借用[官方192MB L2示例中的假设](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0036.html)，不能把该示例当成9392保证规格。[官方性能指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_00014.html)也要求把共享带宽上的读写合计，并提醒小DMA达不到峰值。

因此，B16/B28如果GM有效服务率仅2TB/s，当前数据流即使忽略等待也要约391/616μs；这比只看Cube算术更能说明为什么不能把全部差额看成调度损失。

## 6. 与真机阶段证据对照

同版本已有72核阶段探针给出：

| 阶段 | B16阶段区间 | B28阶段区间 | 资源解释 |
| --- | ---: | ---: | --- |
| Score AIC / AIV | 192.64 / 181.67 | 329.82 / 324.70 | Cube、排序、MTE1、FIX均需考虑 |
| Attention AIC / AIV | 153.47 / 158.41 | 304.19 / 304.77 | 计算工作小于数据交接及实际流水区间 |
| 主QB AIC | 60.57 | 88.91 | 20核列条带、权重供给和短矩阵效率 |

这些探针区间扣掉外层GM依赖等待，但仍含内部READY/ACK、DMA、Scalar和探针；它们不是纯引擎忙碌时间。旧独立Score PMU及修改过退避的模拟流水没有作为本轮整层实测。

Score最忙核的Cube下界B16/B28为91.33/170.21μs，仅排序归并约80～82/148～150μs，QK经FIX转L1再参与WS也受片上端口限制。两边算力接近，所以单独加速Cube可能转成Vector受限；不能忽略任一边。

Attention的Cube约22.14/38.75μs，Vector主要算术约21/37μs；但当前QK/PV/概率/KV环槽主要GM流量已达374.34/655.10MB。这支持优先审视数据交接、DMA粒度与流水重叠，不能仅根据Cube低利用率继续加大矩阵流水。

## 7. 能得出与不能得出的结论

能得出：完整CSA未逼近纯Cube/Vector算力资源上限；长历史Score已同时受到Cube、排序和片上转运约束；Attention有明显的数据交接与流水成本。当前核分工约束下，理论资源差额仍大，但它是乐观余量上限。

不能得出：某个精确微秒值已经可达，或还能确定省掉某个百分比。要把资源下界收紧为可达预算，需要同版本整层HBM/L2事务与有效带宽、Vector/Cube/FIX/MTE忙碌时间，以及未改通信语义的关键路径证据。本次没有因此擅自启动新一轮设备测试或优化。

## 8. 复现及文件

```bash
python tests/pypto_test/csa_five_way_latest_20261020/theory_roofline/full_roofline.py
```

- `full_roofline.py`：组合模型、源码覆盖核查、JSON/CSV/本文生成。
- `vector_model.py`：26个Vector工作分组、各类别吞吐、48核分配。
- `vector_sort_bound.py`：长G6与短G2逐叶排序计数。
- `memory_model.py`：24核矩阵/片上端口、冷热HBM及GM语义账本。
- `analyze.py` / `ANALYSIS.json`：15个Cube阶段的有用/执行运算量组件。
- `FULL_ROOFLINE.json`：七档资源摘要、假设、来源和详细组件文件引用。
- `FULL_ROOFLINE.csv`：七档扁平表，便于下载后比较。
- `VECTOR_MODEL.json` / `MEMORY_MODEL.json`：两个资源组件的独立导出。
- `SOURCE_COVERAGE.json`：37文件字节一致、共享RMS helper及可达函数审计。
- `INITIAL_CUBE_ONLY.md`：历史不完整版本，保留过程，禁止当最新结论。

核查包括15个矩阵阶段独立计数守恒、既有Score逐核块数真机对照、两种Top-K路径有效候选覆盖及Softmax/query/HCpost元素守恒。没有下载权重、提交二进制或生成hash清单。
