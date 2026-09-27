# CSA 七档 Native / PTO 核内差异与优化顺序

更新：2026-09-27。分析基线为统一 V10 性能版，算子源码 `0ed4f926`，七档汇总提交 `60c0ee63`。
本文件保存核内阶段的差异分析；新增候选必须另列实测结果，不能回填为 V10 基线。
当前保留实现包含 `21d99f8a` 的性能版B40 KV投影优化、第6.4节按工作量选择的KV提前发布，
以及第6.6节的跨query连续流水。
上述保留策略已完成同一源码 `da2e2368` 的七档单卡测量，见第6.7节；整模型验收未完成。
第2～4节用于核对七档基线，第5节列下一步，第6节区分已保留和已撤回的实验。
最新保留规则：incore task有明确收益且必要功能检查通过就保留，本体/长尾变化另记。
不以本体未改善为核内优化的撤回理由；基本可做的核内优化完成后转调度。
此前按本体否决的双query合并head规约已恢复并保留，见第9节，历史测量值保持不变。

## 1. 当前结论与边界

1. 128K Indexer 与 Native 在 query 复用、流式 Top-K、重复准备方面仍有差异。
   4＋2 query 复用已经试过并退化；合并head规约有核内收益，先前仅因本体未改善撤回，现已按用户新规则恢复并保留。
2. 8K 四档更直接的核内问题是稀疏注意力：PTO `qk_pv` 的 block 平均核内耗时，
   已超过对应 Native 整个 `SparseAttnSharedkv` kernel，PTO 后续另有 `merge_norm`。
3. V10 已采用 QK→FP16→第二次 Cube 做 head 加权规约，不能继续把旧 Vector 规约当作当前差异。
4. 本阶段先吸收 Native 的核内策略；任务融合、派发间隙和任务调度留到核内阶段之后。
   按 batch/长度选择策略时，选择逻辑放在同一套当前 PTO 算子里，不按档位切换历史版本。
5. 以下代码差异是真实存在的，但其耗时贡献尚未逐项实测，不能据此承诺优化百分比。
   当前数据也不足以把核内时间完整拆成纯 Cube 算术、DMA、同步等待各占多少。
   新增固定输入探针已定位8K/B40的主要核内等待边界，见第6.3节；不将发射区间当作纯算术耗时。

初始七档基线提取只读取已有源码和 trace；后续先导和 Native 缺口补采见第6节，没有做 hash 校验。
QKV、两个 Compressor、O projection、mHC 的操作对应及七档核内记录已补到第7节；
融合边界、量化位置和缓存写入范围不同，完整等范围归因仍未完成，不能直接按任务均值相减。

## 2. 环境、数据来源和计时口径

- A3 / CANN 9.0.0 / 当前隔离环境，正式 `DeepSeek-V4-Flash-0731-w8a8` 的 layer 4 权重，合成输入与历史。
- 单卡 S6、TP1、mode2、atomic1、确定性0、EPLB关闭，复用同一步第二个 CSA 层 metadata。
- 七档：128K B4/8/16；8K B16/24/32/40。PTO 七档均为同一套 V10 算子。
- Native 六档来自之前矩阵保留的 Native PyTorch profiling JSON，Native 实现未改。
  与 V10 的 PTO DFX 是独立采集，不是同一次调用；这些 trace 不冒充 V10 新采集数据。
- 128K/B16 已用相同 layer4/单卡/S6/TP1/mode2/确定性0 配置单独补采 Native 图重放，
  没有运行 PTO；本次 profile 只补分项，不替换原有20次总区间基线。

**Native**：`Ascend Hardware` 进程中完整融合 kernel 的设备耗时。
**PTO**：仅取 `Worker View` 的 `kernel-duration-us`，每个窗口分别对同名任务所有 block 求均值，
再报告四个窗口均值的最小值和最大值。表中范围不是单个 block 的最小/最大值，也不是 p50/p95。

PTO 核内耗时包含任务内部的数据搬运和同步等待，不包含执行开始前的派发排队。
Native 没有与之对应的逐 AIC/AIV block 分解，两列不能直接计算严格等范围的加速比。
AIC/AIV 并行、不同任务也可能交叠，不能把表格列相加得到 CSA 总区间。
`Score→publish` 是包含调度和交叠的区间，本文件不拿它替代核内耗时。
主性能验收仍使用无 profiler 的稳态设备计时。

来源入口：

- [统一 V10 七档总性能](results/csa_split_optimization_20260927/INDEXER_PROGRESS_V10.md)
- [V10 每档原始 timing / 四窗口泳道路径](results/csa_split_optimization_20260927/indexer_progress_v10.json)
- [Native 原始 profiling 下载目录清单](results/csa_native_cube_matrix_20260927/download/manifest.json)
- [补采的128K/B16 Native profiling JSON](results/csa_incore_20260927/native_h131072_b16/native_pytorch.json)
- [本次全部任务核内统计、block 数、单 block 最大值及原始路径](results/csa_incore_20260927/v10_incore.json)
- [离线提取脚本](results/csa_incore_20260927/summarize_v10.py)，只读已有 JSON，不触发设备执行。

### 2.1 七档本体总区间对照

单位：μs；5次预热后20次无 profiler 设备计时的均值。
本体范围为 HC_pre → norm → CSA → HC_post；完整 PTO 另含入口拆分与更新后写回。
各区间独立计时，不能用各阶段均值精确相加重建完整路径。

| 上下文 / B | 同轮 Native | V10 PTO 本体 | 本体对 Native | V10 完整 PTO |
| --- | ---: | ---: | ---: | ---: |
| 128K / 4 | 865.14 | 716.34 | −17.20% | 996.09 |
| 128K / 8 | 1006.47 | 889.22 | −11.65% | 1210.00 |
| 128K / 16 | 1317.00 | 1304.87 | −0.92% | 1709.66 |
| 8K / 16 | 941.46 | 790.88 | −15.99% | 1085.69 |
| 8K / 24 | 1134.60 | 1026.25 | −9.55% | 1341.13 |
| 8K / 32 | 1283.66 | 1196.32 | −6.80% | 1516.50 |
| 8K / 40 | 1405.57 | 1428.20 | +1.61% | 1771.34 |

这些是同一 V10 源码的七档实测，原始来源见上面的统一 V10 报告。
128K/B16 本体 p95 为1577.50 μs，存在长尾；完整 PTO 七档均慢于 Native。
当前先优化本体，不能据此忽略完整路径成本，也不能用第6节的新 B40 结果替换这一行后称为新七档。

## 3. Indexer：七档核内数据

单位：μs。Score 每窗口包含 24 个 AIC block、48 个 AIV block；独立 Top-K 合并为 48 个 AIV block。
Score 任务包含局部排序，不能将其视为只有矩阵乘法。

| 上下文 / B | Native 整个 QLI | PTO Score AIC | PTO Score AIV | PTO Top-K 合并 AIV |
| --- | ---: | ---: | ---: | ---: |
| 128K / 4 | 240.18 | 89.42–95.85 | 97.10–103.86 | 8.61–9.55 |
| 128K / 8 | 237.26 | 178.65–184.16 | 188.09–193.69 | 14.37–15.28 |
| 128K / 16 | 360.28 | 356.79–367.98 | 366.41–377.34 | 17.69–18.44 |
| 8K / 16 | 56.10 | 26.30–36.59 | 30.76–40.90 | 8.94–9.78 |
| 8K / 24 | 56.34 | 38.61–47.22 | 37.40–51.40 | 11.62–11.79 |
| 8K / 32 | 91.68 | 47.19–59.23 | 51.00–63.20 | 11.12–11.73 |
| 8K / 40 | 95.44 | 73.98–74.56 | 71.03–78.57 | 12.25–13.54 |

PTO 系数准备另有 48 个 AIV block，其四窗口 block 均值范围依次为：
2.01–5.51、2.96–7.03、2.44–4.64、4.16–10.01、4.89–11.11、2.77–4.46、3.54–11.41 μs。
Native 的系数准备在 QLI 内，不是表外额外启动的任务。

### 3.1 已确认的源码差异

| 项目 | Native | 当前 PTO V10 | 核内优化方向 |
| --- | --- | --- | --- |
| 六个 query 共享 Key | 4＋2 分组，同一 Key panel 服务更多 query；M256 分两个 M128 计算 | 2＋2＋2 分组，M128 QK | 增加组内 query 复用，减少重复 Key 加载；保留小 batch 并行度 |
| Query / 权重准备 | 跨多个 S2 块保留复用 | 每个最多 8192 候选的 leaf 重新准备 | 延长片上驻留，减少重复准备 |
| Top-K | 每批 2048 候选，在 UB 中持续维护 Top-512 | 两个 AIV 按候选半区分工，每个半区最多 4096 候选；局部排序结果写 GM，再由独立任务合并 | 借鉴流式 Top-512，减少排序规模及 GM 中间结果 |
| 尾块 | 按实际长度和对应对齐规则处理 | 768 候选步长中有固定 panel 的 padding 计算 | 减少无效矩阵计算；覆盖 padding/尾块而非只测满块 |
| Score 算法 | QK 经 FIXPIPE 转 FP16 到 L1，再用 Cube 做 head 加权规约 | 已采用相同主路线，并有双 query 复用 | 已对齐部分保留，不再用旧 Vector Score 解释当前耗时 |
| 加权结果写回 | 可将同组 query 结果一次批量 FIXPIPE 写出 | 两次独立的单行结果写回 | 可核对指令和流水开销，尚无独立收益数据 |

Native 第二次加权矩阵乘的 M 维也按 Cube 的 16 对齐，不能解释成 Native 只算 M1、PTO 算 M16。
Native 两个 AIV 按 query 分工，PTO 两个 AIV 按候选半区分工；不能仅按 AIV 数量判断工作量相同。

128K 压缩后 32768 行，Key 每行 128 个 INT8：一遍逻辑读取为 4 MiB。
六个 query 下，PTO 2＋2＋2 对应约 12 MiB/请求，Native 4＋2 对应约 8 MiB/请求。
这是忽略 padding 和缓存命中的逻辑加载量，**不是测得的 DDR 流量，不代表时间必然减少三分之一**。

源码：

- [PTO decode_indexer.py](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py)：`indexer_score_topk_native_cube`、8192 leaf、768 步长、两 query 复用及分派条件。
- [Native QLI kernel](../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_kernel.h)：M256 / S2 2048。
- [Native QLI Cube](../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_service_cube.h)：`ComputeMm1`、`ProcessQk`、`ProcessWs`，Key 复用和批量写回。
- [Native QLI Vector](../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_service_vector.h)：`ProcessVec0` 系数、`ProcessVec1` / `MergeSort` 流式 Top-K。
- [前一阶段 Indexer 专项分析](DSV4_FLASH_CSA_INDEXER_NATIVE_GAP.md)。旧 V7 数据保留其历史范围。

## 4. Sparse Attention：七档核内数据

单位：μs。`qk_pv` 每窗口 24 AIC＋48 AIV block；`merge_norm` 为 48 AIV block。

| 上下文 / B | Native 整个 Sparse Attention | PTO qk_pv AIC | PTO qk_pv AIV | PTO merge_norm AIV |
| --- | ---: | ---: | ---: | ---: |
| 128K / 4 | 53.88 | 41.11–50.13 | 43.22–52.09 | 16.35–18.00 |
| 128K / 8 | 100.50 | 94.39–100.14 | 96.52–102.11 | 17.46–18.53 |
| 128K / 16 | 181.08 | 170.48–181.13 | 172.36–183.01 | 20.58–21.09 |
| 8K / 16 | 107.04 | 124.04–137.00 | 126.12–139.03 | 20.92–22.56 |
| 8K / 24 | 166.90 | 200.22–209.62 | 202.12–211.70 | 29.43–30.13 |
| 8K / 32 | 231.78 | 257.64–268.61 | 259.63–270.60 | 35.35–35.77 |
| 8K / 40 | 290.56 | 321.62–336.55 | 323.49–338.48 | 42.34–42.86 |

8K 四档，仅 PTO `qk_pv` 的核内 block 均值就超过 Native 完整稀疏注意力 kernel。
这支持优先优化其核内实现；不能把全部剩余差距统一归因于 Indexer 或调度。
但 `merge_norm` 还包含逆 RoPE 和输出布局整理，不能把它全部算成 Native 没有的额外归约成本。
这些平均值也不能相加后与 Native 相减，声称得到严格的临界路径节省量。

### 4.1 已确认的源码与生成代码差异

| 项目 | Native | 当前 PTO V10 | 含义与待验证点 |
| --- | --- | --- | --- |
| PV 输出维度切分 | N 按 128 切块；L0C 分两个 64 KiB 槽双缓冲 | 64×512 FP32 累加区，占 128 KiB；已核对生成 C++ | 先试 N128 分块，检查编译后的 L0 分配、K 分块和 Cube/FIXPIPE 流水；收益未知 |
| 稀疏 KV 搬运 | 成对处理索引，条件允许时合并 DMA；UB 分批流水搬运 | 压缩 KV 逐行读索引、查页表、gather_row | 减少逐行地址处理和小搬运；Native 在非法索引/跨度等情况下也会回退单行，不能假设永远一次 DMA 两行 |
| Softmax 分块 | S2 按 512 处理，维护累计最大值和归约状态 | 128 候选一块，块内 BF16 量化后再重标定合并 | 改变此项涉及舍入节奏与精度策略，不能与数值中性的输出维度分块混为一项 |
| QK/PV 的 KV 复用 | 工作区搬运结合片上流水 | 已在 QK/PV 间复用同一份 L1 KV | 不能再声称 PTO 为 QK/PV 分别从 GM 读取两遍 KV；Native 也使用工作区，并非零拷贝 |
| query之间的流水 | gloop跨batch/gS1区间延续，只在本核分配范围末尾追加排空轮次 | 每个query重新执行5个候选块，AIC/AIV另有2/3轮排空，再开始下一个query | 优先试跨query延续核内流水；须维持三槽复用及每query独立归约，尚无收益实测 |
| 结果处理 | Attention 输出处理在融合 kernel 内 | 后续独立 merge_norm 完成分母、输出、逆 RoPE 与布局整理 | 本阶段先优化核内，跨任务融合放到后续调度阶段 |

PTO 生成代码中的 PV 为 `Tile<TileType::Acc, float, 64, 512, ...>`，
对应 L0B 的 tile 为 `32×512`，K128 被拆成四段 K32；这比只读高层 `pl.matmul` 更明确。
Native PV 的 N128 与其 QK 使用的 D/K 分块不是同一概念，不能把 `D_SPLIT_SIZE=256` 当成 PV 的 N。
L0C 占用差异与耗时之间目前仍是优化假设；没有计数器证据证明它贡献了多少 stall。

源码：

- [PTO decode_sparse_attn_csa.py](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_sparse_attn_csa.py)：`qk_pv` 内 QK/PV、compressed gather、softmax 与结果合并。
- [V10 B40 生成的 qk_pv AIC C++](results/csa_split_optimization_20260927/v10_short_followup/h8192_b40/swimlane/build_output/_jit__decode_csa_tp1_layer_adevsbj_/kernels/aic/qk_pv_aic.cpp)：PV 累加区和 L0B/K32 分块；本地编译产物未承诺随 Git 保存，可由原始脚本重建。
- [Native Sparse Attention Cube](../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_cube.h)：`N_SPLIT_SIZE`、`L0C_PP_SIZE`、`ComputeMm2`。
- [Native Sparse Attention Vector](../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_vector.h)：`CopyInKv`、`ProcessVec0L`、`SoftmaxFlashV2Compute`。
- [Native tiling](../../csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.h)：`sInnerSize_=512`。

### 4.2 B40 的 KV 投影遗留精度特例

V10 性能版仍在 `T=240` 时调用 `kv_project_native_240`：N32/K64、16个block、
按 Native 的列组规则重排 K256 遍历，用于追求精度对齐。其他形状使用 N128/K256、
部署配置下 split-K=8。这个特例在性能版中也绕过了既有的较宽 tile/split-K 路径。
8K/B40 的 V10 四窗口该任务 block 均值90.05–103.11 μs，明显值得独立处理。

Native 双流调用顺序对应的 KV `MatMulWeightNz` 设备事件为26.26 μs；
这一映射依据 `_mla_prolog_multistream` 的 Q_A→KV 分工及 trace 的 stream/顺序，
不是逐block的等范围比较，不能由两数直接算加速比。
此外 Native 该权重使用 NZ，而 PTO `wkv` 根入参仍是 ND `[D,512]`，
在初始化时由共享 adapter 解包/转置一次；这不是每次 CSA 的权重重排。
性能版的240行精度特例已在 `21d99f8a` 去除，独立收益见第6节；
精度版原路径保留，WKV 的 NZ 复用仍为后续项。

源码：[性能版 QKV](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/qkv_proj_rope.py)、
[Native 双流实现](../../vllm_ascend/attention/dsa_v1.py)、
[权重适配](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/native_adapter.py)。

## 5. 优化顺序与必要验证

| 档位 / 阶段 | 核内工作 | 验证与保留条件 |
| --- | --- | --- |
| 8K B16/24/32/40，先用 B40 代表 | 已保留跨query流水；KV/gate合并投影先导退化已撤回；后续优先检查QKV额外准备及Vector处理 | 隔离核内收益，先CPU编译再代表档；不重复已失败候选 |
| 128K B8/B16 | 优先恢复有核内收益的合并head规约，再评估流式Top-K、尾块及重复准备；4＋2复用核内退化仍撤回 | 核内获益即保留，本体变化另记；只补当前组合的必要代表档 |
| 8K B40 | 已保留KV投影宽tile/split-K；继续评估复用Native NZ权重 | 新改动仍需隔离收益；精度版保持原算术 |
| 128K B4 | 保留 V10 小 batch 完整 leaf 均衡 | 扩大 query 复用组后检查并行度和退化，必要时在算子内部按输入选策略 |
| 核内阶段出口 | 汇总同一当前实现的七档，保留其真实策略与源码、误差及性能结果 | 不用历史版本拼出最优七档；不能只看正常窗口或中位数忽略长尾 |
| 随后调度阶段 | 任务融合、任务数量、派发/依赖间隙 | 先记录核内阶段结果，再评价调度收益，避免混淆归因 |

精度版保持现有 Native 对齐算术。性能版新策略最终仍需整模型逐 token / DSpark 看护；
metadata、整数索引、保护区和非有限值属于功能约束，不因性能优先而放宽。
不做过量 hash、重复测试或无关全量回归；优先单卡，必要时才推进真实权重 16 卡。

## 6. 当前实验状态

PV N128逐块写回、PV N128两块同时存活、Top-K索引/页表预读三个候选均完成8K/B40先导，
没有明确本体收益，已撤回，详见[稀疏注意力实验记录](results/csa_incore_20260927/SPARSE_ATTENTION_PROGRESS.md)。
其中真正双缓冲只有约2.35%的DFX核内均值变化，区间重叠且本体未改善，不认定稳定收益。
上面的七档表格仍是V10基线，未被候选结果覆盖。

Native满128窗口＋512 compressed候选时为两段PV结果，PTO128分块为五段。
每段64×512 FP32写回，逻辑中间结果分别约256/640 KiB/query；这比单纯N分块更值得联合流水分析，
但目前没有设备收益证据，不能按逻辑字节差推算加速比。
Indexer的4＋2 query Key复用已做128K/B16先导：工作量配平版和进一步L0驻留版均退化，已撤回。
[候选结果与限制](results/csa_incore_20260927/indexer_group4_balanced/README.md)保留实际证据；
不能把逻辑Key读取量减少当成性能收益。
随后[双query合并head规约](results/csa_incore_20260927/indexer_fused_ws/README.md)
在128K两档及8K/B40下降了局部Score核内时间，历史上因四档CSA本体未改善撤回；
该撤回依据已被用户新规则修正，正在恢复评估，不再以本体耗时否定核内收益。
[全有效KV省去UB清零](results/csa_incore_20260927/kv_valid_nozero/README.md)亦没有足够稳定收益，不进入正式源码。

128K/B16 Native分项已补齐，QLI=360.28、Sparse Attention=181.08 μs。
可见V10的Score/qk_pv核内均值已经接近这两个Native完整kernel，但PTO仍有额外准备/合并，
这些数据不能支持“128K/B16所有差距都来自纯矩阵计算”的判断。

已保留[性能版B40 KV投影统一宽tile/split-K](results/csa_incore_20260927/kv240_splitk_only/README.md)：
撤回其他候选后独立验证，8K/B40本体由V10的1428.20降至1386.00 μs（−2.95%），
比同轮Native1410.61 μs低1.74%。KV投影block均值90.05–103.11→11.32–12.00 μs，
block数16→32，累计核内工作量1440.86–1649.72→362.22–384.04核·μs。
这是性能版对既有精度特例的清理，精度版原路径保留。完整PTO1736.41 μs仍慢于Native。
本页七档表仍是V10基线，不能将这一档的新结果拼成“新版本七档”。

### 6.1 联合 softmax / PV 累加候选：均未保留

两项均基于 `21d99f8a`，仅改性能版 Sparse Attention；保持24个AIC/48个AIV任务，
没有调整跨任务调度。复用8K/B40代表档、5次预热/20次计时及4个独立DFX窗口。
候选把完整128窗口＋512 compressed候选统一到一次softmax，五段PV在L0C内累计后写出最终结果。
这样减少部分结果写回，但需要为QK与PV分别读取工作区KV，且改变BF16量化时机；
这不是对Native累计512分段softmax的等价复现。

| 实现 | qk_pv AIC四窗口block均值范围 μs | CSA本体均值 μs | 同轮Native均值 μs | 处理 |
| --- | ---: | ---: | ---: | --- |
| 当前保留的V10＋KV投影优化 | 319.81–334.24 | 1386.00 | 1410.61 | 基底 |
| 联合softmax＋N512 PV累加 | 332.92–347.37 | 1392.09 | 1408.96 | 未改善，撤回 |
| 联合softmax＋N128双累加器 | 361.14–388.00 | 1413.71 | 1425.47 | 核内退化，撤回 |

第二项在每个N256范围内同时保留两个N128累加器，并沿五个K128块累计；
实测说明减少写回量、缩小输出tile都不足以证明收益，流水和搬运代价仍需定位。
不同采集轮次有波动，不把本体均值的微小变化认定为稳定收益。

两项保护区失败数0、Top-K结构错误0、输出非有限值0，max_abs均为0.03125，
RMSE分别为0.003291202/0.003291794；Top-K集合替换分别900/901。
零容差仍FAIL，没有通过整模型token/DSpark验收。
CPU编译及单卡任务均成功；两个候选均已撤回，未扩测其他六档。

证据：[N512数据及四窗口路径](results/csa_incore_20260927/sparse_joint_softmax/report.json)、
[N512候选补丁](results/csa_incore_20260927/sparse_joint_softmax/candidate.patch)、
[N128数据及四窗口路径](results/csa_incore_20260927/sparse_joint_n128/report.json)、
[N128候选补丁](results/csa_incore_20260927/sparse_joint_n128/candidate.patch)。

### 6.2 KV搬运先导与PMU定位

[16行UB双缓冲先导](results/csa_incore_20260927/sparse_gather16_pipeline/README.md)
只迁移Native的分批搬运，保留候选顺序和attention算术；8K/B40本体1386.11 μs，
基底1386.00 μs，qk_pv AIC范围322.44–329.31 μs与基底重叠，已撤回。
生成代码确实使用两个不同UB地址，但尚未实现Native动态行距的成对DMA。

随后用[固定Native输入的独立Sparse Attention PMU](results/csa_incore_20260927/sparse_pmu/README.md)
取得24个AIC/48个AIV记录：qk_pv AIC Cube busy=21.23%、MTE1=24.87%、MTE2=22.56%；
AIV Vector busy=33.97%、MTE2=35.41%、MTE3=14.71%。各流水交叠，不能相加。
Scalar busy分别55.35%/47.83%，不能直接归因为地址计算或跨任务调度。
该独立program诊断不替代完整CSA的kernel-mode稳态计时；它支持继续定位核内流水串行和等待，
不足以量化各段可节省时间。随后沿四类ready事件的已有同步边界拆解，结果见第6.3节。

### 6.3 核内等待定位：KV发布晚于上一块softmax

复用同一份8K/B40 Native输入，在基底生成的C++沿已有wait/sync边界读取系统计数器。
没有增加pipeline barrier；24个AIC和48个AIV记录完整，输出与未插桩PTO逐bit一致。
各核平均区间如下，单位μs；这是独立program诊断，不替代无profiler本体计时。

| 核 | 主要区间 | 均值 | 占本核测量区间 |
| --- | --- | ---: | ---: |
| AIC | 等KV-ready | 208.00 | 67.81% |
| AIC | 等Prob-ready | 15.04 | 4.90% |
| AIC | QK发射及原有排空 | 52.01 | 16.96% |
| AIC | PV发射及原有排空 | 25.86 | 8.43% |
| AIV | gather发射及原有排空 | 124.30 | 41.25% |
| AIV | 等Score-ready | 78.94 | 26.20% |
| AIV | softmax发射及原有排空 | 24.74 | 8.21% |
| AIV | 等PV-ready | 63.71 | 21.14% |

AIC/AIV并行，表格不能相加；发射/排空区间不是对应流水独占时间，
KV等待也不全是可消除开销。测量支持优先处理核内搬运和事件衔接，不支持把差距全归给跨任务调度。

基底PTO在当前KV搬完后，还要等上一块Score并完成softmax/Prob发布，才通知AIC读取当前KV。
Native `PreloadPipeline` 在 `ProcessVec0L` 后就发布 `syncV0C1`，随后执行上一轮 `ProcessVec1L`。
因此下一步是保留上一块Score通知的消费顺序，但提前KV通知，使当前QK与上一块softmax交叠。
不改变任务数、跨任务调度和attention算术。

证据：[探针原理、完整分解和复现](results/csa_incore_20260927/sparse_phase_probe/README.md)、
[区间汇总](results/csa_incore_20260927/sparse_phase_probe/summary.json)、
[Native流水实现](../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_kernel.h)。

### 6.4 KV提前发布先导：较大工作量获益，小工作量保留原顺序

只将KV-ready移到上一块Score通知消费之后、softmax之前。三槽缓冲、事件次数、
矩阵计算和attention算术不变。先全档启用，按代表档证据确定工作量选择。

| H / B | 参考本体 μs | 先导本体 μs | 参考qk_pv AIC μs | 先导qk_pv AIC μs | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| 8K / 16 | 790.88 | 796.17 | 124.04–137.00 | 127.59–135.97 | 无明确收益 |
| 128K / 16 | 1304.87 | 1430.40 | 170.48–181.13 | 160.09–178.02 | 核内范围重叠，本体长尾更多 |
| 8K / 24 | 1026.25 | 1020.42 | 200.22–209.62 | 190.83–197.31 | 核内下降，本体−0.57%仍很小 |
| 8K / 32 | 1196.32 | 1178.79 | 257.64–268.61 | 244.09–254.23 | 核内下降，本体−1.47% |
| 8K / 40 | 1386.00 | 1357.36 | 319.81–334.24 | 305.92–318.07 | 核内下降，本体−2.07% |

B40参考是已有KV投影优化，其余参考为V10；各轮独立采集。
128K/B16的20个样本中超过1500μs的次数由5次增至12次，不能只取正常窗口宣称获益。
五档完整PTO均仍慢于同轮Native；本表不替代第2节七档基线或最终七档验收。

当前已保留实现按 `T >= 24×6` 选择提前KV发布（S6时B≥24），较小工作量保留原顺序。
阈值是本轮先导的保守选择，不是硬件固有规则；不引入按档位切换历史源码的脚本。
固定Native输入的8K/B40 Sparse输出与基底逐bit一致，B3短历史尾块解析用例通过。
五档保护区和索引结构通过，非有限值0；对Native的既存浮点差异仍在，未做整模型验收。

加入选择逻辑后单独复验：8K/B40本体1361.40 μs（改动前1386.00，下降1.77%），
qk_pv AIC范围301.01–318.44 μs；同轮Native1401.18 μs，完整PTO1708.50 μs仍更慢。
8K/B16保留原顺序，本体796.12 μs，无明确收益。固定输入及尾块检查仍通过。
带选择逻辑的剩余档位尚未测量，不能用上面的先导值拼成最终七档。

[完整先导、工作量选择及最终复验证据](results/csa_incore_20260927/sparse_kv_early/README.md)、
[五档误差/计时及原始泳道路径](results/csa_incore_20260927/sparse_kv_early/cases.json)。

### 6.5 成对DMA与连续query遍历先导：未加入生产

Native成对DMA可在跨度合法时将两条GM搬运合为一条；最新PyPTO main `b046b15c` 的
`gather_row` 仍未提供动态源行距。本轮先改本地生成代码，未改工具链或生产算子。
固定8K/B40输入的61,440对全部可合并，DMA指令数减半，但逻辑读取量仍是120MiB。
独立program的qk_pv AIC平均cycles：当前513120.83、成对DMA502897.08（−1.99%）、
成对DMA加16行分批写出498027.46（−2.94%）。每项只有一次PMU采集，不能证明稳态本体收益。
两项与当前PTO均仅10个元素不同，max_abs=0.000244140625；对Native最大误差不变、非有限值0。
候选对内按物理地址重排影响舍入，并非数值中性改动；暂不扩大接口实现或七档测试。
[实现、限制、原始计数器与误差](results/csa_incore_20260927/sparse_pair_dma/README.md)。

另将每核query由跨24行跳读改成均衡连续区间，任务数和依赖不变；固定输入和B3尾块逐bit检查通过。
8K/B40本体1361.40→1349.62 μs（−0.87%），qk_pv AIC范围301.01–318.44→301.89–309.44 μs，
两轮范围重叠；merge_norm由39.66–40.70变为38.08–39.08 μs。
未认定稳定核内与本体收益，候选已撤回，未扩测。
[连续遍历先导与原始泳道](results/csa_incore_20260927/sparse_query_contiguous/README.md)。

进一步源码核对发现：Native的 `gloop` 在batch/gS1循环外递增，`extraLoop` 只在 `isEnd` 时为2；
PTO则每个query都排空QK/PV流水。这是接下来优先验证的核内差异。
实现时需让槽位编号随总候选块递增，分别定位当前KV、上一块softmax和更早PV所属的query；
最终PV合并后发布该query输出并重置归约状态，不能让相邻query共用数值状态。
全无效块、非24整除尾部及最后一个query的排空必须保留，跨任务调度仍不改。

### 6.6 跨query延续流水：B40核内下降，B16未证明稳定收益

参考Native仅在本核范围末尾排空，PTO改为每核连续处理所有query的候选块；
三槽编号按全局工作项递增，各阶段分别定位其query，在最后一块PV合并后发布并重置该query状态。
任务数、跨任务依赖、每query的量化/归约算术保持不变，精度版不动。

| H / B | 参考本体 μs | 当前本体 μs | 当前p95 μs | 参考qk_pv AIC μs | 当前qk_pv AIC μs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K / 40 | 1361.40 | 1342.68 | 1401.54 | 301.01–318.44 | 283.06–288.68 |
| 8K / 16 | 796.12 | 795.04 | 813.26 | 124.04–137.00 | 122.51–125.71 |
| 128K / 16 | 1304.87 | 1269.53 | 1549.14 | 170.48–181.13 | 172.52–180.50 |

B40相对上一保留实现，本体−1.38%、核内范围明显下降，改动保留。
B16本体参考分别为最终早通知分派版/V10，核内参考均为V10；独立轮次不能冒充同时采样。
8K/B16基本持平，128K/B16核内范围重叠且长尾仍在，不能认定这两档稳定获益。
两档B16的merge_norm分别24.55–25.26 / 23.68–23.78 μs，高于V10，不能只看qk_pv。
完整PTO三档仍慢于同轮Native；保护区和索引结构通过、非有限值0，Native零容差仍FAIL。

CPU完整编译通过；固定Native输入的7,864,320个Sparse输出元素与基底PTO逐bit一致。
B9有效→全无效→有效、不均分query尾部，以及B3零工作量核的解析检查通过。
首版重用初始sink tile被循环状态别名覆盖，已改为每次重置重新加载sink；该问题按功能错误修复，未放宽容差。
此处是最初三档代表验证，随后同一源码的完整七档见第6.7节；不回填第2节V10基线。
[实现、初版定位、逐项误差和原始泳道](results/csa_incore_20260927/sparse_cross_query/README.md)。

### 6.7 保留策略的同一源码七档复测

`da2e2368` 七档已收齐，5次预热/20次无profiler计时，每档4个DFX窗口。
当前本体按128K B4/B8/B16、8K B16/B24/B32/B40顺序为
729.21 / 891.92 / 1269.53 / 795.04 / 1002.28 / 1140.71 / 1342.68 μs。
对V10依次 +1.80% / +0.30% / −2.71% / +0.53% / −2.33% / −4.65% / −5.99%。
这是核内阶段累计保留改动的变化，不能全部归给跨query一项。

8K/B32 qk_pv AIC为237.70–245.66 μs，V10为257.64–268.61；B40为283.06–288.68，
V10为321.62–336.55。小档位没有本体收益，128K/B16 p95仍1549.14 μs。
当前本体七档均低于各自同轮Native，完整PTO七档仍更慢；拆分和写回成本没有消除。
保护区、Top-K结构、非有限值检查通过，Native零容差仍FAIL；未做新的整模型token/DSpark验收。
[完整七档表、p95与功能/数值](results/csa_incore_20260927/sparse_cross_query/MATRIX.md)、
[全部原始计时与28个泳道路径](results/csa_incore_20260927/sparse_cross_query/cases.json)。

仍缺：其他CSA模块的完整等范围临界路径归因、后续核内候选的独立验证，
以及保留策略的整模型验收。核内阶段未完成，跨任务调度暂不改。

## 7. 其他CSA模块：补齐对应范围与可吸收的策略

已按Native `_mla_prolog_multistream`、`cv_indexer_select_qli`、`_forward_decode`、
`_forward_o_proj` 的调用顺序和trace stream/task id，将七档16组操作逐项对应。
PTO根算子先调用Attention Compressor、再调用Indexer Compressor，因此本批V10 trace中
`kv_score_proj` 对应Attention，`kv_score_proj_0` 对应Indexer；Native两个Compressor的执行顺序恰好相反。
不能按名称相同或在trace中的第几个位置直接把两侧配对。

[七档16组原始统计及对应关系](results/csa_incore_20260927/v10_other_incore.json)，
[只读提取脚本](results/csa_incore_20260927/map_other_tasks.py)。下面仅用8K/B40展开示例，仍为V10基线。
每个分号隔开的数值是不同任务，单位μs；PTO列是四窗口block均值范围，**不求和**。

| 模块 | Native完整kernel | PTO核内任务 | 实现差异 / 解释边界 |
| --- | --- | --- | --- |
| HC_pre | 82.86 | widen 9.61–9.81；RMS 9.47–9.81；linear 14.95–17.02；linear reduce 2.33–3.57；Sinkhorn 16.81–17.77；split 4.89–6.29 | Native融合，PTO有BF16→FP32加宽和多个独立任务；后续融合属于调度阶段 |
| 输入RMSNorm | 15.04 | mix_x_rms_norm 16.64–19.11 | 值得检查向量实现；单block均值不是完整kernel尾部，不能直接得出百分比 |
| Q_A | 20.54 | seed 22.92–26.50；matmul 8.41–9.27 | Native NZ matmul；PTO NZ split-K需要清零及atomic归约，不能只看matmul宣称更快 |
| Q_A RMS＋量化 | 20.24 | 10.69–11.05 | 任务工作量及布局仍需结合生成代码解释 |
| KV投影 | 26.26 | 90.05–103.11，另有seed 11.76–12.60 | V10的B40精度特例；已在第6节独立移除，不能当作当前残留差距 |
| KV RMS＋RoPE | RMS 23.12；RoPE 19.26 | 融合8.41–9.38 | Native后面另有scatter；PTO cache写入也另有任务，此处不含缓存成本 |
| Q_B＋反量化＋RMS＋RoPE | quant matmul 85.42；RMS 26.96；RoPE 16.02 | matmul 67.90–84.17；dequant/RMS/RoPE 52.09–58.39 | Native quant matmul包含输出缩放/舍入；PTO先写INT32至GM，再由Vector反量化，不能按matmul单列直接对比 |
| Indexer Compressor | 70.18 | 投影23.24–25.70；pool 8.57–9.69；state commit 5.94–10.82；RMS/RoPE 19.42–22.40 | Native融合范围不含后续Hadamard/量化/scatter；PTO也不能只取投影当整个Compressor |
| Attention Compressor | 104.24 | 投影31.46–33.99；pool 7.20–12.79；state commit 6.81–9.39；RMS/RoPE/cache write 8.77–15.32 | PTO最后一项含scatter，Native对应scatter在Compressor之外；不存在可直接相减的单列 |
| Indexer Q投影＋RoPE | quant matmul 28.20；RoPE 20.42 | matmul 22.70–26.95；dequant/RoPE 29.38–45.53 | 与Q_B类似，输出缩放的融合边界不同 |
| O_A | TransposeBatchMatMul 138.46 | proj_a_mm 83.74–85.91 | 当前CANN Native无WeightNz入口，保留ND；PTO使用NZ，不能描述成两侧都走相同NZ路径 |
| O_B | quant matmul 76.86 | quant 6.79–7.73；NZ matmul 20.92–21.91；act 15.98–16.06 | Native列不含其独立动态量化，PTO列含量化/反量化；无法仅据单项认定完整区间优势 |
| HC_post | 34.26 | 24.95–27.00 | 输出范围相近，仍保持完整kernel与block均值的口径区别 |

在本阶段继续吸收Native的核内实现，新增一项有明确源码依据的候选：
`compressor_block_cube_perf.h::CopyWeightGmToL1` 将wkv、wgate写入同一L1 tile的不同列组，
`ComputeMm1` 用 `nDealSize=2*dBaseSize` 的一次Mmad计算两路，再分开Fixpipe写回。
当前两个PTO Compressor在每个K512块中各调用一次KV matmul、一次gate matmul。
可尝试在片上拼接两路权重并扩大N，用一次矩阵乘复用同一X加载，保留现有任务数和依赖。
这是指令/流水策略差异，逻辑FLOPs和权重字节没有减半。
随后B40先导完成：Attention投影33.21–38.00→42.95–44.40 μs，Indexer投影23.36–25.52→29.48–35.95 μs，
均退化；本体1342.68→1340.21 μs只有约2.5μs变化，不认定有效，候选已撤回。
生成代码显示扩大N同时将L0B的K分块减半，不能按高层matmul调用数预测硬件收益。
[编译处理、补丁及实测证据](results/csa_incore_20260927/compressor_combined/README.md)。

源码：[Native Compressor Cube](../../csrc/attention/compressor/op_kernel/arch32/compressor_block_cube_perf.h)、
[PTO Attention Compressor](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_compressor_ratio4.py)、
[PTO Indexer Compressor](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer_compressor.py)。

仍需单列metadata、cache scatter、布局/seed/adapter的临界路径范围；
此节补齐的是操作对应和七档数据入口，没有把分散任务的均值拼成完整CSA归因。

## 8. 七档基线的突出差异（da2e2368，后续变化见第9节）

重新只读刚完成的七档28个DFX窗口，得到[当前全部任务统计](results/csa_incore_20260927/current_incore_da2e2368.json)。
这里PTO为当前采集，Native分项沿用已标注的独立profile；Native完整kernel与PTO block均值不能等范围相减。

| 模块 / 档位 | Native完整kernel μs | 当前PTO核内 μs | 当前判断 |
| --- | ---: | --- | --- |
| Sparse，8K/B16 | 107.04 | qk_pv AIC 122.51–125.71；merge_norm 24.55–25.26 | qk_pv自身仍偏慢，不能全归调度 |
| Sparse，8K/B24 | 166.90 | qk_pv AIC 196.01–200.75；merge_norm 30.20–30.53 | 仍是明确核内差距 |
| Sparse，8K/B32 | 231.78 | qk_pv AIC 237.70–245.66；merge_norm 32.65–33.28 | 核内已接近，后续处理仍额外存在 |
| Sparse，8K/B40 | 290.56 | qk_pv AIC 283.06–288.68；merge_norm 41.58–41.86 | 主QK/PV接近；merge还含逆RoPE/布局，不能全算纯规约开销 |
| Indexer，128K/B16 | 360.28 | Score AIC 354.44–363.56、AIV 363.88–372.93；Top-K merge 17.64–18.70 | Score已占接近Native全QLI的量级，仍有准备/合并任务 |
| Q_A，8K/B40 | matmul 20.54 | seed 25.86–27.32；matmul 8.56–9.38 | PTO split-K清零/归约表达的额外工作值得看，不能只报matmul |
| Q_B，8K/B40 | quant matmul 85.42；RMS 26.96；RoPE 16.02 | matmul 64.86–75.78；dequant/RMS/RoPE 50.47–55.49 | Native在quant matmul内缩放；PTO先落INT32至GM再Vector处理，边界不同 |

源码差异仍是：Sparse为128一块、五段softmax/PV及重标定，Native满窗口/候选为128＋512两段；
Native PV按N128并双L0C槽，PTO累加N512；Native条件成对DMA，PTO逐行地址/页表gather。
跨query排空和大工作量KV发布过晚两项已修，不再算尚未处理的差距。
Indexer仍为2＋2＋2 Key复用、独立系数准备、半leaf排序后GM结果合并；Native为4＋2及流式Top-512。
有源码差异不代表照搬就快：4＋2复用和Compressor合并投影实测核内退化，仍不保留。
O_A、HC_post当前核内分别83.37–84.41 / 23.29–27.06 μs，Native为138.46 / 34.26 μs；
没有证据把它们列为当前最突出的核内劣势，额外派发/融合边界留到调度阶段分析。

## 9. 最后三项核内候选后转调度

用户限定本轮再做三项，完成必要代表验证后即转调度，不追加核内候选：

1. **已保留：Indexer 合并 WS 与 Q/系数 L0A 驻留。** 当前128K/B16 Score AIC由354.44–363.56降到335.59–346.15 μs，AIV同步降低；系数准备略增。B40范围重叠，不能宣称全档稳定获益。
2. **已保留：Q_A/KV整行清零写入。** B40 Q_A seed24.54–28.04→6.94–8.50 μs、KV seed11.96–13.14→4.96–5.20 μs；B4 padding检查通过。保持原有单任务和atomic归约规则。[证据](results/csa_incore_20260927/projection_seed_wide/README.md)。
3. **已试验并撤回：量化投影FP16紧凑写回。** B40 Q_B matmul69.74–75.02→75.19–77.19 μs，dequant/RMS/RoPE53.38–57.44→57.77–63.18 μs，核内退化。[证据](results/csa_incore_20260927/qproj_compact_writeback/README.md)。

第一项当前组合仅两档DFX和现有单层诊断，非新的七档或本体计时；Native零容差仍FAIL，最终整模型验收未完成。
[第一项完整结果、误差与原始泳道](results/csa_incore_20260927/indexer_fused_ws_restore/README.md)。

三项已结束：保留Indexer合并规约、连续清零；撤回FP16紧凑写回。当前转入调度阶段，停止追加核内候选。
首项调度目标是Q_A上游链派发：B40四窗口每block核内8.41–9.49 μs，整组启动分散47.76–108.88 μs。
首轮用显式依赖让两个Compressor投影等Q_A完成；上述分散包含资源占用，不全部等同调度器软件开销。

## 10. 已转调度：Q_A先行与选择性预派发

B40先导让两个Compressor投影等待Q_A，保持核内计算不变。
Q_A启动分散47.76–108.88→36.72–46.64 μs，本体1359.26→1327.86 μs（−2.31%）；先保留，其他档位待本阶段验证。
Top-K链尾部仍有波动，不能仅凭Q_A提前宣称调度已完成。
[先导结果、数值与原始泳道](results/csa_scheduling_20260927/qr_before_compressors/README.md)。

用户要求：对预派发抢占靠前槽位却使时序更差的任务，选择性关闭allow_early_resolve。
该标志控制生产者的消费者是否可提前占位，不能当成当前任务自己的priority。
现有B40中O_A后quant平均local_setup80.20–83.71 μs，Top-K merge7.77–52.58 μs；
local_setup含准备和等依赖，大数本身不证明有害。先独立检验关闭Score生产者标志，阻止Top-K merge预派发。

关闭Score生产者标志的B40先导已完成：Top-K merge平均前置等待8.16–53.01→0.68–0.71 μs，
但无profiler本体1327.86→1330.48 μs，无明确收益，已撤回；[独立证据](results/csa_scheduling_20260927/score_no_early/README.md)。
下一项针对Query Hadamard：B40部分窗口提前占AIC等待约44 μs，而Q_B仍在计算；
单独关闭其上游idx_qr_dequant_rope的生产者标志，不叠加Score试验。
