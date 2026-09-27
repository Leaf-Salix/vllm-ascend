# CSA 七档 Native / PTO 核内差异与优化顺序

更新：2026-09-27。分析基线为统一 V10 性能版，算子源码 `0ed4f926`，七档汇总提交 `60c0ee63`。
本文件保存核内阶段的差异分析；新增候选必须另列实测结果，不能回填为 V10 基线。

## 1. 当前结论与边界

1. 128K Indexer 仍有多 query 复用 Key、流式 Top-K、减少重复准备的优化空间。
2. 8K 四档更直接的核内问题是稀疏注意力：PTO `qk_pv` 的 block 平均核内耗时，
   已超过对应 Native 整个 `SparseAttnSharedkv` kernel，PTO 后续另有 `merge_norm`。
3. V10 已采用 QK→FP16→第二次 Cube 做 head 加权规约，不能继续把旧 Vector 规约当作当前差异。
4. 本阶段先吸收 Native 的核内策略；任务融合、派发间隙和任务调度留到核内阶段之后。
   按 batch/长度选择策略时，选择逻辑放在同一套当前 PTO 算子里，不按档位切换历史版本。
5. 以下代码差异是真实存在的，但其耗时贡献尚未逐项实测，不能据此承诺优化百分比。
   当前数据也不足以把核内时间拆成纯 Cube 算术、DMA、同步等待各占多少。

初始七档基线提取只读取已有源码和 trace；后续先导和 Native 缺口补采见第6节，没有做 hash 校验。
QKV、两个 Compressor、O projection、mHC 的完整等范围归因尚未完成；
本文件先展开已有证据指向的 Indexer 和 Sparse Attention，不宣称已穷尽全部 CSA 差距。

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
8K/B40 的 V10 四窗口该任务 block 均值约90 μs，明显值得独立处理。

Native 双流调用顺序对应的 KV `MatMulWeightNz` 设备事件为26.26 μs；
这一映射依据 `_mla_prolog_multistream` 的 Q_A→KV 分工及 trace 的 stream/顺序，
不是逐block的等范围比较，不能由两数直接算加速比。
此外 Native 该权重使用 NZ，而 PTO `wkv` 根入参仍是 ND `[D,512]`，
在初始化时由共享 adapter 解包/转置一次；这不是每次 CSA 的权重重排。
先独立去除性能版的240行精度特例，保留精度版原路径；WKV 的 NZ 复用另列后续项。

源码：[性能版 QKV](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/qkv_proj_rope.py)、
[Native 双流实现](../../vllm_ascend/attention/dsa_v1.py)、
[权重适配](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/native_adapter.py)。

## 5. 优化顺序与必要验证

| 档位 / 阶段 | 核内工作 | 验证与保留条件 |
| --- | --- | --- |
| 8K B16/24/32/40，先用 B40 代表 | PV N128 分块；随后独立尝试 KV 成组搬运 | 先 CPU 编译检查，再单卡原有 case 检查输出、保护区、无 profiler 本体与 DFX qk_pv；有收益再覆盖受影响档位 |
| 128K B8/B16 | Indexer 多 query 共享 Key、Query/系数驻留、流式 Top-K | 单项实现独立测量，核内下降须有证据；B16 Native分项已补齐，不重跑无关历史矩阵 |
| 8K B40 | 移除性能版KV投影240行精度特例，评估复用Native NZ权重 | 先隔离改动测量，按整套当前实现记录结果；精度版保持原算术 |
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
在128K两档及8K/B40下降了局部Score核内时间，但四档CSA本体均未改善，已撤回。
[全有效KV省去UB清零](results/csa_incore_20260927/kv_valid_nozero/README.md)亦没有足够稳定收益，不进入正式源码。

128K/B16 Native分项已补齐，QLI=360.28、Sparse Attention=181.08 μs。
可见V10的Score/qk_pv核内均值已经接近这两个Native完整kernel，但PTO仍有额外准备/合并，
这些数据不能支持“128K/B16所有差距都来自纯矩阵计算”的判断。

仍缺：各项策略独立收益、完整其他 CSA 任务的等范围映射，
以及新策略的整模型验收。阶段目标保持有效，尚未完成。
