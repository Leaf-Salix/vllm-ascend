# HCA TND：按 Native 核内实现重排优化优先级

2026-10-22。用户要求抓大放小，并参考
[cann-recipes-infer 的 AscendC 算子](https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc/src)。
本文只分析已有数据，不发起新的 Native 测试，后续候选全部直接调用，不使用 ACLGraph replay。

## 来源和实际调用链

GitCode 网页本次访问失败，git 远端也返回 Empty reply。本地 recipes 只读参考为
`6d060311ae75f6105fe4216e20566875da5a8838`，提交日期2026-10-20；没有声称已更新到远端最新 master。
参考 checkout 采用 sparse-checkout，只展开 DeepSeek V4 文档和 hc_post；本轮未修改它。

- recipes 中的 `hc_pre`、`hc_post`、compressor epilog 是相关融合算子，不是完整 HCA attention。
- [DeepSeek V4 AscendC 说明](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/deepseek_v4/deepseek_v4_ascendc_operator_guide.md)
  描述 C128A 的 shared KV、QK→softmax→PV→输出更新，以及 L1 三缓冲、L0 双缓冲。
- 本仓 `vllm_ascend/attention/dsa_v1.py:2540` 的 C128 decode 调用经
  `vllm_ascend/device/device_op.py:681`，实际走 `_C_ascend.npu_sparse_attn_sharedkv`。
  C128 不传 `cmp_sparse_indices`，选择 CFA 分支；不能误拿 C4 的 SCFA 稀疏选择路径比较。
- 对应核内代码在 `csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/`；
  `.cache/csa/csa-native-ops-install/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_attn_sharedkv/`
  中也有已安装源码，可核对本地源码与实际包的接口/模板。
- O_A/O_B 是 TransposeBatchMatMul/QuantBatchMatmul，不位于 recipes 的 attention 目录；
  继续参考已有 `hca_native_five_20260930/README.md` 中的 ops-nn/CANN MDL 证据。
  HC_post 的 A3 入口是 `hc_post_d_split.h`，不把 arch35 MicroAPI/FMA 当作 A3 能力。

## 文档示意与当前实现不能混为一谈

recipes 的 SAS 设计文档介绍 M256、KV512，并允许小 M。历史2026-09-30日志也记录过
另一个 ops-transformer 版本的多 query 合轴策略。**当前仓库和安装源码的 CFA/SWA
`InitTilingData` 明确设 `mBaseSize=gSize`**；host `SplitBalanced` 同样设为 gSize。
metadata AICPU 在 ASCEND910 分支设 `mBaseSize=groupSize`、`s2BaseSize=512`。
本模型 H64/KVhead1 对应 M64。不能据文档或旧版本直接断言现在 Native 将四条 query 合轴。
以上是当前源码分支证据，没有额外宣称已导出本次二进制全部运行期 tiling 内容。

PTO 已有三槽、两步 QK 提前发起和跨 query 延续；这些不是尚未吸收的新优化。
PTO 的压缩块为128行，而 Native 当前 A3 策略为512行。更大块可以减少 softmax/PV
归约次数，但也增加 UB/L1 活跃区；旧512列候选已因 KV 重载和流水组织出现回退。
后续若重试，必须明确新的供数/驻留结构，不能只把常量128改成512。

## 当前真正的大头

原始统计由 [audit_existing_pipeline.py](audit_existing_pipeline.py) 生成，结果在
[NATIVE_KERNEL_AUDIT.json](evidence/NATIVE_KERNEL_AUDIT.json)。Native 是已有 B16 eager
profiling 的5次采样；PTO 是 Q_B24核实验中的原版 control，各7窗，均为旧公共 SDK。
不同时间窗、不同采集器的分项只用于定位，不能直接相减为同窗优化收益。

| 分项 | B16 PTO task包络中位数 μs | B16 PTO核占用/核数 μs | B28 PTO task包络中位数 μs | B28 PTO核占用/核数 μs |
|---|---:|---:|---:|---:|
| Attention AIC | 162.66 | 140.36 | 288.40 | 231.20 |
| Attention AIV | 166.86 | 144.40 | 290.60 | 235.63 |
| O_A | 84.10 | 55.59 | 129.16 | 94.41 |
| O_B | 58.10 | 29.44 | 95.58 | 53.10 |
| 分组量化 | 73.62 | 3.72 | 109.92 | 7.49 |
| 部分和归约与HC_post | 35.34 | 31.40 | 46.28 | 35.90 |

核占用使用 L1 的 `kernel-duration-us`，AIC除24、AIV除48。task包络还包含设置、
提前派发后的等待和不同核错峰；量化的73.62μs包络并不意味着有73.62μs量化计算可消除。
分项有重叠，不能把上述包络相加为整层耗时。

Native B16已有 kernel Duration 中位数：SAS193.384μs、O_A53.761μs、DynamicQuant13.820μs、
O_B33.421μs、HC_post21.420μs。Native O_B使用整token量化，PTO保留八组量化加浮点归约，
两条数学路径不同；PTO的尾段还多承担八份INT32结果反量化，不能只按HC_post名称对比。

因此本轮优先优化 **O_A→量化→O_B→归约/HC_post 的完整尾链**，而不是只因 attention
单个 task 大就重复重写它。O_A现有生成码已经有L1/L0双缓冲，不能再声称“加stage=2”
就是补齐了缺失流水。Q_B16→24核已经在B16/B28回退，不继续围绕核数扫点。

## 本轮结构性候选：O_B归约及HC_post流水

[候选完整模块](candidates/output_fused.py) 只在冻结实验中替换新 HCA TND 的 output 模块：

1. O_A分组发布和每组INT8量化不变。
2. 一个24组MIX任务按 token行块×输出列块分工。Cube依次计算八组INT32点积，
   两个AIV各负责一半输出列，按g=0→7的原顺序乘scale并累加。
3. 每核两个GM交接槽；FIX发布、MTE2读完确认，复用硬件事件，不新增Scalar软同步。
4. 某列块的g7归约完成后，直接执行原weight scale、BF16舍入和四路HC_post。
   不等待其他列块的O_B完成，不保留全量八组partials。

原partials固定容量为384×8×4096×4=48MiB；候选交接区在M96/M128分支分别为
2.25/3MiB。**这只是存储容量减少，不是声称GM流量同比减少**：A3交接仍经过GM。
按有效行计，八组INT32的逻辑写/读字节仍存在，目标是较小工作集、任务交接和收尾重叠。

代价必须同时验证：新任务要等八组量化全部就绪，会失去一部分原分组O_B提前执行；
大T按行块重新读取权重，也可能增加加载次数。只有完整层真机配对才能判定净收益。
先做B16/B28，要求输出、全部cache/state、写保护和重复直接调用均严格保持控制组结果。
保留既有Native storage严格差异，不能把候选逐bit等价写成Native全面精度通过。

共享SDK现已更新为 `csa-shared-20261022-logical-codegen`。候选和控制使用同一新版本；
旧515/771μs等数据只作历史，不与新任务跨版本计算收益。候选CPU编译和设备结果另记优化日志。

实测收口：`task_20261022_183801_30125409797`，B16各7窗，候选全部结果严格保持控制，
但完整L1中位数511.60→623.54μs，增加21.88%，不采用，不扩测B28。
O_A基本未变；新MIX AIC核占用/24为63.27μs、AIV核占用/48为120.66μs，
原O_B和归约/post对应为30.39μs、29.05μs，新任务还失去约40μs的提前开工时间。
本轮否定的是这套具体分工和流水，不把它推广为所有融合都无效。
编译中发现的正步长信息丢失已独立复现并提交 [PyPTO #3105](https://github.com/hw-native-sys/pypto/issues/3105)。

## 已有负结果和待办边界

- 旧双query、512列归并、直接分页读的源码、失败与真机回退均在旧HCA优化日志中。
  它们不证明此方向永远无效，但重试须有实质不同的内存/流水方案。
- 新TND的同址metadata检查前四场景通过；最后A回到Native调用时发生MTE异常，
  尚未归因，不能写成完整A→B→A通过。暂不围绕该测试反复占卡，保留失败证据。
- 完整八档、服务生命周期及整模型token验收仍未完成；本文不宣称30%目标已达成。
