# TND-SPMD CSA 与实际 Native 的数值策略差异清单

审计日期：2026-10-23～24（北京时间）；源码快照固定于下述合并提交。

除已授权合并的上游提交外，本轮只审计、记录和做 CPU 算术示例；
审计没有新增算子、精度阈值、SDK 或 Native 修改，
没有提交 NPU 任务，也没有恢复性能优化 Goal。本文不能作为新增精度验收通过的证明。

审计对象是 deepseek_v4_flash_csa_spmd_tnd 的完整 attention 半层：
**内部 HC_pre → input RMSNorm → Q/KV、两路 Compressor、Indexer、Attention、输出投影 → HC_post**。
不审计独立 hc_pre.py、BSH-SPMD、TND-MPMD 或 HCA 的实现；它们的已有记录仅用于交叉说明。

目前最明确的差异不限于浮点加法顺序：**中间 BF16 舍入边界、量化尺度的定义、
softmax 分块与概率舍入也不同**。有些地方保留 FP32，有些地方重新组织归约；
二者都可能改变结果，不能仅凭“位宽更高”或“归约树更平衡”判断最终谁更准确。

## 1. 固定版本、来源和证据口径

### 1.1 本轮已合并的代码

| 项目 | 固定来源 |
| --- | --- |
| 远端 | nalinaly/vllm-ascend，dsv4-flash-pto-v0.25.1rc1 |
| 拉取到的远端提交 | cafcadec96d48790608e5b55d456150fcf00e794，跨 K 部分和二分归约 |
| 本会话合并前 | ede64202f，本轮 Attention PV 预取实验已结束、优化已暂停 |
| 本会话合并提交 / 本文源码行号基准 | df57259b82722c7d335442cc0d260a83f34f66da；无冲突合并 |
| 审计的 TND-SPMD | 当前生产包；采用记录为 [retained_output_rectangle.json][retained]，output_rectangle_production_v1 |
| 合并的影响 | 远端此次没有改动 TND-SPMD 包；独立 hc_pre.py 的二分归约没有自动进入 CSA |
| 公共 SDK | csa-shared-20261022-logical-codegen，status=validated；PyPTO 26e94e05、PTOAS 914863ed、Simpler 57e49923 |

SDK 版本选择仍以主检出 [manifest.json][sdk] 为唯一来源。本文读取了现有生成代码，
没有重新编译或重建 SDK；已有八档代码等价链见 [production codegen][codegen]。
本轮读取的实际 PTO-ISA 头文件根为工作区 pto-isa-src，提交
327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d，读取时工作区干净；
它是源码审计版本标识，不把旧设备实验重新标记成此版本复验。

### 1.2 Native 是性能测试实际使用的版本

基线来自 [CSA_PERFORMANCE_TEST_SPEC.md](CSA_PERFORMANCE_TEST_SPEC.md) 和
[八档性能来源清单][performance]，不是最新 ops-transformer 的另一套算子。

- 单卡 A3、TP1、正式第 4 层 W8A8/NZ2、BF16 激活、seed=1024。
- CANN 9.2.0-beta.2：/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2。
- 自定义算子实际来自主检出 .cache/csa/csa-native-ops-install/vendors/custom_transformer；
  静态编译另用 native-template-cann92-opp 私有 OPP。不是 A5 arch35 路径。
- C04 等原性能记录的框架提交为 beaefa6f6788a9448d49293238f6818c246f24f5；
  C01 补测为 a658188cde9c4d58081feb3a3fbe96307b21ff23；
  C02/C08 补测入口归档在 csa_performance_20261022，统一清单提交为 61af469e。
- 对照上述旧提交，本文引用的 Native 算术文件未改；dsa_v1.py 仅增加
  max_seqlen_q_actual metadata 字段和赋值，未改变 Native 数学或 tiling 参数。
- 本轮一次性比对实际安装包与本 worktree 中对应源码：HC_pre 8、HC_post 4、
  Compressor 13、QLI 8、Sparse 10、RMSNormDynamicQuant 7、RoPE 22，
  共 72 份 .h/.cpp 内容一致。没有反复做逐文件 hash 校验。

Native 并非所有计算都是手写 AscendC：Q 的无权重 RMS 在这套环境实际调用
triton_rms_kernel；F.linear、QuantBatchMatmul 等由 CANN 选择实现。
下文保留这些真实边界，不为了“与 AscendC 比较”而替换参考路径。

本轮只读复核了 C04 原始 kernel_details.csv。它在实际 SuperKernel/多 stream
执行中记录了下列接口 dtype，因而不能假设融合已经消除了这些舍入边界：

| Native 边界 | 原始记录中的 shape / dtype |
| --- | --- |
| HC_pre mixed → input RMS | 96×4096，BF16；post/comb 为 FP32 |
| Q_A → QR norm/quant | 96×1024，BF16 |
| 原始 KV 投影 → RMS | 96×512，BF16 |
| 主 Q_B | 96×32768，BF16 |
| Indexer Q_B | 96×8192，BF16 |
| Indexer key / query Hadamard 矩阵乘 | 40×128 / 6144×128，BF16 输出 |
| head weights 投影 | 96×64，BF16 输出 |
| O_A → O_B 量化 | 96×8×1024，BF16；展平量化为 96×8192，scale 为 96 |

对应原始报告：
[C04 Native report][native-report]；CSV 路径位于该报告的 device_profile 和
[性能清单][performance] 的 cells.C04.Native.raw_csv。它证明接口和实际执行路径，
不等于已经获得每个内核的全部内部舍入次序。

### 1.3 如何读这份清单

本文分为三类，编号是后续逐项定位的入口，不是风险评分：

- **N：数值策略**，包括 dtype、归约、舍入、量化、近似函数和融合边界。
- **S：语义策略**，包括候选选择、同分次序、mask、TND/状态与布局合同。
- **R：潜在正确性风险**，包括同步、复用、别名、整数边界和生成代码。

每项说明双方策略、代码位置、影响、已有证据和待验证事项。
“已确认”指源码或已保存产物直接支持该差别；“待核实”不当作已发生的错误。
“已对齐”单列，不混进已确认差异。没有给各项排序，也没有判定哪方更准确。

必须区分三种目标：

1. 对同一精确输入，更接近实数/高精度参考。
2. 更接近实际 Native 的有限精度计算路径。
3. 与本会话原 TND-SPMD 逐 bit 一致，没有新增回归。

目前性能优化大量使用第 3 种验收，它不能替代前两种。

## 2. 首个完整案例：CSA 内部 HC_pre 的切 K 累加

### N01：四份部分和的合并，以及 Native 的分片合同

**当前 CSA**：[hc_stages.py][p-hc]，hc_project_range（171–206）和
hc_reduce_range（210–233）：

~~~text
BF16 hidden → 无损扩展 FP32
K16384 → 4 份 K4096
每份按 16 个 K256 的 DSL 块顺序累加 → 4 份 FP32 partial
raw = ((p0 + p1) + p2) + p3
~~~

hc_projection_piece（237–263）的消费者就地归约候选也保持这个左折叠顺序，
且当前 policy.hc_consumer_reduce=False。不能把文件里存在候选 helper
当作当前生产路径已开启，更不能把独立 hc_pre.py 的更新当作 CSA 已更新。

**实际 Native**：[HcPre tiling][n-hc-tiling] 的 CalcOpTiling（276–295）按 M 和核数选
cubeBlockDimK；[HcPre stage2][n-hc-stage] 在 357、391、449 行调用
[ReduceSumARAPerf][n-hc-base]（443–482），从第一份开始按 j=1..n−1 顺序 Add。
**Native 跨 K 部分和也是顺序合并**，但分片数量和每片内部矩阵乘次序不同。

由这份 tiling 源码、24 个 AIC、K16384 且 T≤256 推导：
mDimNum=1，splitKSize=ceil_align(ceil(16384/24),256)=768，
cubeBlockDimK=22，末份 K256。八档的 T 均在此范围。
这是源码公式推导；没有重新采八档的设备 tiling，不能将“4 份左折叠”
与“22 份左折叠”视为相同浮点表达式。

**独立 hc_pre.py 的更新仅作提示**：cafcadec9 给四个 col_sum 调用补临时区，
分别涉及两个消费者中的投影部分和与 RMS 部分和；不是只改一处投影加法。
当前 ISA 的 TColSum.hpp::BinarySum 对相邻两项配对；四项时是
(p0+p1)+(p2+p3)。CSA 的显式 Add 链并不调用这条新路径。
提交正文还记录：普通/随机 134 例通过；24 个构造相消样例对 FP64 参考，
原顺序与二分各有 12 例更接近。本文只引用该提交记录，未重验独立 HC_pre。

**可能影响**：部分和相消、量级悬殊、BF16 中点附近的末位差，可经 inv_rms、
sigmoid/Sinkhorn、mixed、QR/INT8、TopK 逐级传播。树形改变误差传播路径，
不保证每个样本更准，也不保证更接近顺序累加的 Native。

**已有证据**：10/21 TND 的阶段采集已经观察到 mixed/post/comb 差异，
但没有隔离四份合并这一变量，不能把差异全部归因于 N01。详见第 7 节。

**待验证**：固定同一批已计算 partial，在同一输入下分别比较左折叠/配对树；
另单独对齐 Native 的分片和片内顺序。对 partial 的 FP64 求和只评价
“合并误差”，不能充当整个投影的 FP64 点积参考。本轮未做设备实验。

### N02–N07：同一个 HC_pre 案例还必须区分这些环节

| ID / 状态 | 当前 TND-SPMD | 实际 Native | 影响、已有证据、待验证 |
| --- | --- | --- | --- |
| N02 数值，部分待核实 | [HC project][p-hc] 171–206：FP32 输入/权重、FP32 Acc，DSL K256，四段；没有显式 HF32 模式设置。 | [HC Cube][n-hc-cube] 418–476：投影 MmadAB 显式 SetHF32Mode(1)、SetHF32TransMode(1)，L0 K32，结束关闭 HF32；平方和走另一个 MmadA2。 | dtype=FP32 不足以说明乘法有效精度一致。模式设置的源码差异已确认；当前 SPMD 生成 C++ / ISA TMATMUL 没有同样的显式设置，但实际寄存器初态、编译展开及片内乘法误差未隔离，不能直接断言 PTO 是完整 FP32 或 HF32。待核对既有二进制/实际模式，后续再固定输入诊断。 |
| N03 数值，已确认归约不同 | [HC widen][p-hc] 113–167：Vector 逐元素平方，每 K512 做 row_sum，32 次顺序累加。 | [HC Cube][n-hc-cube] MmadA2 做 X×Xᵀ，取对角平方和；[stage2][n-hc-stage] 合并跨核部分，再除 K、加 eps。 | 平方积/累加树与 N01 独立。注意当前带临时区的 rsqrt 实际为 VSQRT+VDIV(1,root)，与 Native 这里 Sqrt+Div 的指令策略同类；不能仅凭 API 名称列成另一项算法差别。待固定同一 hidden 比较平方和，再比较同一平方和的 inv。 |
| N04 数值，源码写法不同，当前指令策略已对齐 | [HC gates][p-hc] 299–308：exp(−z)+1 后 recip。当前 TRECIP 经 common/pto_instr.hpp:776–783 映射到 TDIVS(dst,1,src)，A3 TDivS.hpp::SDivOp 填1后调用 VDIV。 | [SigmoidPerf][n-hc-base] 390–424：Exp、Adds 后 Div(1,denom)。 | 不能仅凭 recip 名称声称有另一种近似倒数误差。z 的乘 inv→乘 scale→加 base、pre 加 eps、post 乘2也同类。历史 post 差异尚未控制 logits；若后续定位这一段，应比较实际操作数/生成码，当前不列为已证实算法差异。 |
| N05 数值，已确认 | [HC comb][p-hc] 400–457：4 个有效列置于 8 列物理 tile；行 row_sum；列和为 (r0+r1)+(r2+r3)。 | [HC base][n-hc-base] LastDimReduceSumPerf 用 WholeReduceSum；[stage2][n-hc-stage] 461–483 的列和经 ReduceSumARAPerf 左折叠。 | 列加法树明确不同，行归约在 4 有效列/8 物理列下的生成次序仍需对应。双方是首轮 softmax 后列归一化、再 19 轮，eps 位置未发现不一致。待固定相同 raw/inv/scale/base，逐轮比较，不能用最终 comb 误差反推首个差异。 |
| N06 数值，已确认 | [HC mix][p-hc] 539–557：四路乘 pre，(y0+y1)+(y2+y3)，然后 BF16 rint。 | [ProcessY][n-hc-base] 504–512：四路乘法后 ReduceSumARAPerf 顺序相加，再 BF16 CAST_RINT。 | 与投影切 K 是另一棵四项树；即使 raw 完全一致，mixed 仍可能不同。10/21 mixed 有 90 个零容差差异，尚未控制 pre。待同 hidden/pre 独立对照；最终 BF16 模式本身已对齐。 |
| N07 数值，归约表达不同，精确路径待核实 | [HC mix/norm][p-hc] 557–577 + [rmsnorm.py][p-rms]：先对 BF16 mixed 再扩 FP32，16 个 K256 平方和左折叠，sqrt/div 后乘 BF16 gamma。 | [layernorm.py][n-layernorm] 调 CANN9.2 RmsNorm；安装包有 MultiN、HalfInterval 等分档，merge_n 路径先按 64 列 lane 累加再 WholeReduceSum。 | mixed 的 BF16 边界与 gamma BF16 已对齐；归约分组不能因最终都是 FP32 就认作相同。已确认 C04 trace 是 RmsNorm；八档的具体 tiling/归约路径未全部恢复。待相同 mixed、相同 gamma 比较中间平方和/rstd。 |

## 3. 其余数值策略逐项对照

以下位置均指第 1 节固定快照。相同文件的链接集中指向实际源码，
括号中保留函数名/行号，便于查找。表中“历史”见第 7 节，不能外推成当前实测。

### 3.1 主 Query、QR 和原始 KV

| ID / 状态 | 当前 TND-SPMD 策略及位置 | 实际 Native 策略及位置 | 可能影响 / 已有证据 / 待验证 |
| --- | --- | --- | --- |
| N08 数值，分块表达不同 | [qkv_stages.py][p-qkv] _q_a_worker 102–144：K4096 单 Acc，DSL K256，M32/M64 主块与 M16 尾块。 | [CVLinearWrapper][n-cv] → [F.linear][n-linear]；本轮实际 Q_A 是 MatmulWeightNz/MatMulV2，CANN 按 M/N/NZ 选择 tiling。 | 旧 §45/46 证明改变 Native M/权重 stride 可改变结果；不代表当前八档已找到同样差异。待从这批实际 NZ 二进制恢复 K 顺序，不套用旧 ND/B40 的公式。 |
| N09 数值，已确认 | Q_A FP32 Acc 存 qa_fp32，直接进入 q_r_stage；中间不落 BF16。 | C04 trace 明确 Q_A 输出 BF16，再传 RMSNormDynamicQuant。 | 实际参与 RMS 和量化的输入不同，是比 K 次序更早可见的边界。保留更宽输入不自动等于更接近 Native；待相同 Acc 上加/不加 BF16 往返，仅作归因。 |
| N10 数值，已确认策略不同 | [q_r_stage][p-qkv] 235–259：四段 K256 各 row_sum，再顺序合并；带 tmp 的 tile.rsqrt。 | [RMSNormDynamicQuant normal][n-qr-normal] ComputeRmsNorm 和 [single_row][n-qr-single] 98–130：ReduceSumHalfInterval；设备 Scalar 计算 1/sqrt(sum/1024+eps)。 | 同为 RMS，归约树与 Scalar/Vector sqrt/div 均不同。旧 §43–45 隔离过相同输入下的末位差；当前未采用旧标量 sqrt 精确修正。待先对齐输入，再分离归约、sqrt、倒数。 |
| N11 数值，已确认 | q_r_stage 235–277：amax=max(1e−4, inv_rms × max(abs(x×gamma)))；量化值却按 (x×inv_rms)×gamma 计算。 | Native QR 对实际算出的归一化结果取 Abs/ReduceMax，见 [ScaleTensor][n-qr-single] 182–199，normal 分支同类。 | 将 RMS 正比例因子提出 max 在实数下等价，FP32 乘法结合顺序不同；amax 与实际量化值的极值可能差末位。不能因此直接断言会越界。待构造临界极值和半整数边界。 |
| N12 数值，已确认 | q_r_stage 260–291：Vector 127/amax，再 recip 求反量化 scale；有 1e−4 floor；INT32 rint→FP16 round→INT8 trunc。 | Native QR 每行以设备 Scalar 求 127/amax 和 1/scale；整数转换链见 [helper][n-qr-helper] 180–188。 | 整数转换在正常有限范围内基本同类；主要差别是除法、floor、amax 输入。旧 TDIV 4096 点证据支持存在末位敏感性，不是当前新测。待同 amax 比较 scale，再比较 INT8，零/极小行另列。 |
| N13 数值，SPMD顺序已确认，Native具体分支待核实 | [query_projection_stages.py][p-q] 276–287：INT32→FP32，先乘token scale，再乘weight scale；权重scale由Native BF16无损加宽。 | [DSA prolog][n-dsa] 1913–1920调CANN QuantBatchMatmul。安装CANN9.2的pertoken.h 492–536先AscendDequant(channel scale)，再乘token scale；pertoken_opt.h 494–580则先合并两种scale，再乘转FP32的整数Acc。 | (acc×a)×w、(acc×w)×a、acc×(a×w)可不同。已找到实际安装版本的AscendC策略分歧，但尚未把八档实际tiling逐一映射到模板；不能只按API名选一个作为事实。旧§43/74支持该问题值得隔离，适用于主/Indexer Q_B及O_B反量化。 |
| N14 数值，已确认 | 主 Q_B 的 FP32 反量化结果直接进入每头 512 维 RMS。 | [DSA prolog][n-dsa] output_dtype=hidden_states.dtype；C04 trace 为 BF16 Q_B 输出，再调用 Q RMS。 | 缺少 Q_B→RMS 的 BF16 舍入，会改变平方和和后续 Q。待固定同一 INT32 Acc/scale 比较这一边界，不能将其混进 N15 的归约误差。 |
| N15 数值，Native 归约/低层细节待核实；SPMD两条写法部分已对齐 | [Q dequant/rope][p-q] 282–288、333–343：完整8行用带tmp rsqrt，尾行用 recip(sqrt)，全头 FP32 平方归约。当前ISA两者都展开为 VSQRT+VDIV(1,root)。 | [实际 triton_q_rms][n-q-rms] 27–33：输入来自 BF16 Q_B，tl.sum(x*x)/DIM 与 tl.rsqrt，再写输入 dtype；C04 trace 确认 triton_rms_kernel。 | 不把完整/尾块的API写法差别直接当成数值差异。Triton源码未显式把x扩为FP32，乘法/归约的准确编译dtype不能只凭源码猜测。待恢复实际Triton IR，比较同输入的归约及rstd；完整/尾块仍核对valid_shape和生成后的归约，但尚未证明二者输出不同。 |
| N16 数值，已确认 | 主 Q 的 RoPE 使用未落 BF16 的 FP32 normed；两乘一加后才 BF16。见 [Q rope][p-q] 289–301。 | Q RMS 先写 BF16，随后 [InplacePartialRotaryMul][n-rope] 285–315 加宽、旋转、再 CAST_RINT。 | 少了一次 norm→RoPE BF16 舍入；不是 cos/sin 生成公式不同。待同 normed 输入分离该舍入与乘加顺序。 |
| N17 数值，分档表达不同 | [kv_stages.py][p-kv] 86–160：单 K4096、K512；C05/C06 的 [joint_prefix][p-joint] 46–96 改为 K256、M192。 | [DSA prolog][n-dsa] 1874–1907 的 cv_wkv，实际 MatmulWeightNz/MatMulV2。 | DSL 切块与所有者变化不能直接当作舍入变化，仍须看实际 Mmad K 粒度。旧 B40 K 遍历证据只覆盖当时 ND；当前 C05/C06 采用有旧基线逐 bit 证明，不是 Native 对齐证明。待固定同一输入对照分档和 NZ kernel。 |
| N18 数值，已确认 | [KV norm/RoPE][p-kv] 164–308：FP32 投影直接 RMS；RoPE 也消费 FP32 normed；NoPE/旋转结果才 BF16。 | Native KV 投影先 BF16，RMS 输出再 BF16，之后旋转；C04 trace 与 [DSA prolog][n-dsa] 一致。 | 两个额外 Native 舍入点都被推迟；差异可写入 SWA 并影响后续 decode。待分别固定投影、RMS 输入，不把状态累积误差全部归到当前 Attention。 |
| N19 数值，部分已对齐 / 分档待核实 | [KV norm][p-kv] 188–204、240–256：8 个 64 列平方先按 lane 顺序相加，再 row_sum；sqrt 后 Div(1,root)。 | CANN RMS 的 ReduceSumMultiN + Sqrt/Div 与这条思路一致；它另有 HalfInterval 等分档。 | 旧 §79 的修正保留在当前源码，不应再写成“仍按每块 row_sum 后相加”。剩余风险是 N18 输入边界及当前实际 tiling；待逐档确认，未证明当前归约本身有新的不一致。 |

### 3.2 主 Compressor 与 Indexer Compressor

| ID / 状态 | 当前 TND-SPMD 策略及位置 | 实际 Native 策略及位置 | 可能影响 / 已有证据 / 待验证 |
| --- | --- | --- | --- |
| N20 数值，分块表达不同 | [compressor_projection][p-cproj] 67–138：主 M64/N64、Indexer M16/N32，K512 顺序单 Acc；C05/C06 主分支改用 [joint_prefix][p-joint] M192/N128/K256。value/score 以 FP32 发布。 | [Compressor Cube][n-cmp-cube] 由 tiling 划 M/N/K，value/gate FP32 中间量，接同一 FP32 state。 | value/gate 的状态误差会跨调用保留；不只是当轮输出。旧 §49/74 曾定位片内 K 次序；当前没有逐档对齐 Native 内部 tiling。待给定相同 x/weights 比较两投影与 state 写入值。 |
| N21 数值，已确认 | [compressor_pool][p-pool] 79–213：以最新 token 的后半 value/score 初始化 m/l/o，再按旧到新处理其余 7 行；最后 oi/li。主一次 512 列，Indexer 两片 64 列。 | [SoftmaxDN / KvMulReduceScore][n-cmp-vec] 953–979 调 [ColumnSoftMax][n-cmp-soft] 和 [ColumnSum][n-cmp-reduce]：8 行统一 max、Exp、对半归约总和，先归一化概率，再乘 value，再树形求和。 | 实数目标相同，指数平移、归约树、除法与乘法先后均不同。旧 §75 已隔离过该因素，精度路径曾改为 Native；当前性能 SPMD 仍是在线算法。待固定同一 8 行窗口，比较 pooled，并保留当前/历史 state 两种输入归因。 |
| N22 数值，已确认除法差别，归约部分已对齐 | [main output][p-cout] 94–116 与 [Indexer output][p-icout] 137–152：512 维按 256→128→64 折叠；128 维按两半折叠；recip(sqrt) 后乘 x，再乘 gamma。 | [Compressor RmsNorm][n-cmp-rms] 46–84：对应折半 RowSum，Sqrt 后直接 x/root，再乘 gamma。 | 当前已经保留折叠树，不能重复列为“未修复归约”；乘倒数与直接除仍不同。Native gamma 当前也加载 BF16，不沿用旧“compressor gamma FP32”的历史描述。待同 pooled/gamma 比较 root、normalized 与 compact cache。 |

主/Indexer Compressor 均在 FP32 norm/RoPE 后输出 BF16，这个输出边界与 Native
融合 Compressor 同类；它不同于原始 KV 那条“独立 BF16 RMS → 独立 RoPE”路径。
两条 KV 路径不能合并成一个统一舍入结论。

### 3.3 Indexer Query、量化、head weights 与评分

| ID / 状态 | 当前 TND-SPMD 策略及位置 | 实际 Native 策略及位置 | 可能影响 / 已有证据 / 待验证 |
| --- | --- | --- | --- |
| N23 数值，已确认边界不同 | [indexer_query][p-iq] 107–237：INT32 先乘 QR scale 后乘权重 scale，FP32 直接旋转，最后 BF16。 | [cv_indexer_select_qli][n-dsa] 2897–2923：QuantBatchMatmul 输出 BF16，再调用 InplacePartialRotaryMul。 | 少了投影→RoPE BF16 边界；尺度结合的精确 Native 路径另见 N13。旧 §43 精度修正不能当作当前性能路径已采用。待固定 QR INT8/scale，分别隔离反量化与旋转。 |
| N24 数值，已确认，query/key边界不完全相同 | [query Hadamard/quant][p-iq] 241–295：BF16矩阵乘输出FP32，FP32乘1/√128后量化。Key的 [Hadamard][p-icout] 177–201也发布FP32，但 [key quant][p-icache] 98行先做BF16往返，再FP32缩放。 | [hadamard_linear / hadamard_scale][n-dsa] 72–119：F.linear输出BF16，缩放结果也是BF16，再DynamicQuant。C04 trace有两路BF16 matmul输出。 | Query缺少矩阵乘后和缩放后两处BF16；key已恢复前一处，但缩放后仍保留FP32。不能把key写成“两处都缺失”。可能改变INT8边界及scale；待同RoPE输出分别比较两路Acc、缩放和量化。 |
| N25 数值，尺度公式依Native分档待核实 | [query quant][p-iq] 261–295、[key quant/cache][p-icache] 90–126：amax至少1e−4；Vector 127/amax后recip作dequant scale，key最后写FP16 scale。 | [DeviceOperator][n-device] 703–752调CANN DynamicQuant。安装CANN9.2 single_row模板以amax×(1/127)写scale；另一dynamic_quant.h模板先Vector 127/amax，再Scalar 1/scale。 | 不能把某一模板当作所有Native档位的实际公式。1/(127/a)、a×(1/127)、Scalar/Vector除法及floor均需按实际分档区分；整数转换在正常范围同类。待恢复每档tiling、比较同amax下scale和零值规则，最后FP16舍入可能掩盖或保留差异。 |
| N26 数值，已确认 | [weights project/reduce][p-iq] 299–348：四段 K1024，各两次 K512；FP32 partial 左折叠后乘 WEIGHTS_SCALE，直到 QLI 系数才转 FP16。 | [DSA][n-dsa] 2448、2461–2474：weights_proj/F.linear 先 BF16，再乘模型 scale（BF16 输出），最后转 FP16。 | 与 HC 切 K 同类但属于独立投影，且额外缺少 BF16 边界。旧 §52 证实一个 head 系数的 FP16 subnormal ULP 可改变候选排序；当前未做该变量的隔离。待同 x 比较投影、缩放和最终 FP16 coefficient。 |
| N27 数值，已确认计算组织不同 | [indexer_score][p-score] 129–1165：G2/G6 把各 query 的 64 heads 拼接，系数矩阵其余位置补零；长 G6 split 路径分两个 K192 累加、短尾按 live 分支。 | [QLI ComputeWs][n-qli-cube] 494–509：每 query 的 M16/K64 FP16 Cube 归约；QK→FP16 后再做 heads 加权。 | 零块拼接、不同 M/K 形状和累加初始化可能改变舍入/特殊值行为；不能从“加零”推导所有输入 bitwise 一样。当前对旧 SPMD 的有限样本验证通过；待同 FP16 QK/coefficients 对照，尤其正负权重相消、live=1/2/4/5。 |

QLI 的以下数值边界已经对齐，不能误列为当前缺失：
INT8 QK→INT32；FIX 按 1/1024 缩放并 ReLU 后落 FP16；
query scale 和 weights 分别转 FP16 再相乘；WS 输出 FP32 后乘 FP16 key scale
扩展出的 FP32。对应 [QLI Cube][n-qli-cube] 514–550、
[QLI Vector][n-qli-vec] 245–266、345–360、
[head_coefficients][p-group] 293–329 及 [Score][p-score] 的 pre_quant/pre_relu。
这些相同边界之前的输入差异，以及 N27 的归约组织，仍可能使分数不同。

### 3.4 Attention、输出投影及 HC_post

| ID / 状态 | 当前 TND-SPMD 策略及位置 | 实际 Native 策略及位置 | 可能影响 / 已有证据 / 待验证 |
| --- | --- | --- | --- |
| N28 数值，部分已对齐，PV 组织不同 | [attention_stream][p-attn] 131–215：QK 的逻辑 K512，PV 每 128 候选单独形成 FP32 partial。已有 C04 生成码的 QK Left/Right 为 K128，顺序累计四块。 | [Sparse Cube][n-sa-cube] 344、510–527：QK K128 逐段累计；PV 在 Native 512 候选块内按 K128 累计再发布。 | 不能把 DSL 的 K512 误写成一条硬件 K512 指令；QK 的 K128 思路已对齐。PV 的四块先 Cube 累加还是先输出再 Vector 重标定不同。待固定相同 BF16 概率/KV 分离 PV 累加误差；输入概率差异先看 N29/N30。 |
| N29 数值，已确认 | [Attention softmax][p-attn] 403–435：640 个候选分为 5 个 K128；各块使用局部 max，局部 FP32 exp/sum 后落 BF16 概率；PV 后再用 alpha/beta 合并。 | [Sparse tiling][n-sa-tiling] sInnerSize_=512；[SoftmaxFlashV2Compute][n-sa-vec] 422–449 使用前块累计 max/sum，原始窗口后处理 512 压缩候选。 | 即使 QK 完全相同，落 BF16 时的指数参考点也不同；FP32 后合并不能恢复已经丢掉的概率位。旧 §55 的 CPU 和设备边界已有定位；当前性能实现仍采用局部128策略。待同 Q/KV/indices 固定分块与 max 策略作对照。 |
| N30 数值，已确认 | [Attention][p-attn] 424–431：概率转 BF16 为 rint，中点取偶。 | [Sparse Vector][n-sa-vec] 482：概率 CAST_ROUND，中点远离零；最终输出 854 行才是 CAST_RINT。 | 不要把两处 cast 混淆。旧 §71 和本文 CPU 中点例说明能直接产生不同 BF16；当前源码明确保留性能版 rint。待当前真实分数中点命中率及传播影响，暂不改模式。 |
| N31 数值，已确认 | [Attention merge][p-attn] 488–550：running m 从 sink 起步、l=0；各局部块 m/l/PV 重标定；最终分母再加 exp(sink−m)。 | [Sparse Vector][n-sa-vec] CopySinksIn、SliceAndContactSinksValue、SoftmaxFlashV2Compute：sink 作为初始 max，初始 sum=1；每块把先前 sum/O 缩放，末块直接除总 sum。 | sink 的贡献在末尾添加或随累计 sum 多次重标定，舍入位置不同；局部 l 的 row_sum、online O 加法方向也不同。待固定 exp/PV 只比较状态更新，不能把差异都称为最终除法精度。 |
| N32 数值，已确认 | [Attention publish][p-attn] 543–570：NoPE 用 n_full 的 BF16；RoPE 部分直接从 FP32 n_full 逆旋转后才 BF16。 | [DSA forward][n-dsa] 1825–1835：SparseAttnSharedkv 先输出 BF16，再对其做独立 inverse RoPE，最后再次 BF16。 | 少一次 Attention→inverse RoPE 的 BF16 舍入。旧 §79 留有相同 Q/KV 时 Attention/逆 RoPE 仍不同的记录，但并未归因到这一项。待同 normalized O 对照两条舍入链。 |
| N33 数值，已确认边界不同，矩阵乘次序待核实 | [O_A][p-oa] 74–261：BF16 输入/权重，FP32 Acc，K256 循环，直接给量化；M128 或 M160/N192。 | [DSA _forward_o_proj][n-dsa] 1753–1765：TransposeBatchMatmul/WeightNz 输出 BF16；C04 trace 明确 96×8×1024 BF16。 | O_A 舍入被推迟；不同 shape/NZ tiling 的 Acc 次序还可能增加差异。旧 §260 与 HCA 本仓记录都指出这一边界，HCA 的修正不代表 CSA 已修复。待固定 Attention 输出比较 Acc 和 BF16 输出。 |
| N34 数值，已确认量化定义不同 | [O_A quant][p-oa] 265–323：每 token 的八组各对 1024 列取 amax、独立 scale；有 1e−4 floor。 | Native 先把 O_A 展平为每 token 8192 列，再 DynamicQuant，一行共享 scale，见 [DSA][n-dsa] 与实际 C04 trace。 | 这是量化策略差异，不只是 FP32 加法次序。分组可改变小幅值组的量化分辨率，也改变与 Native 的结果关系；不判定谁更准确。待相同 O_A 输入分别量化，比较 INT8/scale/最终 O。 |
| N35 数值，已确认 | [O_B][p-ob] 94–202、266–297：每组 K1024 得 INT32，逐组转 FP32×组 scale，八次顺序相加，再乘 weight scale，落 BF16。 | Native 对统一量化后的 K8192 做整数点积，随后按 token/channel scale 反量化并输出 BF16。 | 量化输入与 scale 已不同；即使人为统一 scale，“先转 FP32 并缩放再相加”和“先 INT32 合并再转 FP32”也不同。K8192 的 INT32 最坏绝对和 134217728 未溢出，但转 FP32 并非全域精确。旧 §76/77/260 与 HCA 记录可参考；待拆开 N33、N34、N35 单变量比较。 |

HC_post 当前没有发现额外数学策略差异：
[hc_post_row_block][p-ob] 227–262 先 post×x，再按残差 0、1、2、3 左折叠，
每项是分开的 Mul/Add，最后 BF16 rint；A3 实际走
[hc_post_d_split.h][n-post] 299–350，同样次序。不要拿 arch35 的
MicroAPI MulAddDst 当作当前 A3 Native 路径。旧 §260 也有同输入 1572864
元素一致的证据；它只说明当时同输入下 HC_post 一致，不能消除上游 post/comb/O 差异。

## 4. 语义差异与边界合同

这组包括“表达方式不同但合法输入下目标语义相同”的项目。它们是逐项比较结果，
不等于已经发现语义 bug；需与上面的数值误差分开。

| ID / 状态 | 当前 TND-SPMD 与 Native 的策略比较及位置 | 可能影响、证据与待验证 |
| --- | --- | --- |
| S01 TopK 同分规则，组织差异已确认 | [topk][p-topk] merge2_top512_pairs 新右根在旧左根之前；多根按 incoming3→2→1→旧 root；Native [QLI Vector][n-qli-vec] 366–397、[排序实现][n-qli-sort] 采用 SortAll/滚动 MergeSort 和另一套块/核分割。 | TopK 不仅有集合，还包含顺序；同分时归并输入先后可能决定留下谁。即使选中集合相同，KV 输入次序也可能影响 Attention。已有自身同分/量化/全相等探针及失败负例，不是与 Native 的全域 tie 证明。待固定 score 比较 index、集合、顺序与第512名并列带。 |
| S02 阈值筛选，新增组织但未证明语义差异 | [threshold_merge][p-threshold] 108–143：以旧根第512项为阈值，使用 ge 保留等于阈值的项；仅满足容量条件才缩小排序。Native 使用完整块 SortAll/MergeSort。 | ge 避免简单丢掉并列候选；仍要保持 gather 后顺序、padding 与 fallback 的同分优先级。C03/C08 开启，旧参考逐 bit 检查通过。待与 Native 同分边界比较；不能称为近似 TopK，也不能把剔除数量等同误差。 |
| S03 可见长度与因果 mask，合法域目标一致 | [request_geometry][p-geometry]、[indexer_query_visibility][p-group]、[attention_plan][p-plan] 从 Q/L/positions 推请求归属和 floor((position+1)/4)；Native [QLI kernel][n-qli-kernel] 与 DSA metadata 使用累计 query 长度、原始 KV 长度和 cmp_ratio=4。 | TND 混长不能使用 token//6。当前会拒绝重叠/非法 Q 区间或失配 position；Native 对畸形 metadata 的行为未承诺与之相同。已有动态 metadata/混长/页表测试；待逐项对照不足4、跨4边界、S1–6及不同 H。 |
| S04 有限 sentinel / 无效页，表达差异已确认 | [config][p-config] FP32_NEG_INF是−FLT_MAX；Score/TopK填无效位置，Attention加相应bias并跳过全无效block。Native [QLI排序实现][n-qli-sort] 24行NEG_INF=0xFF800000，即−∞；Sparse另用actualColumnCount/mask，初始化常量还有−2e38。 | −FLT_MAX不等于−∞，不能仅凭变量同名认为等价。部分无效块在合法有限输入上通常exp下溢为0；不能推到NaN/Inf。PTO历史负页的保护分支可能留零value/score，Native页有效性合同也须单列。待固定极端分数、全无效/部分无效及非法页对照。 |
| S05 空行、dummy 和全 masked，合同差异待核实 | [publish][p-publish] 无可见候选时 index=−1、score=sentinel；Attention 保留 sink、无 KV 时分子0；服务对全空调用另行处理。Native [QLI][n-qli-vec] CleanInvalidOutput，Sparse 的无有效列路径继承 max/sum。 | 部分计算按 token 容量执行，dummy 数值不属于业务输出，但不能污染有效 cache。现有 empty/fallback/图保护验证覆盖部分场景；待明确 Native 与 PTO 对“有 token 但无有效候选”的逐项合同。 |
| S06 Compressor 历史窗口 / 状态顺序，目标一致但机制不同 | [pool][p-pool] 当前投影覆盖窗口内新 token，读取旧 FP32 state，全部 pool 读完才 commit；Native [Compressor vec][n-cmp-vec] OverLap/ReadState/SaveState 与 workspace 管理八行窗口。 | 更换同一前态与从初态连续运行不是同一实验；舍入会进入后继 state。当前已有4步连续状态及共享前缀保护；不代表无限轨迹。待固定相同旧 state，核对窗口顺序、负历史、关闭点及最后写者。 |
| S07 Native Tensor stride / 页面别名，合法域已适配 | [native_adapter][p-adapter] + [native_storage][p-storage]：按真实页 stride/offset 暴露物理连续页；INT8 key 后4096B处是32个 FP16 scale；不是将“未传stride”当作无padding。Native scatter/PA 按其 descriptors 地址访问。 | 布局错会表现为分数/状态错误，不能称作舍入误差。现有保护区和碎片/逆序页覆盖；任意 strided hidden、不同 block size 或独立 key/scale storage不在当前支持域。待扩域前核对每个根 descriptor，而非改变数值容差。 |
| S08 特殊数值与合法域，未有完整等价证据 | 全链有 max/min、ReLU、Abs、Exp、sqrt/除法、signed zero、整数cast；PTO floor=1e−4、finite sentinel、FP16系数，Native模板的零/NaN/Inf/subnormal规则不全相同。位置见 N04/N10/N12/N21/N25/N27/N29–N35。 | 特别是有 NaN/Inf 时，补零 head 的 0×Inf 不能当作无作用。当前性能 fixture 为有限普通值，历史离散探针不覆盖全链极端数值。待先定义特殊值合同，再做针对性验证；不据理论可能性宣称已有非有限故障。 |

## 5. 单 SPMD 特有的潜在正确性风险

这些项目可能造成错读、漏写、运行间不一致，和浮点误差的性质不同。
现有保护/逐 bit 回归降低了已覆盖范围的风险，但没有证明所有形状与调度都正确。

| ID | 双方机制、当前代码位置 | 已有证据与待验证事项 |
| --- | --- | --- |
| R01 阶段发布、读可见性和代际 | [signals][p-signals]、[phase_sync][p-sync] 使用每写者独立512B槽、PIPE_ALL、cacheinvalid/fence、generation；Native 为多个 kernel/stream/event 的依赖和各 kernel 内部事件。 | 64B Scalar cache line 与512B槽间距不是同一个概念；不能因最小线宽就缩短协议隔离。当前没有新发现的丢通知；需持续核对实际生产者集合、空角色发布、GE等待以及 generation 回绕/并发合同。 |
| R02 AIC/AIV 环形缓冲与局部统计复用 | [Score][p-score] 和 [Attention][p-attn] 的事件、三槽、QK领先PV、LOCAL_ATTENTION_STATS；Native 自有 preload/ring/ACK。 | 先发布后写完、消费前覆盖、动态分支少等一次会制造“精度差异”。现有源码/编译产物及完整链回归通过；仍需在更改槽数、prefetch、流水顺序时重证 RAW/WAR/WAW。本轮未删除屏障。 |
| R03 Indexer FP16 scale 的64B读改写 | [scale page plan/commit][p-scale-plan] 102–132 将同物理页交同写者，读64B、改其中一个half、写回；Native scatter按元素/slot写入。 | 同页多请求若拆成多写者可能丢失邻元素。既有碎片/共享页和保护检查通过。Scalar FP32→FP16 cast未显式标 mode，最后舍入仍须对实际生成码/边界值核实，不能只根据 Python cast 认为已等同 Native .to(FP16)。 |
| R04 共享历史 state 的读后写生命周期 | [vector_prefix][p-prefix] 等所有相关 pool 完成才调用两个 state_commit；Native Compressor 在自己的任务与状态生命周期内组织。 | 非本核消费者未读完就写新 state 会成为跨请求污染；合法共享只读前缀与共享可写 slot 不是同一合同。现有回归通过，待更大动态矩阵/重复目标slot的合同核查，不推断 Native 支持冲突写。 |
| R05 padding、valid_shape、Acc 首次初始化 | [HC][p-hc] 尾scratch，[cproj][p-cproj] 动态M首个matmul建立Acc布局，[O quant][p-oa] 唯一尾零owner；Native DataCopyPad/mask及自己的物理行距。 | 未加载pad被cast恢复有效、Acc行距错、init条件错会污染有效列；不是降低容差能解决。现有尾块/guard通过，历史有编译与alias负例。待每个新增形状核对输出有效范围、初始化和编译后的物理跨度。 |
| R06 图重放、容量策略与根地址 | [binding][p-binding]、[graph_contract][p-graph]、[service][p-service] 固定容量/策略、更新同地址Q/L，合同不符回退；Native有自己的metadata生产者与graph节点。 | 不能把捕获时的均长/连续页证明复用于新内容。当前有同址变metadata、服务/图/回退验证；无新的NPU证据。须保留每代证明、根地址与Host上界身份，并禁止同一workspace并发使用。 |
| R07 精度模式、FMA、自动归约与 SDK lowering | Native HC 明确设置 HF32，部分数学用 Scalar，PTO经历 tensor→tile→ISA及运行时包装；编译器可分块、合并或物化tmp。见 N02、N03、N10、N15、N28。 | 已检查代表 C04 生成代码；不是所有内核的反汇编。不能从源码 API 名称直接判断最终精度，也不能从A5实现推断A3。缺少的寄存器初态、Triton lowering、Native各档tiling作为待验证，非已证实SDK缺陷。 |
| R08 整数范围、页地址证明及输入输出别名 | [leaf_scale_layout][p-leaf] 明确要求 physical_pages≤2²⁴才用FP32证明整数页连续；其它几何用INDEX/INT32；[binding][p-binding]约束shape/stride，生产服务分配工作区。 | INT8点积在现有K上不溢出INT32，INT32→FP32是否精确须逐界判断；FP32地址比较超2²⁴可能误判，现有页证明有回退。hc_post先读四路残差仅保障本token当前块，不能证明任意重叠根Tensor安全。未见通用全部区间别名拒绝器，手工binding的重叠输入仍须合同检查；没有本轮复现的别名故障。 |

## 6. 分档、共同点与源码覆盖范围

### 6.1 不能用 B16 结论替代所有分支

本轮用纯 Python 调用了当前 select_policy，未导入算子或运行设备。下表只展示与
数值组织有关的差别；完整配置仍以 policy 和实际 frozen descriptor 为准。
所有档 hc_consumer_reduce=False，HC 四份合并均保持左折叠。

| 档位 | T | Score G / 组织 | 特殊路径 | O_A M/N；O_B M |
| --- | ---: | --- | --- | --- |
| C01 128K/B1/S6 | 6 | G2，balanced半叶 | Q的8行尾块；B1叶分配 | 128/128；32 |
| C02 256K/B16/S6 | 96 | G6，balanced，split Score | 更大 plan 页表容量；具体值依根descriptor | 128/128；96 |
| C03 128K/B8/S6 | 48 | G6，balanced，split Score | threshold前缀、宽Q_B | 128/128；96 |
| C04 128K/B16/S6 | 96 | G6，balanced，split Score | 叶内scale布局证明 | 128/128；96 |
| C05 128K/B24/S6 | 144 | G6，balanced，split Score | joint前段M192/K256 | 160/192，静态N64尾；128 |
| C06 128K/B28/S6 | 168 | G6，balanced，split Score | joint前段M192/K256 | 128/256；128 |
| C07 8K/B32/S6 | 192 | G2，非balanced，Score768 | 原Score及双根组织 | 128/256；128 |
| C08 128K/B16混长 | 56 | G6，balanced，动态live | threshold前缀、宽Q_B、组次序 | 128/128；96 |

C08 是固定 [1,2,3,4,5,6,6,5,4,3,2,1,1,3,4,6]，不是补成96个业务token。
这里主Q的完整/尾块取决于T容量，不按请求S划8行；不同请求混排时更需区分
“逻辑请求边界”和“数值tile尾块”。除八档外，服务允许域内的S1–6、dummy、
非均匀历史和容量边界也应在后续精度工作中保留。

### 6.2 已核实相同或部分相同的保护点

| 点 | 当前结论 |
| --- | --- |
| 权重来源与 dtype | 使用实际 Native 加载后的权重；norm_w/gamma_cq/gamma_ckv/main_gamma/idx_gamma当前都是BF16。FP32权重文件经Native参数加载转BF16是共同来源，不是本轮新增精度损失。 |
| BF16→FP32 | 对有限 BF16 值本身是精确扩展；真正差别是是否先有一次FP32→BF16。 |
| HC 参数与数学常数 | 四路、HC维16384、20轮Sinkhorn、HC/NORM eps=1e−6；本轮未发现参数不一致，未作调整。 |
| HC高精度rsqrt | 现有生成码为三参数TRSQRT；A3 ISA带tmp分支执行VSQRT+VDIV。不能标成直接VRSQRT。 |
| recip与1/x | 当前TRECIP映射到TDIVS(dst,1,src)，A3填1后VDIV；因此HC sigmoid、Q完整/尾块不能仅凭API名称声称两种近似指令。Compressor的x×(1/root)与Native x/root仍是不同舍入位置。 |
| 原始KV RMS | 64列lane先累加再归约、sqrt+Div的历史修正仍保留；当前需查输入边界和Native分档。 |
| QLI关键低精度边界 | QK/1024→FP16+ReLU、FP16系数与FP16 key scale路径已经存在；不存在“当前仍整段FP32评分”这一笼统结论。 |
| HC_post | A3的乘加顺序与最终CAST_RINT同类；不能引用A5 MulAddDst构造差异。 |
| 原始Q_A/KV atomic | 当前包在ATOMIC_ADD=True时拒绝进入；无旧QR/KV split-K atomic运行间归约。HC/weights的显式partial归约不等于atomic。 |
| 业务cos/sin | 复用Native metadata的FP32表；PTO准备符号/相邻交换，不独立重算三角函数。舍入差异主要在进入/离开RoPE的值及乘加，而非自行改了RoPE频率。 |
| INT8点积上界 | QLI K128≤2097152、Q_B/每组O_B K1024≤16777216、Native O_B K8192≤134217728，均低于INT32上限；K8192转FP32则可能舍入。按绝对输入128保守计算，未以此证明更长K安全。 |

### 6.3 本轮覆盖的文件族

| 范围 | 已核对的文件 / 比较入口 |
| --- | --- |
| 入口、策略、ABI、绑定、服务 | entry、policy、runtime_abi、binding、native_adapter、service、service_config、graph_contract、README；共同 native_adapter/native_storage/config/reduction |
| HC及norm | hc_stages、共同rmsnorm；Native HcPre/HcPost/RmsNorm源码与实际dtype |
| Q/KV | qkv_stages、query_projection_stages、kv_stages、cache_stages；Native CVLinear、DSA、RMSNormDynamicQuant、RoPE |
| 两路压缩 | compressor_projection/pool/output_stages、indexer_compressor_output_stages、joint_prefix_stages；Native Compressor及state/metadata |
| Indexer | indexer_query/group/score/topk/publish/cache_stages、topk_threshold_stages、scale_plan/layout、leaf_scale_layout、score_group_order |
| Attention/输出 | attention_plan/dependencies/stream_stages、output_projection_a/b_stages；Native QLI/Sparse/O投影/HC_post |
| 时序/资源/动态语义 | cube_schedule、vector_schedule、vector_prefix、projection_overlap、signals、phase_sync、metadata_stages、request_geometry；既有生成码及状态/图证据 |

覆盖是对当前源码可见阶段与相关代码路径的检查，不宣称穷举了编译器、硬件或
所有浮点输入。特别是未拿到的 Native 精确 tiling、Triton IR、HF32运行状态，
已作为未闭合项保留，而没有用最新版外部仓库的实现补成“事实”。

## 7. 本仓既有精度记录及其当前含义

主历史来源为 [DSV4_FLASH_CSA_VALIDATION_LOG.md][history]；
TND迁移范围调整见 [TND实现记录][tnd-log] 182–245 与
[GOAL_UPDATES.md][goal-updates]。部分早期大产物已清理，历史文档不能替代不存在的原始张量。
v4/v5原始JSON和C04 Native原始report均为本机留存产物，链接在本机有效但未入Git；
仓内日志和性能来源清单保留其路径、任务与摘要，本文未额外上传这些产物。

| 历史记录 | 当时得到的证据 | 当前审计如何使用 |
| --- | --- | --- |
| §43–45，2026-09-22，约1197–1298行 | QR的设备Scalar除法/平方根、两尺度结合顺序、QA的M/stride变化均可改变边界值；CPU FP64并非总是与Native逐bit一致。 | 对应N08–N16/N23；当时精度路径的修正不能自动算入当前性能SPMD。 |
| §49、52，约1419、1489行 | Indexer compressor投影和head weights顺序；一个FP16 subnormal系数差可改变TopK；相邻名次变化并不必然是同分。 | 对应N20/N26/N27/S01；先核实分数，再谈tie。 |
| §50–51；独立 [TDIV问题记录][tdiv-history] | 当时A3/CANN9.0，4096个正normal样本，默认/高精度TDIV相同，其中244个较正确舍入商差1ULP；设备Scalar路径该样本零差。提交9031197a。 | 当前A3 TDiv.hpp仍未根据PrecisionType选补偿路径；本轮只读源码，没有把旧统计标成新SDK重测。 |
| §55，约1590–1620行 | Native512累计softmax与PTO128局部softmax的BF16边界；固定Q/KV的CPU归因显著改变差异数量。 | 当前性能SPMD仍有N29的策略差异；CPU归因不等于Cube逐bit模拟。 |
| §71，约2202行 | Native概率CAST_ROUND与PTO rint；0.595703125的BF16两侧值不同。旧精度版改为round后若干连续轨迹失败消除。 | 当前Attention仍明确使用rint（N30）；旧修复没有迁到本包。 |
| §72、74、75，约2280、2356、2401行 | 旧M240 Native的K访问顺序、反量化结合、Compressor在线池化/折半RMS与直接除法分别定位。 | N17/N20–N22仍要按当前NZ2和实际Native逐项检查；B40/旧权重结论不外推为当前八档误差。 |
| §76–79；提交18ec20a | WO-B量化/反量化、SWA RMS修正；正式权重936项通过，但仍留Attention/逆RoPE的35个BF16差异。 | 与当前源码核对：SWA lane归约保留；groupwise O量化、部分中间FP32等性能策略仍不同。未取得Native内部累加器时不预判QK/softmax/PV哪个负责。 |
| §100–101，约3330–3395行 | 旧atomic=1自身重复输出出现283个差异，关atomic后该例稳定，但仍有Native差异。 | 当前SPMD拒绝ATOMIC_ADD，不把旧运行间波动算作现存已证实问题；跨实现差异仍独立。 |
| §260，约8185行 | 同Native输入/gate的HC_post 1572864元素一致；HC_pre+norm仍有差异；已指出O_A BF16与整token量化合同。 | 对应N01–N07/N33–N35，HC_post本体单列已对齐。 |
| 10/21 TND [阶段采集v5][tnd-stages] | B3、Q=[2,6,1]、H8192；mixed 90、post 27、comb 144、normed 79个零容差差异；Indexer query INT8有20432个不同。 | 这是旧TND阶段证据，不是当前八档的新精度统计；也未单变量隔离，不能给某条N项分配全部误差。 |
| 10/21 [归属对照v4][tnd-attribution] | 新TND与原SPMD同输入的输出/TopK/score/cache/state零差异；两者对Native均17个输出超0.01/0.01、TopK替换26个。 | 用户随后明确“原有差异单列、本轮无新增回归”；这解释了后来性能PASS不等于Native严格精度PASS。 |
| 10/22–23 当前采用 [integration][integration]、[runtime validation][runtime-validation] | 当前输出矩形版本有旧SPMD逐bit、16种metadata、4步连续状态、服务、图、保护区和八档代码等价证据。 | 支持“当前保留优化未新增已测回归”，不是所有N/S/R项已关闭。 |
| [HCA NATIVE_OUTPUT_PRECISION.md][hca-evidence]，10/22 | 另一实现把O_A BF16、8192整行量化、INT32先合并恢复后，原3处输出超差归零。 | 仅作同仓同类问题的旁证，支持审查N33–N35；不是CSA实验，未复制修改、未引用其收益作为CSA结论。 |
| 独立HC cafcadec9 | 二分partial合并的134普通/随机例、24相消例，FP64参照下原版/二分各12例更近。 | 仅作N01动机及“更平衡不保证每例更准”的边界；不是当前CSA的测试。 |

10/21 v5 的 Indexer weights/query_scale 比较还因 FP32 vs FP16 被诊断器直接判
dtype不一致。当前消费边界实际会显式转FP16（head_coefficients），因此
**那个比较失败不能直接等同于“当前 QLI 没有按 FP16 消费”**。
v5任务最后还遇到服务代理接口异常，不能把整个任务改称完整PASS。

原始v4/v5保留的最终输出：max_abs=0.015625、RMSE=0.0028023490799762288；
零容差不同元素数与0.01/0.01超差元素数是两种统计。本文沿用原数值，不放宽门槛。

## 8. 本轮补充的纯 CPU 算术示例

只运行标准库 struct/math，对四个已经是FP32的部分和模拟每次加法舍入。
没有导入torch/PyPTO，没有使用NPU；它不模拟Cube内部归约、HF32或整个CSA。

| partials | 左折叠 FP32 | 相邻配对树 FP32 | 精确和 | 说明 |
| --- | ---: | ---: | ---: | --- |
| [33554432, 1, −33554432, −1] | −1 | 0 | 0 | 本例树更接近精确和 |
| [33554432, 1, −33554432, 1] | 1 | 0 | 2 | 本例左折叠更接近精确和 |

复现代码（本轮实际执行的同一表达式）：

~~~python
import math
import struct

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]

for values in [
    [33554432, 1, -33554432, -1],
    [33554432, 1, -33554432, 1],
]:
    p = list(map(f32, values))
    left = f32(f32(f32(p[0] + p[1]) + p[2]) + p[3])
    tree = f32(f32(p[0] + p[1]) + f32(p[2] + p[3]))
    print(left, tree, math.fsum(p))
~~~

另外按FP32位型做BF16就近舍入，历史中点0.595703125得到：
rint=0.59375，round=0.59765625。它说明N30的舍入模式差异可以独立存在；
不是本轮重新获得的Native/PTO设备输出。

这两个示例的精确参考只针对给定partial或给定概率，不证明哪种完整模型策略更好。

## 9. 本轮交付边界与待闭合事项

本文登记35个数值检查点、8个语义检查点、8个潜在正确性风险检查点。
其中明确写成“已对齐 / 待核实”的项不计为已发现错误，不能汇报成“51个bug”。

尚未闭合的核心证据是：当前八档各Native内核的精确tiling/舍入序、HF32实际模式、
Triton的中间dtype、固定相同中间输入后的各项因果贡献，以及更长连续状态和整模型影响。
没有根据历史单点差异给当前八档编造误差数字。

这轮不提出或应用修复补丁，不改变当前性能分档。后续若开展精度修正，
应以本文编号逐项固定输入/前态/版本，先隔离边界，再比较高精度参考与实际Native；
不能用“原SPMD逐bit通过”替代Native精度，也不能用“更接近Native”替代真实误差评价。

## 源码与证据链接

下列相对链接可从本仓直接打开；行号采用第1节固定快照。CANN/SDK源码不在本仓，
因此保留实际本地路径及版本，不冒充可移植的仓内文件。

[p-hc]: deepseek_v4_flash_csa_spmd_tnd/hc_stages.py
[p-rms]: deepseek_v4_flash_csa/rmsnorm.py
[p-qkv]: deepseek_v4_flash_csa_spmd_tnd/qkv_stages.py
[p-q]: deepseek_v4_flash_csa_spmd_tnd/query_projection_stages.py
[p-kv]: deepseek_v4_flash_csa_spmd_tnd/kv_stages.py
[p-cproj]: deepseek_v4_flash_csa_spmd_tnd/compressor_projection_stages.py
[p-pool]: deepseek_v4_flash_csa_spmd_tnd/compressor_pool_stages.py
[p-cout]: deepseek_v4_flash_csa_spmd_tnd/compressor_output_stages.py
[p-icout]: deepseek_v4_flash_csa_spmd_tnd/indexer_compressor_output_stages.py
[p-iq]: deepseek_v4_flash_csa_spmd_tnd/indexer_query_stages.py
[p-group]: deepseek_v4_flash_csa_spmd_tnd/indexer_group_stages.py
[p-score]: deepseek_v4_flash_csa_spmd_tnd/indexer_score_stages.py
[p-topk]: deepseek_v4_flash_csa_spmd_tnd/indexer_topk_stages.py
[p-threshold]: deepseek_v4_flash_csa_spmd_tnd/topk_threshold_stages.py
[p-publish]: deepseek_v4_flash_csa_spmd_tnd/indexer_publish_stages.py
[p-attn]: deepseek_v4_flash_csa_spmd_tnd/attention_stream_stages.py
[p-oa]: deepseek_v4_flash_csa_spmd_tnd/output_projection_a_stages.py
[p-ob]: deepseek_v4_flash_csa_spmd_tnd/output_projection_b_stages.py
[p-icache]: deepseek_v4_flash_csa_spmd_tnd/indexer_cache_stages.py
[p-joint]: deepseek_v4_flash_csa_spmd_tnd/joint_prefix_stages.py
[p-config]: deepseek_v4_flash_csa/config.py
[p-geometry]: deepseek_v4_flash_csa_spmd_tnd/request_geometry.py
[p-plan]: deepseek_v4_flash_csa_spmd_tnd/attention_plan_stages.py
[p-adapter]: deepseek_v4_flash_csa_spmd_tnd/native_adapter.py
[p-storage]: deepseek_v4_flash_csa/native_storage.py
[p-signals]: deepseek_v4_flash_csa_spmd_tnd/signals.py
[p-sync]: deepseek_v4_flash_csa_spmd_tnd/phase_sync.py
[p-scale-plan]: deepseek_v4_flash_csa_spmd_tnd/indexer_scale_plan_stages.py
[p-prefix]: deepseek_v4_flash_csa_spmd_tnd/vector_prefix_stages.py
[p-binding]: deepseek_v4_flash_csa_spmd_tnd/binding.py
[p-graph]: deepseek_v4_flash_csa_spmd_tnd/graph_contract.py
[p-service]: deepseek_v4_flash_csa_spmd_tnd/service.py
[p-leaf]: deepseek_v4_flash_csa_spmd_tnd/leaf_scale_layout_stages.py
[n-hc-tiling]: ../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp
[n-hc-stage]: ../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h
[n-hc-base]: ../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h
[n-hc-cube]: ../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h
[n-post]: ../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h
[n-layernorm]: ../layernorm.py
[n-linear]: ../linear.py
[n-cv]: ../cv_linear.py
[n-dsa]: ../../attention/dsa_v1.py
[n-device]: ../../device/device_op.py
[n-q-rms]: ../triton/rms_norm.py
[n-qr-normal]: ../../../csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_normal_kernel.h
[n-qr-single]: ../../../csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_single_row_kernel.h
[n-qr-helper]: ../../../csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_helper.h
[n-rope]: ../../../csrc/attention/inplace_partial_rotary_mul/op_kernel/rotate_interleaved_split_bsn_pad.h
[n-cmp-cube]: ../../../csrc/attention/compressor/op_kernel/arch32/compressor_block_cube_perf.h
[n-cmp-vec]: ../../../csrc/attention/compressor/op_kernel/arch32/compressor_block_vec_perf.h
[n-cmp-soft]: ../../../csrc/attention/compressor/op_kernel/arch32/soft_max.h
[n-cmp-reduce]: ../../../csrc/attention/compressor/op_kernel/arch32/compressor_vector_comm.h
[n-cmp-rms]: ../../../csrc/attention/compressor/op_kernel/arch32/rms_norm.h
[n-qli-cube]: ../../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_service_cube.h
[n-qli-vec]: ../../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_service_vector.h
[n-qli-sort]: ../../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_vector.h
[n-qli-kernel]: ../../../csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch32/quant_lightning_indexer_kernel.h
[n-sa-cube]: ../../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_cube.h
[n-sa-vec]: ../../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_vector.h
[n-sa-tiling]: ../../../csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.h
[retained]: ../../../tests/pypto_test/csa_spmd_tnd_eight_20261022/retained_output_rectangle.json
[sdk]: ../../../tests/pypto_test/csa_shared_toolchain/manifest.json
[codegen]: ../../../tests/pypto_test/csa_spmd_tnd_eight_20261022/output_rectangle_production_v1_codegen.json
[performance]: ../../../tests/pypto_test/csa_performance_20261023/PERFORMANCE.json
[history]: ../../../tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md
[tnd-log]: DSV4_CSA_SPMD_TND_IMPLEMENTATION_LOG.md
[goal-updates]: ../../../tests/pypto_test/csa_spmd_tnd_20261020/GOAL_UPDATES.md
[tdiv-history]: ../../../tests/pypto_test/PTO_ISA_A3_TDIV_HIGH_PRECISION_REPRO.md
[integration]: ../../../tests/pypto_test/csa_spmd_tnd_eight_20261022/output_rectangle_integration.json
[runtime-validation]: ../../../tests/pypto_test/csa_spmd_tnd_eight_20261022/output_rectangle_v3_runtime_validation.json
[hca-evidence]: ../../../tests/pypto_test/hca_tnd_mpmd_20261022/NATIVE_OUTPUT_PRECISION.md
[native-report]: /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_five_way_latest_20261020/results/h131072_b16/native_v1/report.json
[tnd-stages]: ../../../tests/pypto_test/csa_spmd_tnd_20261020/runs/mixed_261_h8192_service_stages_v5/result.json
[tnd-attribution]: ../../../tests/pypto_test/csa_spmd_tnd_20261020/runs/mixed_261_h8192_attribution_v4/result.json

本地 SDK/安装 CANN 的补充定位：

- PTO-ISA A3：/data/pyptouser/qinchuanyu/pto-eager/pto-isa-src/include/pto/npu/a2a3/
  TColSum.hpp:19–84、TUnaryOp.hpp:283–342、TDiv.hpp:18–82、TDivS.hpp:56–101、
  TMatmul.hpp:40–75；同include根下common/pto_instr.hpp:776–783为TRECIP映射。
- 当前代表生成码：
  tests/pypto_test/csa_spmd_tnd_eight_20261022/runs/compile_output_rectangle_production_C04_v1/kernel/ptoas/dynamic_csa_full.cpp；
  HC三参数TRSQRT约50755行，QK K128 tile及累加约33084–33170行。
- CANN9.2根目录见1.2；内置源码在
  opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/：
  rms_norm/rms_norm_merge_n.h:196–290、rms_norm/reduce_common.h；
  dynamic_quant/dynamic_quant_single_row.h:133–180、dynamic_quant/dynamic_quant.h:143–175；
  quant_batch_matmul_v3/quant_batch_matmul_v3_pertoken.h:492–536、
  quant_batch_matmul_v3/quant_batch_matmul_v3_pertoken_opt.h:494–586。
- Native入口对应 vllm_ascend/models/deepseek_v4.py:986–1000、
  csrc/torch_binding.cpp:1398–1427；npu_hc_pre_v2实际调用aclnnHcPre，
  不是单独HcPreInvRms/HcPreSinkhorn组合入口。
