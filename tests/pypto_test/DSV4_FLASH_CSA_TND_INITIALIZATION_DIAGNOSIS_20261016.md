# 完整PTO候选初始化与T240首差定位（2026-10-16）

## 当前状态

分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，正式数学源码仍为`259cf9a5f`。
此前[真实8K/B40回归](DSV4_FLASH_CSA_TND_FULL_PTO_MODEL_REGRESSION_20261016.md)的Native已完成；
CSA在FULL初始化阶段越界，尚未进入生成。本轮复用该Native，不跑Native2，不跑128K。
WKV恢复继续暂停。下面的缩小实验及候选均为独立诊断副本，尚未合入正式算术实现。

## 将运行时故障与精度首差分开

| 实验 | 任务 | 观测 | 限定 |
| --- | --- | --- | --- |
| L4，T240，真实层权重与合成history127 | task_20261016_212110_21519375438 | FULL流程未越界，但heads与output不同 | 不是实际8K整模型输入 |
| Q后RoPE观察 | task_20261016_214428_328244517179 | Q有17177个元素不同 | 后续v2/v3补齐观测活性检查 |
| Q前RMS/QR观察 | task_20261016_215122_360107222527 | QR有2处、scale有10处、Q前RMS有18166处不同 | Native compiled/raw及A/B/A检查通过 |
| 再向前观察Q_A | task_20261016_215704_376375318214 | Q_A最早44个BF16元素不同，输入Norm逐bit相同 | 新探针未改变上一版PTO的Q/QR/scale值 |
| 仅改Q_A为自然K遍历 | task_20261016_220925_403265421277 | Q_A、QR、scale、Q前RMS、Norm均逐bit相同 | 阶段诊断，不是最终heads/整模型通过 |
| 保留最终heads发布，仅改自然K遍历 | task_20261016_221422_417452422863 | T240的A/B共20项比较均逐bit相同 | 原任务exit1，末尾日志路径错误；T6未执行 |
| dummy几何单层 | task_20261016_222415_206065121 | T240→T6 warmup/capture/raw replay完成，无越界 | 仅内存安全，不作Native精度结论 |
| dummy几何21组异步调用 | task_20261016_223123_36980323431 | 两形状均完成，无越界 | 共享L4权重/cache，不是21个真实层 |

Q_A的44处差异仅出现在旧K遍历特殊N分组中。旧规则按9.0 MatMulV2实现校准，
当前环境为9.2及NZ输入；单因素实测支持此case使用自然K遍历。
这并不能证明所有形状应直接删除旧规则，更不能把数值首差认定为AIV UB越界的原因。
Q_A当前最大绝对差`0.00048828125`，其后QR差异可经量化向后传播，
因此后续修正仍应从最早边界开始，而非放宽最终输出阈值。

`natural-full-layer-v1`的数值和graph检查已完成，原始`pass=false/complete=false`与exit1保留。
独立的`postrun-log-gate-qualification.json`核验了实际上层`run.log`，
说明末尾错误为日志路径使用`out/t240/run.log`而实际日志在`out/run.log`。
这不将原任务改写为exit0，也没有补造T6结果。

## dummy实验的覆盖边界

模拟普通FULL初始化的seq_lens=6、positions全部127、共享page0、T240/B40与T6/B1。
真实metadata builder仍运行；记录req.slot_mapping及compressor compact slot，
没有把compressor的None slot误当成普通SWA slot。

第一版`task_20261016_222206_16870025898`在对None执行clone时退出，未执行PTO；
修复诊断脚本后第二版完成。第三版使用21个独立runtime私有buffer、
连续异步执行HC-pre/Norm/attention/HC-post，保留图与各buffer生命周期。

两个成功实验均使用重复合成hidden、零cache及共享L4权重，各形状新建cache。
它们未覆盖真实模型的各层权重、持久cache、HCA/MLP交错、跨rank执行及真实初始化全部状态。
所以阴性结果只缩小范围，不能宣称真实越界已修复。

## 缩小实验复现资产

[精简证据包](evidence_tnd_precision/full_pto_20261016/q-stages-evidence-v2.tar.gz)，SHA256：
`a29a98baf00973da58d0668f05fa11314da0ff8df6b9415eed3bd452e4493560`。
185项脚本、结果、日志和CPU摘要各有SHA，已逐项核验；不包含权重或大张量。
CPU摘要来自真实NPU快照，并非CPU golden。

解包后按`experiments/<实验>/run-*.sh`恢复独立实验目录及既有冻结source/环境引用，
先核对3357文件step05源清单与依赖身份，再经task-submit单卡、max-time780运行。
这些脚本使用原独立目录的绝对路径；迁移须显式更新并重新冻结，不能覆盖历史输出。
`q-stage`系列改变诊断输出槽，仅用于其声明的阶段比较；
最终attention对比使用`natural-full-layer-v1`，不能拿Q阶段脚本替代整层验收。

## 真实模型定位与最小入口修正（10月17日）

真实16卡诊断`task_20261016_235359_192943116698`在780秒上限内退出，未进入生成。
仍复用原Native参考。step05算术不变，仅在非capture程序前后同步、捕获内记CPU日志；
不作为性能测试。每层使用独立operators描述符，未覆盖共享callable。

- 12个rank停在L42/T6的`pto_input_norm`调用后同步；调用前同步成功，HC-pre完成。
- 这些rank已完成165个PTO程序边界：T240的84个、T6前20层的80个，以及L42 HC-pre。
- 其他4个rank完成168个程序边界。故障不要求所有rank同时发生。
- 报错仍为AIV VEC UB越界。由此定位到输入Norm程序，尚未把PC映射到内部具体子核。

静态复核发现，外部`input_norm_sum.cpp`入口缺少自动生成子核具有的Vector模式和全量掩码初始化。
4096列BF16→FP32转换的ISA整repeat路径不会自行建立该前置状态。
修正仅增加`set_mask_norm()`和`set_vector_mask(-1,-1)`，没有改归约或舍入顺序。

常规单卡测试使用L42真实Norm权重、T6非零随机输入，比较Native RMSNorm与PTO：

| 候选 | 任务 | 结果 |
| --- | --- | --- |
| 原入口 | task_20261017_000719_219282311307 | eager A、graph A逐bit相同；B replay同类UB越界，首错停止，未运行fixed |
| 仅标准入口初始化 | task_20261017_000817_220829129465 | exit0，eager及graph A/B/A的output、rstd共8项逐bit相同 |

正式修正文件与单卡实测`fixed.cpp`已逐字节核验相同。
这支持入口前置状态风险及修正有效性；仍需真实8K/B40确认初始化和192token，
不能把单卡结果替代整模型PASS。原失败输出和status未被改写。

[初始化与Norm证据包](evidence_tnd_precision/full_pto_20261016/norm-entry-evidence-v1.tar.gz)，
SHA256 `f326c5fab6cbf08f74206c9ecd9f347a97f65791eabc84042f00dbeb8f9ac5ed`。
53项文件，包括所有16rank初始化事件/日志、配置、诊断helper、单卡原始失败及成功结果和脚本。
该包不包含任何额外异常状态诊断；只执行了正常算子对拍。
