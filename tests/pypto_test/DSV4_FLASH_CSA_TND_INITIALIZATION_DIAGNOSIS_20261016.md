# 完整PTO候选初始化与T240首差定位（2026-10-16）

## 当前状态

分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，定位起点数学源码为`259cf9a5f`；后续最小入口修正已提交`e267e4e62`。
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


## 修正后的真实初始化复测

`task_20261017_001302_228617232359`占卡6分54秒，exit1。
16rank都完成168个非capture同步程序边界，且捕获执行到最后CSA层；没有507015。
实际JIT导出的`extern/0/input_norm_sum.cpp`已核对包含两行修正。

本轮没有获得token结果：rank3在`OFFLINE_MODEL_READY`之后，旧测试入口
`mixed.py`硬编码断言变体为`tnd_precision`，在生成前拒绝`tnd_full_precision`。
其他rank在父进程关闭后产生EOF/connection-refused；这些不是首因，也不是网络服务故障。
原始任务exit1保留，不改写成整模型PASS。

后续在独立step07测试快照中，将该身份校验提取成纯CPU函数，明确映射两类精度包，
继续要求21个CSA均来自指定PTO runtime、20个HCA保持Native。
12项CPU正负控覆盖正确变体、错包、缺层、HCA误替换、未知性能变体、空结果等；全部通过。
生成、接受事件、实际入场和数值验收代码均未改动。
step07数学与`e267e4e62`相同；只新增未进入正式算术提交的测试入口overlay，不能把它描述成完整Git checkout。

[修正后初始化证据](evidence_tnd_precision/full_pto_20261016/norm-entry-evidence-v2.tar.gz)，
SHA256 `9b99a736c0c7292b2c358c41a55b9721411f4f849dee71f4ec041d203889245c`，
62项文件均已核验，包括全部16rank日志/阶段事件、step05/06清单、旧断言首错与新测试guard。
下一轮正常异步复测不带阶段同步，仍只对一次既有Native，不跑Native2；结果另附。

## 正常异步8K/B40、192 token：通过

任务`task_20261017_002449_25681559404` exit0，task-submit报告占卡 **9分47秒**，
低于15分钟；设置max-time=780。没有初始化阶段逐程序同步，也未重跑Native2。
完整数学候选为`e267e4e62`，测试使用step07独立快照，显式保留上节的测试入口overlay。

[完整CPU验收JSON](evidence_tnd_precision/full_pto_20261016/model-8k-production1-qualified.json)：

| 验收项 | 结果 |
| --- | --- |
| 真实模型及配置 | DeepSeek-V4-Flash-0731-w8a8，TP1、DP=EP16，8K/B40、192 token，FULL[6,240] |
| 请求与输出 | 640条请求、122880个输出token，全部完成 |
| Native次数 | 复用前述一次Native；没有Native2 |
| 16rank实际GPU输入与入场轨迹 | 全部一致 |
| 16rank完整token与DSpark接受事件 | 全部一致 |
| rank0 outer hidden、aux、MTP位摘要 | 51轮全部逐bit一致 |
| 实际峰值请求 | 每rank40条，零抢占 |
| 实际CSA替换 | 21层在T6/T240均记录到PTO warmup与capture选择，后续实际FULL240 replay |
| HCA | 20层保持Native |
| 平均接受草稿数 | 3.703590992；没有强设为3.8 |
| 接受0～5个草稿计数 | [1696,2432,2816,2592,2240,14512]，与Native完全一致 |
| 范围外 | 本轮未做内部逐层hook、性能测量、128K、全部CSA模块PTO恢复 |

raw输入记录因停机前空尾不同，CSA各rank为52或54轮；
验收用严格非活跃空尾判据提取51轮生成窗口，没有任意截断活跃差异。
完整记录与窗口审计脚本均在证据包。

本轮`strict_ready_reuse=false`：ready_hits=0，object_hits=248，部分rank首次JIT生成/二进制构建各2次。
所有rank的bypasses、invalid_entries、storage_errors、forced_rebuilds均为0。
没有重新编译或安装vLLM/PyPTO/Simpler框架；该次初始化/首次JIT开销不作为性能结论。

### 可复现材料

[两臂完整数值证据](evidence_tnd_precision/full_pto_20261016/model-8k-production1-evidence.tar.gz)，
SHA256 `1b92fac9c439a53d87174e6c7db1c251863ea80b890ae3532d12ee6833cb0f2b`，
276项文件逐项核验。包含两臂全部16rank的mixed结果、请求事件、输入轨迹与审计、
rank0 outer位摘要、Native派生参考及绑定、3357源清单、helpers及最终qualifier。
不含模型权重、venv、完整bank大张量、设备dump；需沿用既有且SHA校验通过的模型/环境/bank。
与初次回归精简包不同，此包保留了完整跨rank数值原件。

step07源码manifest SHA256：
`83ad8550a84e8d51346671648bbbba33e868e373e9c01513a1b063b8fd15526d`。
最终qualifier逐文件核对：相对Native的step05快照，仅外部Norm两行入口初始化与
`mixed.py`测试变体校验不同。它核验两行补丁的精确字节、已测guard副本，
并以AST确认原batch生成/观察主体和collect/peer等待函数没有变化。
身份差异是显式校验的，未改写Native配置来伪造相同SHA。

在原227独立资产目录复测（新的输出名，保留历史数据）：

```bash
JOB=/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/full-pto-model-8k-b40-20261016
# 先按证据helpers/environment核验Python、SDK/PTOAS、step07源SHA、bank与Native派生绑定。
# 使用原生bootstrap，无初始化逐程序同步：
task-submit --device auto --device-num 16 --max-time 780 --ptoas 0.63 \
  "bash $JOB/production-v1/run-case.sh precision-witness <新的CSA输出名>"
# 任务结束后，在同一环境中仅做CPU验收；不提交Native2：
python "$JOB/production-v1/qualify_production.py" \
  /data/pyptouser/yejia/lc128/full-pto8k-native-20261016 \
  /data/pyptouser/yejia/lc128/<新的CSA输出名> --job "$JOB" --output <新的验收JSON>
```

脚本仍绑定该历史Native和具体源快照；迁移或新建Native时必须显式更新资产路径/绑定和身份校验，
不能把另一轮参考当成本轮。现在可在这个已通过基准上继续恢复WKV投影。
