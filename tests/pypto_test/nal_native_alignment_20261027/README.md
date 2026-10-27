# 基于 nalinaly TND SPMD 的 Native 数值对齐

日期：2026-10-27。

## 分支和目标

- 当前分支：`dev/pypto-dsv4-csa-nal-tnd-native-align-20261027-codex`。
- 原始底座：nalinaly 的 `dsv4-flash-pto-v0.25.1rc1`，
  `1685da6914ae6196537d7f1a66e84e09a9c48ae5`。
- 旧精度分支 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex` 保留，
  其中已验收的 `11b380e3fc16efdc1cb85a0b2888720872c70655` 作为实现参考。
- 保留 nalinaly 的单入口、单 MIX SPMD、真正 TND 以及后续 Indexer/attention/输出流水；
  从最早误差模块逐项引入 Native 顺序修正。所有新增计算均为 DSL。

本文的模块精度结果不能替代完整 CSA、整模型逐 token 或性能验收。

## 固定测试条件

| 项目 | 值 |
| --- | --- |
| 服务器 | 227，独立的 yejia 工作区 |
| 权重 | `DeepSeek-V4-Flash-0731-w8a8`，正式编号 4 层 HC/Norm 权重 |
| 模块输入 | CPU seed=1048，BF16 `randn([24,4,4096])`；B 输入为 `A+0.125` |
| FULL 图 | 实际 T=24；不做请求 S6 补行；P16/P8 仅为内部 tile 容量 |
| Native | 冻结的 vLLM-Ascend 0.25.1 环境，`npu_hc_pre_v2` 调用融合 `aclnnHcPre` |
| 运行库 | CANN 9.2.0-beta.2；Torch 2.10.0；Torch-NPU 2.10.0.post2 |
| 确定性 | `torch.use_deterministic_algorithms(True)` 后设置 NPU level=1；`HCCL_DETERMINISTIC=true` |
| PyPTO | `d44925c17d9ff7d7fa8e5067e41c4b4f80bf3063` |
| PTOAS | `914863ed3d0eb3e92a4433a4a706a86cabf4c91f`，0.69 原生构建 |
| Simpler | `57e499238fd4ee3548947ad860be456aaeeb16b5` |
| 安装 | 复用现有 SDK/venv，通过源码命名空间绑定；未重装软件 |
| 单项时限 | `task-submit --max-time 780`，脚本内 `timeout 750s`，TERM 后最多 15 秒清理 |

checkpoint 中 `attn_norm.weight` 是 FP32；Native 构造的 RMSNorm 参数是 BF16，
加载时 `copy_` 到该参数。模块测试复现同一个加载边界，不把 FP32 checkpoint
直接当作根入口的 BF16 参数。权重 key、原始字节 SHA 和加载 dtype 在 JSON 中记录。

三个工具链提交与 nalinaly 的共用 manifest 相同；本分支仅把 manifest 中的
PTOAS 构建目录名由 `build` 绑定到现有的 `build-native`，没有修改工具链源码。
公共 bootstrap 新增可选的 `--native-plugin-root`，以冻结 Native 控制源码；默认行为不变。

## 已取得的模块结果

下表是相对融合 Native HC 的逐 bit 不同元素数，均为同输入、同权重。
Norm 一列比较相同 mixed 输入下的 Native Norm，避免把上游误差重复归因。

| 实现切片 | HC mixed | HC post gate | HC comb | 同输入 InputNorm |
| --- | ---: | ---: | ---: | ---: |
| nalinaly 原 HC 算术 | 218 | 72 | 383 | 0 |
| 仅改原始输入 RMS | 218 | 72 | 383 | 0 |
| 再改 HC 投影 | 2 | 0 | 249 | 2 |
| 再改四路求和树 | 0 | 0 | 0 | 2 |
| 再改 InputNorm | 0 | 0 | 0 | 0 |

证据分别来自 `hc-v2`、`hc-v3`、`hc-v4`、`hc-v5`。
原 HC 的 mixed relative L2 为 `7.093408173394753e-05`；投影修正后为
`2.037845026317391e-07`；求和树修正后为 0。
不能将原 HC 的百分比推成整层或整模型误差。

### 误差的具体来源

1. **HC 投影是当前 HC 中的主要误差来源。** nalinaly 采用四段 K4096 普通 FP32 MMA；
   修正采用 Native 对应的 22 段 K768（末段 256），显式
   `(bits+2048)&-4096` HF32 权重转换，以及 BF16 高低位布局保留的 8 乘积累加边界。
   保留 Native 原始权重，不引入反量化/再量化。
2. **原始 16384 维输入 RMS。** Cube 平方 partial 取对角后按 22 段顺序累加，
   使用 `sqrt` 和 `1/div`。只修此项没有消除 218 个 mixed 差异，因此没有把它误判为主因。
3. **四路求和树。** mixed 和 Sinkhorn 的列归一化使用
   `((a+b)+c)+d`，替换原来的 `(a+b)+(c+d)`。投影已修正时，这一步使 HC 三输出全部一致。
4. **mixed 后的 4096 维 InputNorm。** 单独采用 Native 的 fold2048→64/F8 归约树、
   sqrt 舍入校正、scalar div，以及先乘 inverse、再乘 Norm 权重的 BF16 边界。
   它与前述 16384 维 RMS 是不同计算，不能互相替代。

旧精度版 `hc_pre_test` 先在同一个新 SDK 上通过融合 Native 的三输出逐 bit 门禁，
才用于解释上述策略。pre/post sigmoid 的 `recip` 在本组同输入下没有产生 post 差异；
本轮没有仅凭源码不同就改动它。

### 观测可信性

- Native 融合 HC 重复执行一致。
- 各模块 FULL replay 与 eager 一致；回放前毒化所有外部中间量和输出。
- HC 修正候选使用同一张图 A→B→A；三次 HC 输出均匹配各自 Native。
- 直接读取 BF16 输入的打包，沿用原布局并在 UB 无损加宽。完整 P16 buffer
  与旧 `wide→pack` 及独立 CPU 的每 8 元素重复构造逐字节一致，A→B→A 全部通过。
- 最后两项防止 capture 遗留 packed_input 或旧平方 partial 导致假通过。
- InputNorm 的 Native 比较在本轮确认 A 输入为 0 差异；B 的图/eager 检查通过，
  后续完整 SPMD 验证继续检查 Native 边界，不能扩大成所有输入的精度资格。

## 一个不能混用的参考接口

最初的 `rms-v1` 调用独立 `npu_hc_pre_inv_rms`。T24 下，nalinaly 写法有 5 行不同，
22 段候选有 6 行不同，最大距离分别为 1/2 ULP。任务在这个首差处停止扩大。
该接口走独立 Vector 归约，不能充当融合 HcPre 内部 inverse 的逐位金值。
所以该负结果既没有被记为“修复通过”，也没有被用来否定随后通过融合 HC 门禁的方案。

## 接回完整 SPMD 的合同

- 外部生产根仍是 55 个 Tensor；TopK debug 根仍是 57 个 Tensor。
- 新增内部 square、packed input、packed weight 和 Norm rstd；只把 HC partial 改成 22 份，
  Indexer 权重的四份 partial 不变。
- 48 个 Vector 完成 widen、直接输入打包和权重打包后全部发布 INPUT。
- Cube 等 INPUT，计算 HC 投影和平方 partial，发布 PARTIAL，然后才等待 NORM。
- Vector 等 24 个 PARTIAL，计算 RMS/投影归约；保留 RAW 的 M16 生产者到 M8 消费者等待。
- Native Norm 由 mixed 的同一 M8 owner 执行，完成后发布 NORM。
- 旧四份 `consumer_reduce` 在本分支明确拒绝，避免错用 owner 或归约份数。
- 追加内部参数保留 Cube cache-policy 的原参数索引；新增 scratch 的生命周期覆盖单个 MIX。

完整入口的无设备编译及 ABI 加载已通过：生产 55、debug 57、观测 65 个 Tensor。
Host 对 T>256 回退 Native，且拒绝旧四份 consumer-reduce。六项本地测试检查
真实 T17 元数据准入、容量拒绝、旧归约拒绝、55/57 外部接口和权重缓存索引。
T≤256 是容量限制，本轮设备有效点仅为 T24。

### 完整 SPMD 的设备定位

`spmd-production-v3` 与 `spmd-production-v4` 已实际执行 8K/B4/T24 的完整
SPMD：生产 FULL replay 与 eager 的八个外部输出逐 bit 一致，cache 写入保护通过。
使用 Native builder 生成五组 metadata 和真实编号 4 层权重；hidden/cache 历史为
seed=1024 的合成值。它与前述 seed=1048 的 CPU 模块输入是两组独立测试。

`spmd-observe-v1` 的观测根先通过生产八边界门禁，再通过自身 eager/FULL 一致性：

| 对比 Native 模块 eager 的边界 | 逐 bit 不同元素数 |
| --- | ---: |
| HC mixed / post gate / comb | 0 / 0 / 0 |
| InputNorm | 0 |
| QA 投影落 BF16 | 0 |
| QR INT8 | **1219，首差；最大绝对差 1** |

到 QR 首差即停止解释后续模块，当前没有完整 CSA 对 Native、整模型 token 或性能结论。
v1 的 QA 参考直接调用 `wq_a`，compact cos/sin 未纳入跨任务身份门禁；因此随后
收紧为 Native 实际多流路径 `cv_wq_a.quantize→matmul`，并补齐 compact 三项 readonly。
`spmd-production-v4` / `spmd-observe-v2` 用于复核这个收紧后的口径，不能把 v1
声称为 Native FULL 全层 hook 验收。

复核 v2 已完成，HC/Norm/QA 仍为 0 差异，QR 首差仍为 1219 个 INT8 元素相差 1。
生产快照和观测分别耗时约 3 分钟；任务均正常退出并释放设备。

| 任务 | 用途 | 结果 |
| --- | --- | --- |
| `task_20261027_185058_29073298217` | HC 最终切片、布局、A→B→A | HC/Norm A 输入逐 bit 一致 |
| `task_20261027_191743_41055407582` | 完整生产 SPMD v3 | FULL/eager 八边界一致，保护通过 |
| `task_20261027_192253_16858527307` | 观测 v1 | QR 首差，后续停止 |
| `task_20261027_193251_68874526968` | 完整生产 SPMD v4 | FULL/eager 八边界一致，保护通过 |
| `task_20261027_193650_86679632031` | 收紧参考后的观测 v2 | 同样在 QR 首差，前五个边界一致 |
| `task_20261027_194116_10660281731` | 只恢复 QA BF16 边界 | QR INT8 差异清零；scale 仍有 14 行差异 |

### QR 首差的单因素证据

`qr-qa-v1` 直接复用观测 v2 保存的 QA FP32 accumulator 与同一 BF16 Norm 权重。
不改生产 helper，只把输入从原 accumulator 换成 `QA.bfloat16().float()`：

| QR 输入 | INT8 不同元素数 | FP32 scale 不同行数 | scale 最大绝对差 |
| --- | ---: | ---: | ---: |
| 原 FP32 accumulator | 1219 | 24 | 1.2881e-5 |
| 恢复 QA BF16 边界 | **0** | **14** | 1.3970e-9 |

两组 FULL/eager 均逐 bit 一致，未舍入组还精确复现完整生产图的 QR 输出。
因此 QA 缺少 BF16 边界解释了本组全部 QR INT8 差异，但剩余 scale 仍未对齐。
这项诊断仅约 14.3 秒。`c69ca959` 检查点尚未把 QR 修正合入生产 helper；下一步针对
Native 的归约树、normalized amax 和 scalar div 顺序修正 scale，不跳过它。

### QR 的 Native 数值顺序修正

新增 `q_native_stages.py`，保留 nalinaly 的 M8 worker、四个输出及发布/等待顺序，
只恢复以下数值边界：QA 先落 BF16；1024→512→256→128→64 求和树；
sqrt 舍入与 scalar reciprocal；先算 `(QA*inverse)*gamma` 再取 amax；
quant/dequant 两次 scalar div。INT32 rint→FP16→INT8 转换链没有改变。
helper 全部 inline，不创建子任务，不调用 Native。

`qr-native-v1` 在同一组 QA/gamma 下完成三组对拍，约 26.6 秒：

| QR 实现 | INT8 差异 | scale 差异 | FULL/eager |
| --- | ---: | ---: | --- |
| nalinaly 原 helper、原 QA | 1219 | 24 | 一致 |
| 原 helper、只恢复 QA BF16 | 0 | 14 | 一致 |
| 新 Native 顺序 helper | **0** | **0** | 一致 |

对应任务 `task_20261027_194850_14909428416`。其中只有 QA BF16 已被单因素证明；
剩余四类顺序联合消除了 14 行 scale 差异，不能声称它们每一项都已被独立证明必要。
首两次无设备编译暴露 Tensor/Tile 混用和缺少 Tile reduce scratch，已修复为显式
Vec load/Tile full/tmp_tile；`cpu-qr-native-v3` 编译、ABI 加载通过，未改 SDK。

完整生产 debug 根 57 参数和扩展观测根 67 参数均通过无设备编译。
`spmd-production-v5`（`task_20261027_203016_381687129968`）已实际运行：
FULL/eager 八边界一致、cache 保护通过，耗时约 188.3 秒。生产 Vector 已接入
新 helper；后续 Native 参考覆盖范围由观测结果单独确认。
上述秒数均是含首次编译的任务墙钟，不是算子延迟，不能据此比较性能。
新观测根额外暴露 WKV accumulator 和 Norm/RoPE 后 KV，先做生产等价门禁，
再按 Native 实际 `cv_wkv.quantize→matmul→kv_norm→inplace_partial_rotary_mul`
检查首差。最后一项是 Norm+RoPE 组合，不能从失败直接归因于 RoPE。

`spmd-observe-v3`（`task_20261027_203549_720244997`）约 188.7 秒后正常退出。
生产等价门禁、观测 eager/FULL、保护区均通过；与 Native 模块 eager 的比较为：

| 同输入边界 | 逐 bit 不同元素数 | 最大绝对差 |
| --- | ---: | ---: |
| HC mixed / post gate / comb | 0 / 0 / 0 | 0 |
| InputNorm | 0 | 0 |
| QA 落 BF16 | 0 | 0 |
| QR INT8 | 0 | 0 |
| QR scale FP32 | 0 | 0 |
| WKV 落 BF16 | **4 / 12288，首差** | **7.6294e-6** |
| KV Norm+RoPE、Indexer、attention、O-proj、HC-post | 未作 Native 精度验收 | — |

WKV 首差坐标为 `[3,188]`、`[3,444]`、`[14,222]`、`[22,302]`；
发现后不再比较后续模块，未提交整模型或 128K 任务。
旧精度版存在按 Native M/N 分档的 K256 累加次序表；当前 nalinaly WKV helper
采用统一的 K512 次序和 N128 分块。它是下一项需要用同一 Normed 输入、实际权重
做单因素验证的方向，当前没有把这四处差异仅凭源码推断归因或套用旧表。

本地合同测试现为 7 项，额外检查观测根调用的 `_csa_core_group` 与生产根 AST 相同。

`qr-tail-v1`（`task_20261027_204311_42718926601`）约 12.3 秒正常结束：
新 QR helper 在 T17、T1 下的 INT8/scale 均与 Native 逐 bit 一致，
FULL/eager 一致，输出前后保护区未改变。没有把 T 补为 6 的倍数。
这只是 QR 阶段的短尾行资格，不能替代完整 CSA 的混合请求验收。

## WKV 累加次序与完整入口复核

以 `4c521b37` 的观测 v3 保存件为起点。先从正式 safetensors 读取同一层的
`layers.4.attn.wkv.weight`，在相同 Normed 输入上调用 Native BF16 linear，
必须逐 bit 重现实际 `cv_wkv` 保存金值；原 helper 的 FP32 accumulator 也必须
逐 bit 复现完整 SPMD 保存值。两道门禁均通过后才解释以下对比。

| WKV 切片，8K/B4/T24 | BF16 不同元素数 | 最大绝对差 |
| --- | ---: | ---: |
| nalinaly 原 N128/K512 自然顺序 | 4 / 12288 | 7.6294e-6 |
| N32/K256，仍按自然顺序 | 4 / 12288，同四处 | 7.6294e-6 |
| N32/K256，Native 分列 paired 顺序 | **0** | **0** |

仅更改分块没有消除首差，相同分块下更改 K 次序使差异清零。
每个原 N128 logical producer 内部顺序处理四个 N32 子块；M32/M64/M16
条带、physical worker 旋转、phase 发布与等待都保留。新增 helper 不产生任务或信号。

Native 在当前小 M 环境的规律是：M16/32/48 使用自然 K 顺序，其余 M<64 使用
分列 paired 顺序。M1…63、两种输入共 126 点，先验证 constexpr 两条链；
随后另用真实生产 runtime dispatcher 复核。输入固定为正式层权重及保存的 Normed
行重复构造，B 输入为 FP32 `+0.125` 后转 BF16，不是所有权重/输入穷举。
**M≥64、joint BF16 路径仍保留原实现，未取得新的 Native 数值资格。**
旧精度版的 whole-model static M6 另有 Native static-package 分档；不能拿本轮 eager
模块金值证明其静态分档，也不能直接继承其他 CANN/模型形状的 K 次序表。

| 设备任务 | 范围 | 结果 | 墙钟 |
| --- | --- | --- | ---: |
| `task_20261027_211041_18132473134` | WKV 三臂 | 4→4→0；FULL/eager、guards 通过 | 38.8 s |
| `task_20261027_211948_246064613974` | constexpr 小 M | 126/126 通过 | 29.5 s |
| `task_20261027_214558_366792429085` | 生产 runtime 小 M | 126/126 通过 | 37.0 s |
| `task_20261027_213416_311589611052` | 生产 SPMD v6 | 八边界 FULL/eager 一致，guards 通过 | 180.2 s |
| `task_20261027_213838_33167561640` | 观测 v4 | 生产门禁通过；WKV 修正生效 | 182.8 s |

完整57参数 debug入口、67参数观测入口先通过无设备编译/ABI加载。
观测 v4 中 HC 三输出、InputNorm、QA BF16、QR INT8/scale、WKV BF16 均为0差异。
新的首差是 **KV Norm+RoPE：3410/12288，max_abs=0.0078125**，发现后停止后续模块。
这项组合检查尚未区分 Norm 与 RoPE；不能把它直接称为 RoPE 算法错误。

以上为同输入模块 eager Native 金值与完整生产 FULL 的内部观测对比，尚不是
Native FULL 全层、整模型逐 token 或性能验收。所有任务正常退出、释放设备。

### KV Norm/RoPE：两处 BF16 边界的因果验证

`kvnorm-boundary-v1` 复用观测 v4 的 **同一 paired WKV FP32 accumulator**、
实际 BF16 gamma 和 Native cos/sin。Native `npu_rms_norm` 及
`inplace_partial_rotary_mul` 分别先重现保存的实际模块输出；原 nalinaly helper
也逐 bit 重现生产 KV。只有两端复现通过后才比较三个切片：

| 切片 | NOPE 448 列不同元素数 | 全部 Norm+RoPE 不同元素数 |
| --- | ---: | ---: |
| nalinaly 原 helper，直接消费 FP32 WKV | 2816 | 3410 |
| 只把 WKV accumulator 先落 BF16 | **0** | 422，全部位于 RoPE 列 |
| 再恢复 Norm→RoPE 的 BF16 边界 | **0** | **0** |

因此，本组 3410 个差异由缺失的中间 BF16 发布边界消除，不需要更改 RMS
归约树、sqrt/div、RoPE 算法或同步。新增 `kv_native_norm_stages.py` 保留原
M32所有权、八段K64归约和尾块有效区，仅在六个输入读取点和两个Norm→RoPE点
恢复BF16往返。独立审查确认：删除这八处转换后，函数体AST与原helper相同。

任务 `task_20261027_214850_381191622910` 约20.4s正常退出。
T1/17/32/33/48/63 的 Norm/RoPE 也全部与 Native 逐bit一致，FULL/eager和guards通过。
这些形状测试只验证Norm/RoPE，不额外证明对应M的WKV投影。
生产接入仅替换 `vector_prefix_stages.py` 的 helper import，等待和发布均不改变。
后续完整观测新增Q投影INT32与Q Norm/RoPE后BF16两处scratch，根参数为69；
生产根仍55、debug根57，没有新增生产外部参数。

### KV 接回完整 SPMD：已通过，继续停在 Q 首差

`spmd-production-v8`（`task_20261027_220512_4590348611`，189.3s）
FULL/eager八边界一致、guards通过。57参数生产debug和69参数观测根均先通过
无设备编译/ABI加载；观测根新增的两处返回值使用独立after名称，避免重绑Out描述符。

`spmd-observe-v5`（`task_20261027_221106_7533633309`，193.3s）先通过相同
输入/metadata/cache和16项生产文件SHA门禁，观测与生产八输出逐bit一致，
自身FULL/eager和保护区均通过；随后取得以下实际Native模块金值结果：

| 边界 | 逐bit不同元素数 |
| --- | ---: |
| HC mixed/post/comb、InputNorm | 0 |
| QA BF16、QR INT8/scale | 0 |
| WKV BF16、KV Norm+RoPE | **0** |
| Q投影/Norm/RoPE组合 | **220781 / 786432，首差** |

Q组合max_abs=0.0625。保存了实际Native的Q投影BF16、Q Norm BF16、最终RoPE，
以及相同生产Q INT32 accumulator、实际/准备后的权重scale。没有继续解释Indexer、
attention和输出链，没有整模型token或性能结论。该Native参考来自实际W8A8 decode
分支的 `npu_quant_matmul→DeviceOperator.apply_dsa_q_rms→inplace_partial_rotary_mul`；
这是模块eager金值，不是Native FULL全层hook。

## 复现

227 工作区：`/data/pyptouser/yejia/vllm-nal-tnd-native-align-20261027-codex`，
checkout 位于 `src/`。工作区的 `env-dsv4-0251rc1.sh` 包装既有隔离 Native 环境；
`.worktrees/` 链接到 manifest 对应的三个现有 SDK 源码/构建目录。
不应将新分支安装到旧验收环境。

既有包装内容为：

```bash
source /data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex/query-sort-perf-20261022-codex/env.sh
export PATH="/data/pyptouser/yejia/csa-spmd-sdk-20261020/sources/pypto/.venv/bin:$PATH"
```

```bash
cd /data/pyptouser/yejia/vllm-nal-tnd-native-align-20261027-codex
source src/tests/pypto_test/nal_native_alignment_20261027/env.sh
TEST_DIR="$PWD/src/tests/pypto_test/nal_native_alignment_20261027"
BOOTSTRAP="$PWD/src/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py"

# 占卡前编译；每次使用全新输出目录，不覆盖历史凭证。
python "$BOOTSTRAP" --shared-toolchain --native-plugin-root "$SOURCE_ROOT" \
  "$TEST_DIR/compile_rms.py" "$PWD/cpu-reproduce-hc" hc-norm

# 完整四个切片及布局验证；只在首差模块上诊断，不跑大模型矩阵。
task-submit --device auto --max-time 780 \
  --env PYTHONNOUSERSITE=1 --env HCCL_DETERMINISTIC=true \
  --env PTOAS_ROOT --env ASCEND_HOME_PATH \
  "bash $TEST_DIR/run_rms.sh reproduce-hc hc"
```

`probe_hc.py` 的 `after-fold` 参数只复测最后两个切片；提交时传
`--env HC_PROBE_PHASE=after-fold`。发现修正候选首差时保存观测再停止。
`rms-v1` 是不同参考接口的排查，需显式使用 `run_rms.sh <新标签> rms`。

完整 SPMD 的 `probe_spmd.py` 分成两个任务：先生产 FULL 图快照，
再设置 `SPMD_PRODUCTION_REFERENCE=<前项输出>/production.pt` 跑观测图。
第二项逐字节检查已记录的 hidden、positions、只读 metadata（含 compact cos/sin/slots）、
初始 cache，以及当前十六个生产文件的 SHA 一致（包含 WKV/KV helper、dispatcher、调用点和 Q 消费代码），
再比较观测根与生产根的八个外部输出；通过后才解释 HC/Norm/QA/QR 的 Native 首差。
这样不把两个完整根的首次编译放到同一个占卡任务。
权重/SDK 通过固定路径、提交和加载参数记录，当前不是全部文件的逐字节身份门禁。

QR 单因素复现时，必须设置
`QR_PROBE_REFERENCE=<工作区>/results/spmd-observe-v2/observation.pt`，
提交 `run_rms.sh <新标签> qr`。该文件由观测任务在 QR 首差前保存，包含 QA accumulator、
Native BF16 QA、Native QR/scale 和实际 gamma；无需重新加载整层权重。
若保存件不存在，先用 `c69ca959` 检查点的生产→观测两任务重建。当前
`spmd-observe-v3` 的 QR 已修复，不能用于未修 helper 的 production_qr 门禁。
历史 v4/v2 检查点只比较七个 HC 文件 SHA；当前 v5/v3 对应上述十个文件。

## 测试侧问题记录

| 问题 | 处理 |
| --- | --- |
| `hc-v1` 把 FP32 checkpoint Norm 直接传给 BF16 根 | 复现 Native 的 BF16 参数加载边界；随后模块测试通过 |
| BF16 尾行直接 Scalar 写零 dtype 不符、fillpad 类型冲突 | 改为 UB 内无损加宽后沿用已验证的清零布局；未改 SDK |
| 完整根首次 CPU 编译缺少 dep-call constexpr | 所有调用显式传递 constexpr，重新编译通过 |
| 观测根动态维度重绑定被 frontend 拒绝 | 返回别名用独立变量名，实际 GM 存储不变；65 参数编译通过 |
| `spmd-production-v1/v2` 使用普通 CSA 的权重准备根 | 按 nalinaly `service.py` 显式传 SPMD `root_function`；不是数值结果 |

这些失败均保留原始目录，不覆盖或算作精度通过。

### WKV / KV 单因素的复现顺序

为重建KV三臂起始保存件，使用KV生产接入前的 `6fa5c09f` 算子快照，测试脚本用本文当前版本（含导入修复）覆盖；在独立目录执行
生产FULL快照，再用其 `production.pt` 设置 `SPMD_PRODUCTION_REFERENCE` 运行观测。
这个快照已修WKV，原KV helper仍会在Norm/RoPE处停止并保存 `observation.pt`。
生产命令沿用上面的环境和task-submit时限，分别用全新label，不覆盖历史输出。

- WKV：设置 `WKV_PROBE_REFERENCE=<观测v3的observation.pt>`，
  提交 `run_rms.sh <新label> wkv`。该保存件必须来自 `4c521b37` 的生产→观测，
  含原nalinaly WKV accumulator；设置 `WKV_PROBE_RANGE=small` 可增加runtime分档126点。
- KV：设置 `KVNORM_PROBE_REFERENCE=<6fa5c09f观测的observation.pt>`，
  提交 `run_rms.sh <新label> kvnorm`。第一臂要求重现未修KV的生产输出；
  不能用已经修正KV后的观测件替换这项因果起点。
- 两组均先用 `compile_rms.py` 的 `wkv-base/natural/paired/dispatch` 或
  `kvnorm-base/native` 选项做无设备编译、ABI加载，再占卡。

完整Q扩展观测首次启动v7因提前import DeviceOperator发生Native循环导入，
任务在设备初始化前退出，未产生数值结果。已改为复用实际impl模块选定的operator，
并新增 `check_probe_imports.py <receipt.json>`，在公共bootstrap下先无设备检查顶层导入。
该启动失败保留为控制问题，不记作精度失败。
