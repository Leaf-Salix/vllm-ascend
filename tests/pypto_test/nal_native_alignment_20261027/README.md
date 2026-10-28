# 基于 nalinaly TND SPMD 的 Native 数值对齐

日期：2026-10-27；Indexer完整接入复核更新于2026-10-28。

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

## Q 反量化、Norm 与 RoPE 的单因素结果

同一完整SPMD保存的INT32 accumulator、Native BF16投影、Norm及最终RoPE作为金值。
先重现Native Norm/RoPE，再核对反量化；Native的BF16 channel scale在adapter中
无损加宽为FP32，数值与往返均须逐bit一致，不把两种dtype的原始字节直接比较。

| Q切片（累计增加） | 最终Q不同元素数 | NOPE列不同元素数 |
| --- | ---: | ---: |
| 0：nalinaly原helper | 220781 | 181119 |
| 1：先合乘activation/channel scale，再乘INT32 accumulator | 220775 | 181112 |
| 2：再恢复投影BF16发布边界 | 31049，全部在RoPE列 | **0** |
| 3：再恢复Norm→RoPE前BF16发布边界 | **0** | **0** |
| 4：再按Native折叠512→256→128→64 | **0** | **0** |
| 5：再将尾recip(sqrt)改为高精度rsqrt | **0** | **0** |

反量化独立PTO根进一步确认：原两次乘法在Native BF16投影处有5个差异，
最大2.4414e-4；先合乘scale后为0。因此小的scale顺序差异确实存在，
本组Q主要误差仍来自越过两处BF16发布边界。MODE0逐bit复现生产原Q输出。
所有切片FULL/eager一致、输出毒化与guards通过。

任务 `task_20261027_224839_279287331096` 约50.1s完成。MODE5又在
T1/17/23/32/33/48/63全部与新计算的Native Norm/RoPE逐bit一致，FULL/eager和guards通过。
这些短T用保存的T24投影/scale等行重复构造，只验证Q反量化/Norm/RoPE，
不额外证明新M的Q GEMM或真实请求metadata。T24整除8，不走尾路径；
不能由它单独证明步骤4/5必要。完整块原本已有高精度rsqrt，没有把它记作新修正。

CPU的反量化根4Tensor、完整helper根7Tensor全部编译/ABI加载通过。
最初两次CPU编译发现分支变量及Tile有效区类型重绑定，改为独立SSA名字后通过，
未改工具链。第一次微任务 `task_20261027_223757_218194014320` 在dtype门禁退出，
约7.5s、未开始PTO对拍；修复后保持0bit门禁，独立审查确认没有放宽阈值。

从最终AllocateMemoryAddr dump统计静态Vec MemRef的最大addr+extent：
原helper106592 bytes，MODE5为77920 bytes，均低于192KiB。
这不是完整SPMD峰值或性能数据；硬件结果也未包含计时验收。
微任务后只更新了候选说明文字，去掉docstrings/comments/location后的AST完全相同，
凭证见 `evidence/q_micro_to_production_ast.json`。

生产采用MODE5；只替换Vector的Q消费helper，保留head分组、owner和QUERY发布前后的等待。
完整生产根55、debug57、观测69的外部ABI保持；17项生产源码SHA参与跨任务门禁。

### Q接回完整SPMD：前缀十个检查点全部逐bit一致

`spmd-production-v9`（`task_20261028_091433_390899927444`，192.9s）
通过完整生产debug入口FULL/eager八输出一致及guards检查。
`spmd-observe-v6`（`task_20261028_092001_416617828439`，190.7s）
通过相同输入/metadata/cache、17项生产SHA、观测与生产八输出、内部FULL/eager及guards门禁。

| 同输入Native模块边界 | 逐bit不同元素数 |
| --- | ---: |
| HC mixed/post/comb（三项） | 0 |
| InputNorm | 0 |
| QA BF16、QR INT8/FP32 scale（三项） | 0 |
| WKV BF16、KV Norm/RoPE（两项） | 0 |
| Q Norm/RoPE | **0** |

57/69参数完整根先通过无设备编译与ABI加载，凭证已归档。
观测v6到Q为止，没有取得主/Indexer Compressor、TopK、attention、O-proj或HC-post的
Native精度资格；也没有整模型逐token或性能结论。此前f22cb314基准保留，可独立revert本次Q提交。
以上两个任务均正常退出、释放设备；不能用模块通过推定完整CSA已对齐。

## 主Compressor：先修正观测，再定位state首差

沿用v9相同初始输入、metadata/cache、真实权重与归一化输入。Q十检查点通过后，
严格调用冻结Native `_forward_decode` 内的主Compressor算子和原参数：
Native state页视图、state页表、compressed完整B+1 query bounds/start_pos、
compact sin/cos、主wkv/wgate/APE/gamma，以及原ratio/coff/eps/mode。
每次恢复初态并同步，记录每次Native guards；没有修改生产算术或Native参考。

首项 `spmd-main-compressor-v1`（`task_20261028_092620_35450513147`，33.1s）
在Native自身全输出重复检查退出。物理输出为10×512，但本次只有4个有效压缩行，
其余6个槽为`[-1,31]`。Native使用empty分配物理输出，未发布的尾部不作为数值金值。
该失败的完整JSON保留，不计作PTO精度失败。

第二项 `spmd-main-compressor-v2`（`task_20261028_093128_6516408760`，34.2s）
保存两份Native物理输出、slots和逐行bit差异，确认差异仅在6个无效行；
完整state及4个有效输出仍要求逐bit自重复，均为0差异，guards均通过。
未放宽有效数据的比较阈值，也没有过滤Native scatter的负槽。

| 边界 | 不同元素数 | max_abs |
| --- | ---: | ---: |
| 前缀至Q的十个检查点 | 0 | 0 |
| 主Compressor完整state | **41005** | **9.5367431640625e-7** |

首差坐标从`[6,0,0,32]`开始。发现state首差后立即停止，未执行scatter，
因此没有主压缩cache或后续模块的精度结论。下一步只验证投影的N/K切分和累加顺序，
保持原state写回不变。

### 主投影单因素：因果确认列组K起点

`spmd-main-compressor-v3`（`task_20261028_094410_132217610232`，35.0s）
额外保存实际Native初始state view和物理布局。其NPU stride为
`[8192,2048,2048,1]`，storage offset=128 bytes；`.cpu()`后的逻辑紧凑副本不能
代替NPU物理stride。原始allocation有1212672 bytes，而逻辑view为606208 bytes。

第一次投影微任务在初态容量断言处退出0.9s，未执行候选；没有通过“除2”修补布局。
修正后从原allocation按实测stride/offset重建CPU view，与保存Native初态逐bit核验。
独立数学根随后使用紧凑逻辑state及双侧128-byte保护，不宣称它就是生产物理布局。

`mainproj-order-v2`（`task_20261028_094814_159105129159`，20.0s）：

| 累计切片 | Native完整逻辑state不同元素数 |
| --- | ---: |
| 0：原nalinaly M64/N64/K512、自然K顺序 | 41005 |
| 1：同一N64逻辑owner内部拆成两个N32 | 41005 |
| 2：再改K128，自然顺序 | 41005 |
| 3：再按Native列组循环偏移K起点 | **0** |

MODE0必须逐bit复现原完整生产state，已通过；所有切片的state/真实24行投影
FULL/eager和guards也通过。由本组因果对拍可确定，**列组K起点是这41005项
state差异的来源**；不能把仅缩小N/K切片记作有效精度修正。

列组规则参考Native arch32 Compressor：uniform且容量内每head16组，其余8组，
由device query bounds实时选择；每组起点偏移256，半head重置，K128顺序累加。
不从T/S推B，不要求每请求S6，不修改APE、state commit或池化数学。
当前硬件资格仅限这组T24；其它TND/tiling分档需要后续独立验证。

非joint生产路径接入时，Cube末尾追加既有`main_bounds`，保留原权重cache策略参数
索引及N64逻辑owner/信号发布位置；cooperative与普通两处同时替换。
生产根55、debug57、观测69外部参数不变，`JOINT_BF16_PREFIX`仍走旧路径，未获本修复资格。
接入完整SPMD的物理stride与前缀回归结果单独记录，不以微任务替代。

### 主投影完整接入：state/cache通过，停在Indexer state

完整57/69参数根无设备编译和ABI加载通过，原cache策略参数索引未变。
`spmd-production-v10`（`task_20261028_095854_2120717567`，203.97s）
通过完整生产FULL/eager八输出及guards检查；任务实际09:58:55至10:02:37，正常释放卡。

`spmd-observe-v7`（`task_20261028_100412_23586879960`，197.34s）
先通过18项源码SHA、输入/metadata/cache、观测/生产八输出、内部FULL/eager与guards门禁。
前缀至Q的十个Native检查点保持0差异。随后结果为：

| Native模块边界 | 逐bit不同元素数 |
| --- | ---: |
| 主Compressor完整state | **0** |
| 主压缩cache完整view | **0** |
| Indexer Compressor完整state | **9585，首差** |

主cache还补了独立控制：两份Native物理输出分别从相同初态原样scatter，
包括6个无效尾行，不过滤负槽；**整个backing allocation逐bit重复一致**，每次guards通过。
因此本组未发布尾部没有干扰cache金值。主pool数学虽然与Native不同，
本组主cache没有首差，暂不修改，不能由此宣称其它输入下内部pool数值逐bit一致。

Indexer调用严格用Native indexcom权重/APE/gamma、其state页表和compact metadata；
`cu_seqlens`按真实调用链来自主compressed metadata，未替换成形状相同的Indexer bounds。
Native完整Indexer state及有效压缩输出先通过自重复、每次guards通过，
再发现`9585`项差异，max_abs=`9.5367431640625e-7`，首坐标`[6,0,0,16]`。
已保存初态、实际NPU布局、权重和输出，首差后停止；未比较Indexer Hadamard/量化/cache、
TopK、attention、O-proj或HC-post。没有整模型token或性能验收结论。

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
初始 cache，以及当前十八个生产文件的 SHA 一致（包含 WKV/KV/Q/主投影helper与调用点），
再比较观测根与生产根的八个外部输出；通过后才解释各Native模块首差。
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

Q单因素复现：在 `f22cb314` 算子快照上配合当前测试脚本，先生产→观测得到Q首差
保存件；设置 `QOUTPUT_PROBE_REFERENCE=<该observation.pt>`，提交
`run_rms.sh <新label> qoutput`。先运行 `check_q_native_imports.py <receipt>`，
再用 `qdequant-base/combined` 和 `qoutput-0…5` 做CPU编译、ABI加载。
该快照尚未接Q修正，故MODE0可复现原生产Q。所有运行仍使用公共bootstrap、固定Native源码、
确定性开关及780/750+15s时限。Native2、128K和整模型扩展仍未运行。

### 主Compressor当前版本复核与历史单因素复现

当前checkout先用`compile_rms.py`的`spmd-debug`、`spmd-probe`选项分别编译57/69参数根。
使用上述`task-submit`环境，按以下顺序运行，每项使用新的label：

1. `run_rms.sh <production-label> spmd`，取得完整生产`production.pt`。
2. 设置`--env SPMD_PRODUCTION_REFERENCE=<该生产目录>/production.pt`，
   运行`run_rms.sh <observe-label> spmd`。全部源/输入/生产一致性门禁通过后，
   逐项比对Native至主state/cache；首差自动停止。

要重建本轮**修正前41005项首差**，使用`a4d97e63`的生产算子快照，
测试目录覆盖为当前版本；补齐诊断引用的新增helper文件，但不覆盖历史生产调用点或已有实现。
先按同样两项流程生成生产和观测保存件。当前版本已经接入主投影修正，不能拿修正后的
state当作MODE0要求复现的历史生产输出。

四个微根先分别用`compile_rms.py <新输出目录> mainproj-0`至`mainproj-3`做无设备编译。
再提交一项`run_rms.sh <新label> mainproj`，额外传入：

- `--env MAINPROJ_PROBE_REFERENCE=<上述修正前观测>/observation.pt`
- `--env SPMD_PRODUCTION_REFERENCE=<其配对生产>/production.pt`

观测保存件须包含真实initial view及stride/offset，微任务先从原raw重建并逐bit核验，
MODE0再重现历史完整生产state。四切片均做FULL/eager、投影有效区及guards检查。
测试固定T24，不包含新batch/变长分档、完整Native FULL或整模型token验收。

### Indexer Compressor投影/state的单因素定位（2026-10-28）

主Compressor state/cache通过后，`spmd-observe-v7`的下一首差为Indexer state：
9585个FP32元素，最大绝对差`9.5367431640625e-7`。Native state和有效输出重复一致，
不以未发布的`[-1,31]`输出尾行作为逐bit金值。继续复用同次T24的Norm、真实权重、
初始state和完整Native metadata，不重造输入。

| 单因素实现 | 对Native完整逻辑state不同元素数 |
| --- | ---: |
| MODE0：原M16/N32、K512自然顺序 | 9585 |
| MODE1：内部拆N16，K512自然顺序 | 9585 |
| MODE2：N16、K128自然顺序 | 9585 |
| MODE3：N16、K128、Native按列组偏移K起点 | **0** |

任务`task_20261028_101348_282810818448`，实际占卡10:31:21–10:32:03（42秒），
脚本31.77秒，四臂FULL/eager、投影有效区与guards全部通过。
MODE0先逐bit重现原完整生产state，初始state先按实际NPU stride/offset从原始allocation
重建核验，再在独立compact逻辑view做微测试。因此该表证明当前输入的投影顺序原因，
**尚不能替代完整生产物理布局与Indexer cache验收**。

列组来自设备端真实请求边界：当前均匀T24按每128列的8组、每组16列，
K起点`(column % 128 // group) * 256`，随后按K128循环遍历4096维。
ragged路径保留Native的group64选择，但本轮未验收新形状。
生产接入保留原M16/N32逻辑owner、state提交、PHASE_IDX_PROJECT发布和等待位置；
复用Native实际传入Compressor的`main_bounds`，不增加根入口参数。

复现微测试需先保留`e185a126`生产算子快照并使用当前测试脚本：
先生成生产→观测v7等价保存件，再设置`INDEXPROJ_PROBE_REFERENCE=<观测>/observation.pt`
与`SPMD_PRODUCTION_REFERENCE=<配对生产>/production.pt`。
先用`compile_rms.py`的`indexproj-0`至`indexproj-3`逐根做无设备编译/ABI加载，
再提交`run_rms.sh <新label> indexproj`。不要用已修正的state作为MODE0历史起点。
证据见[evidence/indexproj-order-v1.json](evidence/indexproj-order-v1.json)。

### Indexer投影接回完整SPMD（2026-10-28）

57/69参数根均先无设备编译和ABI加载通过。生产快照v11
（`task_20261028_104318_120810726`，196.03秒）FULL/eager八个外部输出全部逐bit一致，
cache、metadata保护通过。观测v8（`task_20261028_104735_31014230576`，202.13秒）
先通过同输入、19项源码SHA、生产等价、观测FULL/eager及guards门禁，再得到：

| Native同输入模块边界 | 逐bit不同元素数 |
| --- | ---: |
| 前12项，含HC、Q/KV、主state与cache | 0 |
| Indexer完整物理state | **0** |
| Indexer INT8 key cache | **32，首差** |
| Indexer FP16 scale及其后 | 首差后停止，未验收 |

Native Indexer compressor的state/已发布输出重复一致；再运行原生
`rotate_activation→indexer_quant_scatter_part1→dsa_indexer_scatter_scale_part3`，
两次都恢复初始cache，完整共享allocation逐bit一致，逐次guards通过。
负槽和未发布输出尾部原样传给实际Native函数，未人为过滤。
当前32个key元素均最大相差1，仅证明剩余差异在state之后，尚未归因于具体后缀模块。
下一步导出PTO压缩后Norm/RoPE、Hadamard及量化输入，与实际Native中间结果对拍。
两个NPU任务均完成并释放设备；本轮未做整模型token或性能验收。
