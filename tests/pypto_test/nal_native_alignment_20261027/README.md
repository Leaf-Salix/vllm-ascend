# 基于 nalinaly TND SPMD 的 Native 数值对齐

日期：2026-10-27；Indexer及attention完整接入复核更新于2026-10-28。

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
| 新增选卡规则 | 后续任务仅限物理8–15；本单卡示例显式`--device 8`；启动器也校验卡号 |
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

上述选卡限制自2026-10-28用户新增规则起生效；历史任务证据保留当时的实际卡号。
新任务不使用无设备范围限制的`--device auto`。需要16卡时必须另获明确批准。

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
task-submit --device 8 --max-time 780 \
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

### Indexer key首差继续拆解：Hadamard缩放边界

扩展观测根，仅将5个内部scratch改为外部可见：池化结果、Norm/RoPE BF16结果、
Hadamard FP32 accumulator、FP32 scale、量化前FP32值。生产`_csa_core_group`调用AST
完全相同；debug生产根仍57参数，观测根扩为74参数。
新增scratch只比较真实关闭token`[3,9,15,21]`及有效compact`[0,1,2,3]`，
外部8边界、完整cache与guards仍全量验证。池化/scale scratch当前仅验证图稳定性，
不能列为已经直接对拍Native。

首次无设备编译因`idx_normed`动态维度返回别名重绑定退出，未占卡；
5个LHS改为独立`_after`别名后编译与ABI加载通过。没有修改SDK。
观测v9（`task_20261028_110119_12563384556`，202.27秒）通过生产等价、
FULL/eager、guards及前13项Native门禁后：

| 同输入边界 | 对Native逐bit不同元素数 |
| --- | ---: |
| Indexer压缩Norm/RoPE的BF16输出 | **0** |
| Hadamard FP32 accumulator落BF16，对Native BF16 linear | **0** |
| 乘`128**(-0.5)`后，对Native BF16输出无损加宽 | **512/512，首差** |

缩放最大绝对差`0.006908893585205078`。
Native实际`hadamard_linear→hadamard_scale`与`rotate_activation`在有效行逐bit一致，
先证明拆分观测没有改变Native金值。
再只读同一保存件，在CPU执行`actual.bfloat16().float()`后512个差异全部清零。
这项CPU证据仅解释缩放首差，不代替NPU cache验收。
证据为[evidence/spmd-observe-v9/result.json](evidence/spmd-observe-v9/result.json)和
[evidence/indexer-scale-cpu-round-v1/result.json](evidence/indexer-scale-cpu-round-v1/result.json)。

候选只在Hadamard缩放后恢复Native的BF16舍入，再无损加宽给原量化流程；
已有matmul BF16边界、amax、INT8转换链、key/scale提交、owner和等待全部不变。
独立AST审查确认，移除这两层新增cast后整个helper与修正前逐AST相同。
该helper新增到跨生产/观测源码身份列表，合计20项。

复现缩放首差：在`8957f04a`生产算子上配合当前74参数观测脚本，先生成生产快照，
再以该快照运行观测，取得v9等价的`observation.pt`。新增5scratch只改变可见性，
必须通过完整生产8边界等价门禁。CPU可用下列语句验证同一输入的舍入因果：

```python
import torch

saved = torch.load("<v9-equivalent>/observation.pt", map_location="cpu", weights_only=True)
actual = saved["indexer_hadamard_scaled_actual"]
native = saved["indexer_hadamard_scaled_native"]
assert int((actual.view(torch.int32) != native.view(torch.int32)).sum()) == 512
assert torch.equal(actual.bfloat16().float().view(torch.int32), native.view(torch.int32))
```

修正后的当前版本需要重新生成生产快照，不能继续引用修正前19项SHA的v11。
先分别以`compile_rms.py`的`spmd-debug`和`spmd-probe`编译57/74参数根，
再按前述780/750+15s时限提交生产→观测两项任务。观测通过20项源码身份和输入门禁后，
依次验收Norm/RoPE、Had线性BF16、缩放输出、完整key cache及scale cache。

### 缩放舍入接入后的完整设备验收

57/74参数根无设备编译、ABI加载通过，生产v12
（`task_20261028_111445_188908717115`，202.86秒）及观测v10
（`task_20261028_113007_28471475857`，198.78秒）均已完成并释放设备。
观测先通过20项源码身份、同输入、生产8边界等价、FULL/eager与guards，再验证：

| 当前8K/B4/T24的边界 | 修正前不同元素数 | 修正后不同元素数 |
| --- | ---: | ---: |
| Indexer缩放输出 | 512 | **0** |
| Indexer完整INT8 key cache | 32 | **0** |
| Indexer完整FP16 scale cache | 首差后未验收 | **0** |
| 其余已验收前缀 | 0 | **0** |

Native完整共享cache allocation重复一致，逐次guards通过。至此18个Native模块检查点
全部逐bit一致，当前任务`complete/pass=true`的范围仅到Indexer cache。
**尚未比较后续Indexer query、权重系数、TopK、sparse attention、O-proj、HC-post，
不能宣称整个新CSA或整模型token已验收，也没有本分支的新性能结论。**
证据见[evidence/spmd-observe-v10/result.json](evidence/spmd-observe-v10/result.json)。

### 实际Native QLI消费者的首差（2026-10-28）

`42dc9ca7`的生产算术不变，观测根新增query INT8和FP32 scale两个scratch，
76参数CPU编译及ABI加载通过。观测v11
（`task_20261028_114355_35221432436`，207.24秒）实际执行Native完整attention半层：

- 最小hook只取QLI返回TopK；详细hook在实际消费者前clone全部kwargs。
- 两遍都恢复相同cache，QLI参数和`return_value=False`原样传递，finally恢复op。
- Native控制/详细hook的8个外部输出逐bit一致，guards通过。
- 不调用PTO service，wrapper runtime明确要求None。
- 原18项模块边界通过后，实际Native full-forward的6类cache与PTO replay也全部逐bit一致。

随后首差为实际QLI的query INT8：**23155个元素不同**。Native消费者FP16 scale
尚未比较，权重、TopK及整层输出不继续验收。累计24个前缀/cache检查点已通过；
该任务不能作为新分支完整CSA或整模型token通过证据。
下一步只对该query链hook实际投影落BF16、原地RoPE前后、Hadamard线性BF16及缩放BF16，
与PTO accumulator/RoPE/Had逐级比较，先定位最早舍入差异。

### Indexer query 内部首差：反量化及 RoPE（2026-10-28）

79参数观测v12（`task_20261028_115939_26307930749`，201.76秒）保持生产算术不变，
先通过前24个检查点和Native hook控制，再取得实际Native投影BF16、原地RoPE前后、
Hadamard线性BF16及缩放BF16。首差前移至query RoPE：31209个BF16元素不同，
其中NoPE区1个、旋转区31208个；后续边界不继续验收。

同一保存的INT32 accumulator、逐行scale及无损加宽的Native通道scale在CPU复核：

| 投影落BF16的表达式 | 与实际Native投影不同元素数 |
| --- | ---: |
| `(acc.float() * row_scale) * channel_scale` | 1 |
| `acc.float() * (row_scale * channel_scale)` | 0 |

此外，实际Native的量化投影在RoPE前已输出BF16，原PTO helper在FP32上直接做RoPE，
直到RoPE后才舍入。这一阶段建立独立候选，保持原M8、四head、pipeline、尾行有效域、
RoPE公式及外围发布/等待，只隔离scale结合顺序和RoPE前BF16边界。
CPU保存件分析不替代NPU候选验收；必须依次满足原臂复现生产、实际Native RoPE重跑一致、
FULL/eager一致、guards通过以及修正臂与Native逐bit一致后，才接入生产调用点。
证据见[evidence/spmd-observe-v12/result.json](evidence/spmd-observe-v12/result.json)及
[evidence/query-dequant-cpu-order-v1/result.json](evidence/query-dequant-cpu-order-v1/result.json)。

### query 单因素设备验证与接入

`idxquery-boundaries-v1`（`task_20261028_122311_134649612792`，35.23秒）通过。
使用v12保存的实际Native query投影及同一INT32 accumulator，先重跑实际Native
`inplace_partial_rotary_mul`确认金值0差异，再运行以下FULL graph单因素：

| 同一query算术切片 | 投影BF16不同元素 | RoPE后BF16不同元素 |
| --- | ---: | ---: |
| 原连续乘scale | 1 | 31209 |
| 只先合并scale | 0 | 31208 |
| 合并scale，且RoPE前先落BF16 | 0 | **0** |

原RoPE臂与生产保存件逐bit一致，排除独立入口改变误差的假象。全部臂FULL/eager及
输出guards通过。T1/17/23/32/33/48/63的保存有效行重复域也全部0差异；
这些额外形状只验证逐元素处理和尾块，不代表新GEMM形状或整层真实TND资格。
证据见[evidence/idxquery-boundaries-v1/result.json](evidence/idxquery-boundaries-v1/result.json)。

复现：先以当前79参数观测脚本在修正前生产算术（`42dc9ca7`）取得v12等价保存件；
随后source既定env，使用公共bootstrap分别运行`compile_rms.py`的
`idxquery-dequant-0/1`、`idxquery-rope-0/1/2`，输出必须为新目录，ABI分别4/6且不初始化NPU。
再提交原限时单卡命令，增加
`--env IDXQUERY_PROBE_REFERENCE=<v12-equivalent>/observation.pt`，
`run_rms.sh <新标签> idxquery`。脚本会复用实际Native BF16输入和原地RoPE，不使用手写数学金值。

接入仅改`vector_schedule`一个调用点，固定新helper的MODE2；原先query投影等待、
RoPE发布、M8/四head、pipeline、所有尾有效域和55/57外部ABI不变。
新增helper加入生产/观测SHA清单，现为21项。无设备编译和完整生产/Native复核另行记录，
单因素结果不代替后续Hadamard、量化或完整CSA精度。

接入后的57参数生产根、79参数观测根均已无设备编译并加载ABI，NPU未初始化；
凭据位于`evidence/cpu-query-spmd-debug-v13`及`evidence/cpu-query-spmd-probe-v13`。
在`31d92bcc`提交时，完整生产复测`task_20261028_122941_161574916148`仍等待设备；
因此本次提交只确认单因素数值修正和完整入口可编译，**尚不把接入后完整流水记为通过**。
结果回来后会追加生产/观测证据，不覆盖历史记录。

### query RoPE 完整接入复核

生产v13（`task_20261028_122941_161574916148`，204.48秒）及观测v13
（`task_20261028_124948_283953124444`，204.58秒）均已完成、释放设备。
前者8个外部边界FULL/eager逐bit一致、37项guards通过；后者先通过21项源码身份、
同输入、生产观测等价及Native hook控制，得到：

| 同一8K/B4/T24检查点 | query修正前不同元素 | 修正后不同元素 |
| --- | ---: | ---: |
| Indexer query RoPE BF16 | 31209 | **0** |
| Indexer query Hadamard线性落BF16 | 首差后未验收 | **0** |
| QLI实际INT8 query | 23155（更早v11） | **18496** |
| 其他24个已验收前缀／cache检查点 | 0 | **0** |

累计26个检查点通过，当前首差仍为query量化，因此没有继续验收scale、权重、TopK或
后续attention/输出，也没有开展新分支整模型token或性能测试。
当前T24策略的`FUSED_INDEXER_COEFF=false`，不能把该结果推广到融合系数路径。
下一步只对同一已保存的Hadamard accumulator和实际Native量化输入做单因素；
不混入新的GEMM形状、长序列或请求配置。
证据见[evidence/spmd-production-v13/result.json](evidence/spmd-production-v13/result.json)及
[evidence/spmd-observe-v13/result.json](evidence/spmd-observe-v13/result.json)。

### query Hadamard 与动态量化单因素

`idxquant-boundaries-v1`（`task_20261028_125635_315822823876`，18.29秒）通过。
三个ABI3微根先无设备编译。固定v13的同一FP32 Hadamard accumulator，
其落BF16后已与实际Native线性输出逐bit一致。先对保存的Native scaled BF16以原始
`[24,64,128]`形状直接重跑`torch_npu.npu_dynamic_quant`：INT8 query、消费者FP16 scale
均逐bit复现QLI hook，输入未改变，同时保存Native返回的原始FP32 scale。

| 同一Hadamard accumulator的切片 | INT8 query不同元素 | 消费者FP16 scale不同元素 |
| --- | ---: | ---: |
| 原路径 | 18496 | 1377 |
| 只补线性输出BF16 | 12977 | 1351 |
| 再补缩放输出BF16 | **0** | **0** |

原臂的INT8及完整FP32 scale逐bit复现生产，全部臂FULL/eager、poison重写与两种输出
owner guards通过。修正臂原始FP32 scale还有540个不同元素，最大绝对差为
`2.3283064365386963e-10`；Native在QLI前转FP16，当前PTO系数消费者也先转FP16，
这一消费边界差异为0。这里明确没有宣称原始FP32中间值全部逐bit相同。
CPU使用相同输入及表面相同表达式曾得到147个临界INT8差异，但NPU实际DSL修正臂为0；
不能用CPU浮点结果替代设备舍入验收，也没有据此更换round或提高精度参数。

复现时增加`--env IDXQUANT_PROBE_REFERENCE=<v13-equivalent>/observation.pt`，
使用`run_rms.sh <新标签> idxquant`；三根无设备入口为`compile_rms.py`的
`idxquant-0/1/2`。Native重放控制先于全部PTO臂，MODE0必须复现生产，
MODE2在query或消费者FP16 scale出现首差时保存`quantization.pt`后停止。
输入FP32 accumulator、Native BF16 scaled、Native INT8及原始FP32 scale会保存在
结果目录的该文件中。该诊断只覆盖当前普通量化路径及T24，不覆盖fused或新形状。
证据见[evidence/idxquant-boundaries-v1/result.json](evidence/idxquant-boundaries-v1/result.json)。

生产接入在已有Native query helper文件中增加普通量化实现，仅恢复上述两次cast，
amax、scale、reciprocal、rint、输出dtype及外围phase等待/发布不变；
旧helper保留用于历史复现。`FUSED_INDEXER_COEFF=true`路径未改动、尚未验收。
80参数观测根再把`head_weights`的内部分配换为外部scratch，比较有效24行；
在query INT8及FP16 scale门禁后依次检查Native消费的FP16 weights、TopK和最终输出，
仍然首差停止。最终输出仅统一逻辑reshape后对比，不把存储维度不同记作数值误差。

接入后57参数生产根和80参数观测根均无设备编译、ABI加载通过；
生产v14（`task_20261028_130611_356197120433`，205.61秒）完成，8边界FULL/eager为0，
37项guards通过。在`a9220849`提交时，Native观测v14（`task_20261028_131107_37720968718`）仍等待设备，
尚不把普通query量化的单因素结果推广为完整流水通过。

详细Native控制臂新增只读权重hook：实际`weights_proj`的输入/输出，以及实际
`prepare_dsa_indexer_weights`的乘系数后输入与FP16返回值；forward hook不替换结果，
静态方法描述符由上下文恢复，finally移除hook，统一同步后才复制CPU快照。
最小控制臂与详细臂的8个外部边界及guards仍必须通过。
该观测仅保存真实Native中间值，不通过除系数反推投影，也尚未单独验收PTO原始权重投影。
证据见[evidence/spmd-production-v14/result.json](evidence/spmd-production-v14/result.json)。

### 普通query量化完整接入复核及下一首差

观测v14（`task_20261028_131107_37720968718`，208.48秒）已完成并释放设备。
21项源码、同输入、生产8边界等价、FULL/eager、guards及Native详细hook控制全部通过。

| 同一8K/B4/T24边界 | 修正前不同元素 | 完整接入后不同元素 |
| --- | ---: | ---: |
| query RoPE BF16 | 31209 | **0** |
| query Hadamard线性BF16 | 首差前未知 | **0** |
| QLI query INT8 | 18496 | **0** |
| QLI query scale FP16 | 1377（独立原臂） | **0** |
| 原有24个前缀／完整cache检查点 | 0 | **0** |
| 权重投影输入 | 新增观测 | **0** |
| actual prepare consumer与QLI kwargs | 新增控制 | **0** |
| QLI weights FP16 | 首差后未验收 | **1360** |

权重FP16首差的最大绝对差为`0.00018310546875`。当前未继续验收TopK和最终输出，
更没有宣称该新分支整模型token通过。Native权重projection/scaled/consumer真实中间值
已保存在v14保存件；下一步仅对同输入、正式权重的原四split投影做正向导出，
先验证sum落BF16，再测试两次BF16边界。投影先差则停止，不反除系数、不同时改K次序。
证据见[evidence/spmd-observe-v14/result.json](evidence/spmd-observe-v14/result.json)。

### 权重投影／舍入单因素

`weightproj-boundaries-v1`（`task_20261028_133215_5755179003`，32.01秒）通过。
两个ABI5根先无设备编译。正式权重
`layers.4.attn.indexer.weights_proj.weight`的原始BF16 SHA256为
`b46991cf5a8a032ec111ede4af54fe7df5e54fa32d167bedc315fca869d44a5e`。
同一真实hook输入配合该权重，NZ29 `F.linear`重放先逐bit复现实际Native投影、
乘模型系数后的BF16、最终FP16三组金值，才开始PTO对拍。

| 保持原四份K1024、原FP32左折叠 | 投影落BF16差异 | 最终FP16权重差异 |
| --- | ---: | ---: |
| 原路径；同时必须复现生产FP32 head_weights | **0** | 1360 |
| 补投影及乘系数输出的两次BF16 | **0** | **0** |

修正臂缩放BF16也与真实Native prepare输入逐bit一致；FULL/eager、回放前poison以及
partials/raw/weights三个owner的guards均通过。原始投影是正向导出的sum，
没有用除系数反推。投影一旦首差，本脚本会在原臂立即停止，不能用后续cast掩盖它。
本次投影已经通过，因此**没有引入旧精度分支的Native K次序改写**。

复现使用`compile_rms.py`的`weightproj-0/1`及
`--env WEIGHTPROJ_PROBE_REFERENCE=<v14-equivalent>/observation.pt`，
`run_rms.sh <新标签> weightproj`；固定实际T24、同一正式层权重及观测输入。
结果目录`projection.pt`保存原始partial、正向投影sum、scaled与Native金值。
证据见[evidence/weightproj-boundaries-v1/result.json](evidence/weightproj-boundaries-v1/result.json)。

生产接入仅以新Native reduce替换一处调用，保留原四split投影、左折叠、phase owner、
等待与发布，以及55/57外部接口、80参数观测入口。该权重producer位于普通/fused分支之前，
两类后继会读到新权重；当前只验收普通query路径，不能宣称fused整体精度已通过。
接入后的完整生产/Native结果另行追加，单因素结果不代替完整流水验收。

### 权重完整接入结果：TopK已对齐，继续定位输出

生产v15（`task_20261028_134150_95071215393`，201.56秒）完成，8个外部边界
FULL/eager逐bit一致，37项guards通过。Native观测v15
（`task_20261028_134604_112067922027`，212.17秒）完成并释放设备：
21项源码、同输入、生产等价、FULL/eager及最小/详细Native hook控制均通过。

| 8K/B4/T24实际Native消费边界 | 不同元素 |
| --- | ---: |
| 前缀、六组完整cache、query及其消费者FP16 scale | **0** |
| QLI weights FP16 | **0**（此前1360） |
| QLI TopK | **0** |
| 完整HC-post输出 | **44712** |

最终输出最大绝对差`0.015625`，因此本轮精度仍未通过。当前证据已排除这组输入下的
TopK选点差异，下一步只读抓取实际Native O-proj入口heads及输出，
与PTO内部packed heads对拍，先区分attention与输出投影，不扩大配置或宣称整模型token一致。
以上是单层正式权重、合成hidden/history的定位结果，不能作为新分支整模型交付验收。
证据见[evidence/spmd-observe-v15/result.json](evidence/spmd-observe-v15/result.json)
及[evidence/spmd-production-v15/result.json](evidence/spmd-production-v15/result.json)。

### 输出后缀的最小观测

生产源码保持v15；观测根仅把已有BF16 packed heads内部分配换成外部缓冲，
ABI由80增至81，核心算术/调度调用相同。CPU编译及ABI加载通过。
按8组各128物理行恢复有效24行，再还原`[24,64,512]`，不比较未发布padding。
独立CPU索引验证、双wrapper正常/异常恢复控制及7项契约测试通过。

Native详细臂只读包裹实际`_forward_decode`与`_forward_o_proj`：
保存实际attention返回BF16、逆RoPE后的O-proj入口BF16、实际projection输出BF16，
调用原绑定方法并原样返回原对象。context退出恢复实例属性，统一同步后复制CPU。
最小/详细Native的8个外部输出与guards仍是前置门禁。
先验收post-inverse heads，首差停止；另外两组快照只用于后续定位，不计为已通过。
此观测不增加模型算子替换、SDK改动或Native数学重构。

观测v16（`task_20261028_135701_162382027691`，210.59秒）已完成并释放设备。
生产等价、FULL/eager、Native详细hook控制及此前32个精度门禁全部通过；
首差在post-inverse heads，339127个元素不同。CPU只读分析进一步区分：

| 同一次真实forward快照 | 不同元素 | 最大绝对差 |
| --- | ---: | ---: |
| Native attention返回与O-proj入口的NoPE | **0** | 0 |
| PTO NoPE与Native attention返回的NoPE | **291048** | 0.00048828125 |
| PTO RoPE与Native O-proj入口的RoPE | 48079 | 0.0009765625 |

NoPE没有逆RoPE运算，因此**attention在逆RoPE前已经不同**；不能仅修逆旋转BF16就
宣称解决attention。下一步先抓取实际Sparse算子的完整消费输入，以同Q/cache/TopK/metadata
做独立Native重放与旧Native512精度算法复核，再逐因素定位。
CPU复现用同env解释器运行`analyze_heads.py <v16结果目录> <不存在的输出json>`，
它只读取快照并按BF16位比较，不初始化设备、不反向推断旋转前值。
证据：[v16原始结果](evidence/spmd-observe-v16/result.json)、
[heads分域分析](evidence/spmd-observe-v16/heads_analysis.json)。

### 同输入Native Sparse及旧K512参考复核

v17（`task_20261028_141205_245202121516`，38.96秒，已有编译缓存命中）
只新增真实Sparse消费者只读捕获，原生产仍为v15。完整前缀、TopK及观测等价门禁
继续通过，heads首差保持339127；这次耗时不是性能测量。
真实Sparse返回与`_forward_decode`返回逐bit一致，query、cache、TopK、页表、
query bounds、seq lengths、sink、metadata及标量选项均保存。

`attention-reference-v1`（`task_20261028_142323_292186016811`，25.38秒）通过：

| 同一8K/B4/T24实际消费输入 | 不同元素 |
| --- | ---: |
| 实际Sparse Q与此前Native Q前缀 | **0** |
| 实际Sparse TopK与QLI输出 | **0** |
| bounds/seq lengths/两页表与生产fixture | **0** |
| 原Native Sparse全参数重放 | **0** |
| 原Native inverse RoPE重放 | **0** |
| 旧K512参考的pre-inverse BF16 heads | **0** |
| 旧K512参考的post-inverse BF16 heads | **0** |

FULL/eager、回放前NaN覆盖、输入输出guards和输入逐bit只读检查全部通过。
positions另按实际Native bounds/seq lengths逐token重建并核对，没有用token//6。
参考数学来自旧精度提交`7472fb9a66e23d1b0998a20e7b89cfbd472395a3`，
原`decode_sparse_attn_csa.py` SHA256为
`dafd595cacf0bad5978776b07594ea9f8ae25f6fb22ea15d2545180ced47c434`。
复制仅调整相同config/layout的导入、固定诊断T24输出容量、root/Out标注及直接导出
`n_bf16`；数学没有借用Native模型算子。最初无设备编译因原Tensor输出标注被降低为Out
触发ABI门禁，改成显式Out后14参数编译/加载通过，没有修改SDK或放宽检查。

独立复现为`compile_rms.py <新目录> attention-reference`，再提交
`run_rms.sh <新标签> attention`，传入
`ATTENTION_PROBE_REFERENCE=<v17-equivalent>/observation.pt`和
`ATTENTION_PROBE_FIXTURE=<v15-equivalent>/production.pt`。
启动复用本文env/bootstrap，外780秒、内750+15秒；结果保存两个输入件SHA，
只对同形同权重快照有效。本次没有接回生产SPMD、没有整模型token或新性能结论。
证据：[v17](evidence/spmd-observe-v17/result.json)、
[独立K512](evidence/attention-reference-v1/result.json)、
[无设备ABI](evidence/cpu-attention-reference-v2/receipt.json)。

### Native K512数学接回单SPMD

独立参考通过后，将相同数学拆成同一MIX内的Cube与双Vector角色，保留
K512概率与PV累加边界、实际K求和树、sink初始权重、概率CAST_ROUND、
BF16发布后再inverse RoPE。没有调用Native attention完成候选计算，也没有
改SDK、runtime或生成C++。本小节仍只验证8K/B4/T24的实际同输入。

| 微测试 | 结果 | 处理 |
| --- | --- | --- |
| v1 | 首次候选执行停滞，约60秒释放设备 | micro漏声明双AIV，编排第三kernel ID为空；补`split_aiv=True` |
| v2/v3 | FULL/eager有925/957个不同元素 | 保留首差快照，未继续下游或宣称精度通过 |
| v3 CPU定位 | 与Native各830差，全部在head0/32首32个BF16 | 实际生成的死`next_m`读取与PV-left同UB起点，重叠128B |
| **v4** | **pre/post inverse均逐bit对齐Native** | **FULL/eager及额外3次NaN覆盖回放全部0差异** |

v4任务`task_20261028_155621_331511711767`测试主体28.09秒，占卡约39秒并已退出。
v4相对于发生漂移的候选，仅删除不参与计算的local max循环传递及其读取：真正的
`sm_old_m`仍从原GM状态读取，sink初始化、max写回、softmax、PV与舍入均保留。
这证明该删除解决了本微测试的运行漂移；没有据此认定SDK内部具体哪条指令有bug。
编译资格额外硬检查真实生成的`MixedKernels`，要求两条AIV均被调度，避免只审
Cube/Vector二进制而漏掉实际launch合同。

完整接入候选另补原plan已有的table列/cache页数保护，并使用独立393216字节
copy64工作区，不复用O-proj缓冲。score/prob从根开始分配为`[1536,512]`；
保留原raw TopK、Native请求边界、cache依赖和ATTNVEC后才能执行O_A的规则。
正常55参数、debug 57参数与81参数观察根均已通过无设备编译和双AIV检查；资源与同步
独立审查通过。正式根仍是一个PTO顶层入口和一个完整MIX SPMD。

微测试复现沿用上一小节命令，增加`ATTENTION_PROBE_VARIANT=native-spmd`，
CPU selector为`attention-native-spmd`。生产完整接回需要重新生成快照及观察，
不能复用旧21项source SHA或把微测试收据当作整层验收。
当前没有新增整模型token一致性、混合短query或性能结论。

证据：[v1执行故障](evidence/attention-native-spmd-v1/result.json)、
[v2图间首差](evidence/attention-native-spmd-v2/result.json)、
[v3分域定位](evidence/attention-native-spmd-v3/drift_analysis.json)、
[v4逐位通过](evidence/attention-native-spmd-v4/result.json)、
[完整根CPU资格](evidence/cpu-full-attn-native-v21/receipt.json)、
[观察根CPU资格](evidence/cpu-observe-attn-native-v21/receipt.json)、
[正常服务根CPU资格](evidence/cpu-normal-attn-native-v21/receipt.json)。

### 完整CSA接回结果：首差推进到O-proj之后

生产快照v18（`task_20261028_161715_3025584033`，204.51秒）与观察v18
（`task_20261028_162321_7374823997`，211.95秒）均已退出。
仍为真实layer4权重、合成hidden/history、8K/B4/T24，未测整模型token。

| 完整接入门禁 | 结果 |
| --- | --- |
| 生产FULL/eager八个外部状态 | 全部逐bit一致 |
| 观察根与原生产根八个外部状态 | 全部逐bit一致 |
| 跨任务同输入与23个source SHA | 通过 |
| Native详细/最小hook控制八个外部状态 | 全部逐bit一致 |
| HC-pre至TopK的原32项Native门禁 | 全部0差异 |
| **attention post-inverse heads** | **0差异**，此前339127差已消除 |
| 最终HC-post输出 | **41859差**，最大绝对差0.015625，首差在此停止 |

这证明Native heads已在完整SPMD里对齐，剩余首差位于O-proj/HC-post链路。
尚未将其唯一归因为某一量化或矩阵步骤，下一步以实际Native `wo_b`的BF16输入
（本轮只读保存为`native_output_stages.wo_a`）继续拆分WO-A、量化和WO-B。
不能把当前版本标为完整CSA精度通过，也没有新增性能结论。
生产快照JSON的`pass=false`表示没有执行Native精度判定；其`phase=production`、
`pass_kind`及八个自一致门禁用于区分该用途，不能按退出0直接判定精度。

复现先提交`run_rms.sh <新生产标签> spmd`生成`production.pt`；再提交
`run_rms.sh <新观察标签> spmd`，显式传
`SPMD_PRODUCTION_REFERENCE=<新生产目录>/production.pt`。两个任务共享同env、
bootstrap和源码，独立控制780/750+15秒，不并行覆盖源码或输入件。
观察必须按顺序通过等价/guards、原前缀门禁、heads，才能比较最终输出。

证据：[完整生产快照](evidence/spmd-production-v18/result.json)、
[同输入Native首差](evidence/spmd-observe-v18/result.json)。

### WO-A首差与同输入累加顺序诊断

本节仍仅覆盖8K/B4/T24、真实layer4权重和合成hidden/history，不代表整模型token。
新增一个FP32 `o_r`观测缓冲后，82参数观察根通过无设备编译及双AIV检查。
生产57参数与正式55参数ABI未变。coreGroup调用AST保持相同，返回使用独立
`o_r_after`变量，避免观察缓冲的符号形状重绑定。

观察v19（`task_20261028_165423_230040516941`）显式使用物理8，执行235.31秒。
8项生产等价、8项Native详细/最小hook控制、39项跨任务输入相同、全部guard通过。
原33项Native门禁一直到post-inverse heads均为0；首差是WO-A的BF16发布边界：
24×8192个元素中39个不同，最大绝对差`6.103515625e-5`，在此停止。

微测试直接使用同次Native heads和真实`wo_b` BF16入口，先验证checkpoint重建的
Native `npu_transpose_batchmatmul`逐bit复现该入口，再要求原PTO helper逐bit复现
观察到的FP32 `o_r`。两项控制均为0。只读权重、packed输入和输出保护区均通过。

| 微测试实现 | 与Native的WO-A BF16差异 | 三次NaN覆盖后的FULL/eager |
| --- | --- | --- |
| 原M128/N128/K256 | 39 | 全部0 |
| M16/N128，K256自然累加 | 39 | 全部0 |
| 同M16/N128，K256半程旋转 | **0** | **全部0** |

后两臂仅改变K起点，仍完整累加16个K256块。令`g`为输出组、`n`为N128块：
`first_panel=8*((g%2+int(3<=n<6))%2)`，顺序为0…15或8…15、0…7。
这个起点来自旧精度分支的T24假设，经本次同输入微测试确认；没有将旧的全T标定表
直接复制为新工具链契约。两段非负`pl.range`保留原旋转顺序，避免NZ环绕索引
证明失败，不修改SDK或生成C++。结果证明单改M布局不能消除这39差，而起点能在
本组输入消除；没有测性能，也未证明其它T或完整SPMD接回后的结果。

微测试v1（`task_20261028_171117_321584821667`）因诊断脚本误读不存在的WO-A
哈希字段退出，未执行矩阵对拍。修正后先做无设备参考件schema检查，再提交v2
（`task_20261028_171433_338704628719`，物理8，28.20秒），上述三臂全部完成。
原23项前缀哈希清单不含WO-A；微测试另核验已推送`a57c3cb4`的WO-A文件：
`bcdc7ec91fcf70c7709a859395396c5250c48b2ab9b8890771fe73dbeca78c29`。
文件身份与原FP32数值复现是两个独立前置条件。

复现步骤：沿用同一env/bootstrap，CPU selector为`spmd-probe`及`woa-0/1/2`；
先复用production-v18提交`run_rms.sh <标签> spmd`，取得v19等价的观察件；随后
显式`--device 8 --max-time 780`提交`run_rms.sh <标签> woa`，另传：

```bash
--env WOA_PROBE_REFERENCE=<观察目录>/observation.pt
--env WOA_PROBE_SOURCE_SHA256=bcdc7ec91fcf70c7709a859395396c5250c48b2ab9b8890771fe73dbeca78c29
```

本节代码提交对应的WO-A正式源仍未改；后续接回必须保留原64个N128任务owner，
在owner内部拆两块M16，不能直接沿用微测试128任务的分配并误用旧完成信号。
下一门禁是完整根重新做生产等价和Native WO-A；量化、WO-B和HC-post暂不修改。

证据：[82参数CPU资格](evidence/cpu-woa-observe-v20/receipt.json)、
[原helper](evidence/cpu-woa-micro-v4-0/receipt.json)、
[自然M16](evidence/cpu-woa-micro-v4-1/receipt.json)、
[旋转M16](evidence/cpu-woa-micro-v4-2/receipt.json)、
[完整根WO-A首差](evidence/spmd-observe-v19/result.json)、
[v1脚本问题](evidence/woa-micro-v1/result.json)、
[v2微测试结果](evidence/woa-micro-v2/result.json)。

### WO-A完整接回：维持M128并消除39差

四臂微测试v3（`task_20261028_172347_40032144471`，物理8，29.57秒）补测
M128/N128半程旋转：Native BF16差异同样为0，三次NaN FULL回放、guards与只读检查
通过。因此正式修正继续使用M128，保留原64个N128任务owner，没有拆M16。
这仅证明数值与布局资格，没有测新延迟，不能据此声称性能提升。

正式helper仅在`T==24 && A_COL_TILE==128 && A_ROW_TILE==128`启用；其它T/布局
保留原路径。packed步长仍取真实容量轴，写回24行，工作区padding不作为精度对象。
量化与WO-B均未修改。按组通知仍在本owner整块完成和原release之后发布。

接回的无设备v22–v25在NZ偏移证明处失败，没有占卡。检查失败前IR后确认：局部
`column_blocks=8`会以Scalar别名出现在商余除数中，而NZ非负证明直接要求常量除数。
商余改为编译期表达式`O_LORA//128`后，v26通过。另将K起点写成constexpr 0/8，
并由物理worker枚举全局条带后过滤组区间，避免含减法的动态起点源链；独立枚举
24个worker×45个组区间，所执行的logical序列与原公式逐项一致。前序`max`会被
优化删除，不能把加`max`当作修复。没有修改SDK、生成代码或绕过NZ验证。

v26正式55、debug57、观察82参数根全部通过编译、load和真实双AIV分派检查。
独立生成物审查确认K0/2048起点、16个K256块的原累加次序、有效24行和原发布规则。
静态资源最高端为Vec163872、Mat475136、L0A/L0B各65536、L0C131072字节，均未超限。

| 完整接回验证 | 结果 |
| --- | --- |
| 生产v20 FULL/eager八状态、guards | 全部通过 |
| 观察v20与生产八状态、同输入及24个源码SHA | 全部通过 |
| Native详细/最小hook控制八状态 | 全部0差异 |
| 原HC-pre至post-inverse heads的33项门禁 | 全部0差异 |
| **WO-A BF16** | **39 → 0差异** |
| 最终输出 | **41858差**，最大绝对差0.015625，首差即停 |

生产v20为`task_20261028_175322_144460319103`（220.57秒），观察v20为
`task_20261028_175930_176488129919`（217.98秒），均显式物理8且已退出。
生产使用已完成资格的57根时，82/55在无设备环境继续编译；观察提交前82资格已通过。
复现采用新生产`production.pt`，不能复用23项源身份的旧快照。

本轮只证明8K/B4/T24单层的WO-A修正，没有证明完整CSA/整模型token一致。
最终输出仍不通过；下一步基于同次Native WO-A BF16入口及实际O-proj输出，分别
测试整行动态量化、八组INT32先合并后反量化，保持首差停止和每任务15分钟内。

证据：[M128微根CPU](evidence/cpu-woa-m128-v1/receipt.json)、
[四臂微测试](evidence/woa-micro-v3/result.json)、
[完整57资格](evidence/cpu-woa-integrated-v26-spmd-debug/receipt.json)、
[观察82资格](evidence/cpu-woa-integrated-v26-spmd-probe/receipt.json)、
[服务55资格](evidence/cpu-woa-integrated-v26-spmd/receipt.json)、
[生产v20](evidence/spmd-production-v20/result.json)、
[观察v20](evidence/spmd-observe-v20/result.json)。

### WO-B后缀的两处Native数值边界

在完整SPMD v20的WO-A已归零后，独立微测试从同一次Native的真实
`wo_b` BF16入口出发，读取checkpoint正式layer4的INT8 WO-B、FP32 scale和全0
量化offset。先用同版本`npu_dynamic_quant → npu_quant_matmul`复现实际hook捕获的
Native projection，逐bit为0，避免用自建参考链代替真实金值。

与本分支修正前的继承实现相比，Native在这里有两处不同：

1. WO-A FP32结果先舍入BF16，然后在完整8192列上做每token动态量化；继承实现
   从FP32直接做每1024列独立量化，既改变舍入边界，也改变量化尺度。
2. Native先完成8192维INT32点积，再一次转FP32；先计算`weight_scale*token_scale`，
   再与点积相乘并舍入BF16。继承实现逐1024组转FP32、乘不同scale后相加，数值边界不同。

独立诊断用纯DSL复现上述Native顺序，复用原WO-B的INT32 partial生产者；
八组INT32先合并，最坏范围`8192*128*128 < 2^31`，不会溢出。

| 8K/B4/T24后缀微测试 | 相对同次Native的逐bit差异 |
| --- | ---: |
| Native重放实际projection控制 | 0 |
| DSL整行INT8量化、FP32 scale | 0 |
| DSL WO-B BF16输出 | 0 |
| 三次毒化后的FULL replay/eager | 全部0 |
| INT8 matrix padding、保护区、只读输入/权重/scale | 全部通过 |

任务`task_20261028_201347_32824666272`显式物理8，诊断26.60秒，已完成exit0。
该结果只证明此T24实际边界输入下的后缀，不自动授予完整SPMD、其它T、整模型
逐token或性能资格。完整接回仍须重新验证原owner/发布、生产/观察等价和最终输出。

复现：先取得WO-A归零的完整观察件，沿用本目录env/bootstrap；无设备执行
`compile_rms.py <新目录> oproj-quant`与`oproj-b`，核对ABI分别为3/6，且不初始化NPU。
再按既定时限显式物理8提交`run_rms.sh <新标签> oproj`，另传：

```bash
--env OPROJ_PROBE_REFERENCE=<完整观察目录>/observation.pt
```

`probe_oproj.py`先校验观察件的WO-A、生产等价、Native hook控制和同输入门禁，
然后逐门检查Native重放、量化、WO-B；首差立即停止并保存实际值/金值及partial。
正式接回时保留这份微测试和原精度基准，不能覆盖历史结果。

证据：[量化CPU根](evidence/cpu-oproj-suffix-v1-oproj-quant/receipt.json)、
[WO-B CPU根](evidence/cpu-oproj-suffix-v1-oproj-b/receipt.json)、
[后缀NPU微测试](evidence/oproj-suffix-v1/result.json)。

### 完整SPMD接回：T24最终输出首次全部逐bit一致

正式修正保留单入口、单MIX、真正TND的原数据流和55/57参数公开接口，计算全部
使用DSL。没有修改SDK、runtime或生成的C++。修正只在`T==24`启用，其它T保留
此前实现，不能将本节结果扩展为所有T已通过。

- 量化仍用原`(group,32行块)%48`的owner；T24仅worker0…7各写自己的1024列、
  自己的scale与原INT8尾零。每个owner重复读取全8192列计算BF16 amax，保留原
  writer/发布合同。此处优先完成精度闭环，尚未优化重复搬运。
- 全行amax前，只有实际有量化任务的AIV等待八组WO-A的真实Cube owner。Cube仍
  先完成并发布本核所有WO-A，再进入任何量化等待，因此没有新增等待环。
- 各组scale逐bit相同；全0行scale仍为0，只把分母副本改为1，不沿用旧EPS floor。
- WO-B仍复用原INT32 partial生产者，最后先整数归并八组，再一次FP32联合scale、
  BF16舍入，最后执行原HC-post顺序。其它T的按组量化、FP32归并和等待保持原样。

无设备资格：55服务根、57生产debug根、82观察根均完成compile/load，未初始化NPU，
并检查实际`MixedKernels={0,1,1}`双AIV。独立生成物审查确认全组等待和发布、
amax/dequant/denominator的Tile生命周期、整数carry与scale互不覆盖；Vec静态峰值
仍163872字节，未增加原Cube资源上限。生成物仅用于审查，不作为手写算子源码。

| 8K/B4/T24完整接回门禁 | 结果 |
| --- | --- |
| 生产v21：FULL/eager八个输出与cache/state | 全部逐bit一致 |
| 观察v21：与生产八状态、39项同输入、25份源码SHA | 全部一致 |
| 观察根FULL/eager：八外部状态、25项阶段观察 | 全部逐bit一致 |
| Native详细hook/最小hook八状态控制 | 全部逐bit一致 |
| HC-pre至WO-A的原34项Native门禁 | 全部逐bit一致 |
| **最终HC-post输出，相对Native** | **41858 → 0差异，max_abs=0** |
| **全部35项Native门禁** | **全部逐bit一致** |
| 只读metadata/cache保护区 | 全部通过 |

生产`task_20261028_202510_345251811023`用时208.00秒；
观察`task_20261028_202931_352249218285`用时215.42秒。两项均显式物理8、780秒外层/
750秒内层时限，没有使用其它卡。生产JSON的`pass:false`表示没有执行Native验收，
须与其`phase=production`及八状态结果一起阅读；最终观察JSON为`complete:true, pass:true`。

**验收范围：**真实checkpoint第4层权重、合成hidden/history、8K上下文、B4/T24；
同一输入下完整HC-pre至HC-post替换，包含TopK与六类cache/state。不是整模型
192-token验收；也未证明其它T、短/混合query、其它层/seed或128K精度。
本轮没有测延迟，不能将任务总用时当CSA性能，也不能声称已追平nalinaly。
旧精度分支及本分支修正前的提交保留，可正常revert。

复现完整接回：沿用固定env/bootstrap，依次无设备运行`compile_rms.py <新目录>`的
`spmd-debug`、`spmd-probe`、`spmd`三个selector，核对57/82/55 ABI及双AIV回执。
随后按本文相同`task-submit --device 8 --max-time 780`和env参数：

1. `run_rms.sh <新生产标签> spmd`，取得新`production.pt`，核对八状态和guards。
2. 新任务传`--env SPMD_PRODUCTION_REFERENCE=<新生产目录>/production.pt`，执行
   `run_rms.sh <新观察标签> spmd`。必须检查25源码SHA和39项同输入后，才解释35项Native门禁。
3. 任一首差即停止后续配置，保存JSON和`observation.pt`供定位，不弱化逐bit断言。

证据：[55参数CPU](evidence/cpu-oproj-integrated-v1-spmd/receipt.json)、
[57参数CPU](evidence/cpu-oproj-integrated-v1-spmd-debug/receipt.json)、
[82参数CPU](evidence/cpu-oproj-integrated-v1-spmd-probe/receipt.json)、
[生产v21](evidence/spmd-production-v21/result.json)、
[Native完整对拍v21](evidence/spmd-observe-v21/result.json)。
