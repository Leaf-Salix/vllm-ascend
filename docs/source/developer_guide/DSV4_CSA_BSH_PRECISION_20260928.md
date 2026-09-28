# DSV4 CSA 对照 BSH 精度实现的数值修正（2026-09-28）

## 结论与验收范围

本轮在 CANN 9.0.1 / vLLM-Ascend 0.25.1rc1 上，用完整 `DeepSeek-V4-Flash-0731-w8a8` 的第 2 层真实 attention 权重，对照原生 `AscendDSAImpl.forward`。最终 Leaf 候选 `native_boundaries_v4` 的整层输出 relative L2 从本轮 control 的 **1.716002%** 降至 **0.074391%**，通过固定的 `allclose(rtol=1e-2, atol=1e-2)`；本固定输入的最终输出与参考 BSH 精度 kernel **98,304 个 BF16 元素全部相同**。

Q、QR INT8、QR scale 已与原生逐元素完全相同。attention heads 的 99.8524% 元素与原生 bitwise 相同，原生绝对值不小于 0.01 的 heads 全部在 1 BF16 ULP 内。固定同一份 heads 后，Leaf O-proj 和原生 O-proj 输出完全相同。

后续非观测生产入口的 graph 验证也已完成：8K/128K、B1/S1、B1/S6、B5/S6、B8/S6 的 A/B/A replay 全部复现各自 eager 输出和 cache，最终版 A/B 输出均通过原生 allclose。精度修正没有带来速度提升：B4/S6/8K 的 CSA graph 中位耗时由 0.60762 ms 增至 0.67448 ms；128K 为 1.00610 ms，对应同进程原生为 0.81910 ms。

这不等于“整层只差几个 bit”：最终输出与原生有 95.0633% 元素 bitwise 相同；显著值最大相差 10 BF16 ULP，接近零的少数值有更大的有序 ULP 距离。全模型多层误差累积、真实服务、长期 cache 状态和性能需要各自验收，不能由本次单层结果推断。

## 源码、环境与固定输入

| 项目 | 固定条件 |
| --- | --- |
| Leaf 分支 | `dev/pypto-dsv4-csa-v0.25.1rc1-cann9.0.1` |
| 本轮 control | 文档 HEAD `6fbcdf8a1a4f7babcb7db60448bf3d610edb39ec`；运行时代码基线 `778bf158b53e3a7dbd86065d775115e3e6d01d3e` |
| 修复与报告归档 commit | 本提交；实际运行源码身份以本报告 SHA256 表和冻结候选 manifest 为准 |
| 参考实现 | [nalinaly/dsv4-flash-pto-v0.25.1rc1 的 `0d1b8ea7292ffefc6e90cac0e2f4308749be53db`](https://github.com/nalinaly/vllm-ascend/tree/0d1b8ea7292ffefc6e90cac0e2f4308749be53db)；使用 `deepseek_v4_flash_dspark` 精度包，未使用 `_perf` 包 |
| 设备与软件 | Ascend A3；CANN/NNAL 9.0.1；Python 3.12.14；vLLM 0.25.1；vLLM-Ascend 0.25.1rc1；Torch 2.10.0+cpu；Torch-NPU 2.10.0.post2 |
| 编译依赖 | 沿用既有隔离环境：PyPTO `54957491`、Simpler `166852bf`、PTOAS 0.63；每个进程记录实际 Python import 路径、CANN 与 PTOAS 路径 |
| 模型和输入 | 完整模型第 2 层真实权重；TP1、B4、S6、起始 position 8191；固定合成 hidden 和约 8K 历史 cache |
| 运行方式 | 单层 eager；同一 attention 实例和权重，分别执行原生和候选；每侧使用同一初始数据的独立 cache allocation |
| 固定阈值 | `torch.allclose(rtol=1e-2, atol=1e-2)`；relative L2 为 `L2(candidate-native) / L2(native)`；没有放宽阈值 |
| 记录阶段 | QR、QR scale、Q、raw KV、heads、top-k、最终输出，以及 compressed/raw/main state/inner state/index key/index scale 六类 cache |

A3 实测的逻辑 hidden/history SHA256 为 `eb841527b5fda671b680d4c30db61722816963a87cd53cc78c3af5e3e519fd3d`；hidden SHA256 为 `bfddbf195fb64adf84a94c246efd1390e282e066d16bbbd8ea4ab651027d9ccf`。control、v1～v4 与 BSH 的逻辑输入一致。Mac CPU 的预检查摘要不同，因此没有拿跨架构随机数生成结果充当 NPU 输入一致性证据。

BSH 的 KV/state 物理页面是 32/2 token，Leaf 是 128/8 token。测试器按逻辑请求、cache 类型和绝对位置填入相同数据，再分别构造符合各自接口的物理页。独立核对表明，两种页面布局下原生的 QR、QR scale、Q、raw KV、heads、最终输出均逐元素相同。没有通过重新解释不兼容的页面 stride 来勉强运行参考 kernel。

## 原始 BSH 精度复核

先直接注册并调用参考分支未修改的 `decode_csa_tp1_attention_test`；再在隔离副本中把已有内部 GM 中间张量暴露为输出，重复运行 hook 版。两次的最终输出、全部 cache allocation、top-k indices 均 bitwise 相同，因此可把 hook 阶段记录用于解释原始 kernel 的结果。测试没有经过可回退到原生的服务 eligibility 入口。

参考 BSH 本身也不是整层 bitwise 原生：本次 output relative L2 为 0.074391%，heads relative L2 为 0.00121955%。Q/QR/QR scale 完全一致；raw KV 12,288 个值中有 1 个相差 1 BF16 ULP；固定相同 heads 的 O-proj 完全一致。后面的表同时保留 BSH 结果，避免把“精度版”名称当作全层逐 bit 验收。

## 逐组累加对拍

各候选继承上一组修正，使用同一输入、权重、原生基线和指标。这是分组累加实验；不能把一组中多个修改的收益全部归因于其中某一行代码。9 月 24 日的严格单因素实验见 [数值边界报告](DSV4_CSA_PRECISION_BOUNDARIES_20260924.md)。由于本轮使用跨页面 canonical 输入，不能直接与旧报告的误差数字相减来衡量改善。

| 候选 | 在上一组基础上增加的内容 | heads relative L2 | output relative L2 | output bitwise 相同比例 | output allclose |
| --- | --- | ---: | ---: | ---: | --- |
| control | `6fbcdf8a1` 原运行时代码 | 0.872300% | 1.716002% | 10.2773% | 未通过 |
| v1 | Q-A/Q-B/Q RMS、KV projection/KV RMS 的 BF16 边界；inverse RoPE 前 heads BF16；O-A BF16、整行 8192 维量化、O-B INT32 先合并后反量化 | 0.714955% | 1.191740% | 14.9567% | 未通过 |
| v2 | QR 归约、sqrt/标量除法和量化 scale 顺序；Q-B 组合 scale；indexer Hadamard、FP16/FP32 舍入与 scale 边界 | 0.088273% | 0.402876% | 42.9026% | 通过 |
| v3 | 512 候选 online softmax、sink 初值、BF16 概率 `round`、四个 128 Cube 切片共享 FP32 PV accumulator | 0.083086% | 0.200610% | 90.2964% | 通过 |
| v4 | Q-A 从 split-K 改为一个连续 FP32 accumulator（`QR_OK=1`）；同期仅更新 O-proj CPU golden 的参考边界 | **0.00124083%** | **0.074391%** | **95.0633%** | **通过** |
| BSH 精度参考 | 参考提交原始精度 kernel | 0.00121955% | 0.074391% | 95.0633% | 通过 |

v1 的固定同 heads O-proj relative L2 已降至 `7.94e-10`；v2～v4 为 0。v2 起，index key 与 FP16 index scale 全部精确匹配原生。

v3 残差定位到了具体 token：token 12 的 QR scale 为 `0.003660938935354352`，原生/BSH 为 `0.0036609156522899866`。虽然 QR INT8 已相同，这个 scale 差异仍使 Q 有 74 个元素改变，并让 top-k 选中集合出现 `538 ↔ 1014` 的替换，该 token 因此占据大部分 heads 误差。v4 保留一个连续 Q-A FP32 accumulator 后，QR scale、Q 和 top-k indices 均与 BSH 完全相同，主要误差随之消失。

## v4 阶段与 cache 精度

下表 relative L2 为无单位小数；cache 统计取本步原生写槽，避免大量未改变历史值稀释误差。FP32 state 的精确相同比例不高，但绝对误差约 `1e-6`；不能把它们写成 bitwise 一致。

| 阶段或 cache | relative L2 | 最大绝对误差 | 精确相同比例 | 说明 |
| --- | ---: | ---: | ---: | --- |
| QR INT8 | 0 | 0 | 100% | 24,576 个码完全一致 |
| QR scale FP32 | 0 | 0 | 100% | 对齐 `[T,1]` 与 `[T]` 形状后，24 个值完全一致 |
| Q BF16 | 0 | 0 | 100% | 786,432 个值完全一致 |
| raw KV / raw cache | `3.91752e-6` | `2.44141e-4` | 99.9919% | 12,288 个值中 1 个相差 1 ULP，与 BSH 相同 |
| heads BF16 | `1.24083e-5` | `2.44141e-4` | 99.8524% | 比 BSH 多 46 个不同值，主要对应 compressor 的剩余微差 |
| 最终输出 BF16 | `7.43908e-4` | `3.90625e-3` | 95.0633% | 与本次 BSH 最终输出全部相同 |
| 固定相同 heads 的 O-proj | 0 | 0 | 100% | 完全匹配原生 O-proj |
| compressed cache BF16 | `5.72013e-7` | `3.05176e-5` | 99.9512% | 4,096 个值中 2 个不同，均为 1 ULP；BSH 此项为全相同 |
| main state FP32 | `2.91128e-7` | `1.66893e-6` | 17.3340% | 保留 Leaf compressor 累加顺序；BSH 此项为全相同 |
| inner state FP32 | `3.26217e-7` | `2.62260e-6` | 22.0540% | 保留 Leaf compressor 累加顺序；BSH 此项为全相同 |
| index key INT8 | 0 | 0 | 100% | 1,024 个码完全一致 |
| index scale FP16 | 0 | 0 | 100% | 8 个写入值完全一致 |

v4 和 BSH 的 top-k indices 全部相同，top-k FP32 scores 仍有低位差异；不能从选中集合相同推导评分逐位一致。所有已记录张量有限，输出 guard 未变，测试前后模型参数未变。

## BF16 ULP 的两种统计范围

ULP 在这里表示 BF16 相邻可表示数之间的有序距离，不是“二进制位有几个不同”。同时报告全体有限值与 `|native| >= 0.01` 子集；后者用于帮助理解接近零时 ULP 的放大，没有用它过滤验收失败。allclose 仍检查全部元素。

| 张量 | bitwise 相同比例 | 全体有限值最大 ULP | 全体 P99 ULP | `abs(native) >= 0.01` 最大 ULP | 该子集 P99 ULP | 该子集在 1 ULP 内比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 Q | 100% | 0 | 0 | 0 | 0 | 100% |
| v4 raw KV | 99.9919% | 1 | 0 | 1 | 0 | 100% |
| v4 heads | 99.8524% | 93 | 0 | 1 | 0 | 100% |
| BSH heads | 99.8582% | 93 | 0 | 1 | 0 | 100% |
| v4 / BSH 最终输出 | 95.0633% | 29,024 | 2 | 10 | 1 | 99.1592% |
| v4 固定同 heads O-proj | 100% | 0 | 0 | 0 | 0 | 100% |

v4 heads 全部有限值中 99.9733% 在 1 ULP 内，99.9844% 在 2 ULP 内；最终输出全部有限值中 98.7498% 在 1 ULP 内，99.2635% 在 2 ULP 内。接近零的输出可能跨过符号或多个极小 BF16 间隔，因此全体最大 ULP 很大；必须结合最大绝对误差和 relative L2 一起阅读，不能省略该数字后声称“所有输出至多 1 ULP”。

## 实现范围与原生接口关系

本次必要数值改动限于以下六个源文件。没有引入 BSH 的 `NativeCSACall`、52-tensor ABI、32-token cache 页面或另一层服务 adapter。

| 文件 | 改动作用 |
| --- | --- |
| `attention/pto_attn.py` | post-load Hadamard 保留未缩放 ±1 矩阵；归一化放回 kernel 的原生 BF16 物化边界之后。没有新增调用参数。 |
| `attention/pto_kernels/dspark/qkv_proj_rope.py` | 补齐 BF16 物化边界，QR 归约、sqrt 舍入/设备标量除法、Q-B scale 运算次序及 Q-A 单 accumulator。 |
| `attention/pto_kernels/dspark/decode_indexer.py` | 对齐 Hadamard 的 BF16 matmul→归一化顺序、query scale 和原生权重系数的 FP16/FP32 边界。 |
| `attention/pto_kernels/dspark/decode_indexer_compressor.py` | 对齐 index key Hadamard 舍入及 FP16 cache scale；保持既有 state/cache 寻址。 |
| `attention/pto_kernels/dspark/decode_sparse_attn_csa.py` | SWA128+masked384、compressed512 两个 softmax 块；sink、概率舍入及 PV 累加边界；保留 Leaf inverse RoPE 组织。 |
| `attention/pto_kernels/dspark/decode_o_proj.py` | O-A BF16，整行 8192 维 amax/scale，O-B INT32 partial 先求和再按原生顺序反量化；同步 CPU golden。 |

`PyptoDSAImpl.forward` 的原生签名和返回方式保持，生产 kernel 仍为 **40 个 tensor 参数**。Leaf 原来的 metadata、`token_valid`、`window_swa_indices`、128-token cache 页面及零拷贝绑定继续使用。本轮没有增加独立适配层，也没有把既有内部绑定工作全部删除；这两件事不能混为一谈。阶段 hook 增加的七个输出仅存在于隔离测试入口，不属于生产 ABI。

v4 被测生产源文件的 SHA256 如下，来自冻结候选的 `source_manifest.json`；调试副本的 `decode_csa.py` 另加了观测输出。后续提交若仅格式化，应保留这组已执行源码身份，避免用新的文件 hash 倒填旧测试。

| 文件名 | v4 生产源 SHA256 |
| --- | --- |
| `pto_attn.py` | `19f2b7295836e505cbd02d08576f2152a3cabce93a7aea66053ff2e76e81bdec` |
| `qkv_proj_rope.py` | `f693d93bf45b80cd511426210a5aff7efeb10683ca775ba713af2649965a2853` |
| `decode_indexer.py` | `07a9f50b9b6fe468cd927c2f2ad6cd975451cf7545930de555ac01e373f2c3c7` |
| `decode_indexer_compressor.py` | `27549d9081cbba67a71acd839121f69f147661b9d063d3f7319b4374c3183376` |
| `decode_sparse_attn_csa.py` | `17c2cda50e03e38fb99a37b88c5c24d668c2bbefeeb057a50b12a336462601bf` |
| `decode_o_proj.py` | `6f414fc655d003c5de1e12a01b70dd2f850671cbfeec00e60337d6e94ecf9113` |

上表对应冻结的非观测生产源 `production_final`。归档前修正了 QKV 的 `QR_SPLIT_K_TILE` 注释，并经 Ruff 格式化 sparse 文件；最终 `production_release` 全部运行时 Python 文件的 AST 与 `production_final` 一致，没有改变计算。随后使用仓库内整理后的测试入口和 `production_release` 再做 B1/S1 NPU 冒烟，A/B 输出与原生完全相同，A/B/A replay 输出及 cache 字节检查通过。

## Graph、边界与长上下文验证

使用非观测 `impl.forward`，测试器确认加载正确的隔离源码、生产 kernel 只有 40 个参数且不含 `debug_*`。每侧先生成 eager A/B 参考，再捕获图并按 A/B/A 顺序回写同一 hidden 地址；B 为 A 乘以 `-0.375` 后转 BF16。每次重置该侧初始 cache，检查 replay output 和完整 cache allocation 的字节均与各自 eager 参考相同。CSA capture 必须真实命中，replay 期间 Python forward 调用计数必须保持不变，输出 guard 也必须保持。

下面全部最终版用例均满足上述检查，A/B 输出和六组本步 cache 写槽均通过固定原生 allclose。旧 control 的 replay 也精确复现了它自己的 eager 结果，但其 A/B 输出与 index key 对原生仍未通过精度检查，不能把 graph 复现正确误写成数值正确。

| 用例 | B / S | 起始 position | A 输出 relative L2 | B 输出 relative L2 | 原生 graph 中位耗时 | CSA graph 中位耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 旧 control，约 8K | 4 / 6 | 8191 | 1.716002%（未过） | 1.731235%（未过） | 0.59567 ms | 0.60762 ms |
| 最终版，约 8K | 4 / 6 | 8191 | 0.074391% | 0.115102% | 0.61942 ms | 0.67448 ms |
| 最终版，单 token | 1 / 1 | 8191 | 0 | 0 | 0.48130 ms | 0.59112 ms |
| 最终版，短历史跨页 | 1 / 6 | 127 | 0 | 0.003850% | 0.48171 ms | 0.55286 ms |
| 最终版，30 行尾部 | 5 / 6 | 8191 | 0.066747% | 0.109742% | 0.65023 ms | 0.77334 ms |
| 最终版，B8 | 8 / 6 | 8191 | 0.035738% | 0.045261% | 0.79217 ms | 0.85352 ms |
| 最终版，约 128K | 4 / 6 | 131071 | 0.056939% | 0.121151% | 0.81910 ms | 1.00610 ms |

计时使用 NPU Event，仅包围一次 graph replay；cache/hidden reset 和同步在计时区间外。原生/CSA 交替先后顺序，丢弃前 5 轮，分别保留 20 个样本后取中位数。旧/新版 8K 在同一张卡的不同进程测得，原生时间也有约 4% 漂移，因此不能把全部差值严格归因到某一个源码模块。当前观测是 CSA 旧/新版增加约 11.0%；最终版相对各自同进程原生，8K 慢约 8.9%，128K 慢约 22.8%。这些都是单层延迟，不是整模型吞吐。

Q-A 单 accumulator 和 512 候选 softmax 改变了原来的并行方式；本轮优先恢复原生数值边界，性能代价已如实保留。后续性能优化需继续守住当前阶段和 graph 精度基线。

| 项目 | 方法与通过条件 | 当前状态 |
| --- | --- | --- |
| 8K 原生与最终版 graph replay | 固定地址 hidden A/B/A，每次重置 cache，核对 eager/graph 一致性 | 通过 |
| 同卡 graph 延迟 | 原生/CSA 交替运行，Event 只测 graph replay | 已完成，见上表 |
| B1/S1、B1/S6、B5/S6、B8/S6 | 检查真实 kernel、原生输出及 cache 写槽，覆盖尾部和 position 127 跨页 | 通过 |
| 128K 原生对拍与 graph | TP1、B4/S6、起始 position 131071 | 通过 |
| 多步 cache 状态 | 连续推进 position 与 metadata、观察 FP32 state 微差是否累积 | 未完成 |
| 完整模型服务 | 真实 prompt、DSpark 接受情况、持续 decode 与完整模型精度 | 本轮单层实验不覆盖 |

## 证据与复现记录

归档目录按候选保存 `control`、`bsh`、`native_boundaries_v1`～`native_boundaries_v4`，每组含 `result.json`、`stages.pt` 和 `run.log`；`source_manifest.json` 保存冻结源码身份。BSH 另外保存未插入 hook 的结果及 hook 等价性检查。大张量和设备日志保留在测试归档，不在本报告中嵌入。

Leaf 使用 `precision_factor_ab.py`，BSH 使用 `bsh_hook_ab.py --diagnostics`。共同参数为 `--batch 4 --seq 6 --start-pos 8191`，权重固定为完整模型第 2 层；模型目录、源码 overlay、已编译扩展和独立结果目录均显式传入。所有版本执行真实 kernel，无 fallback 成功替代。Graph 使用 `graph_replay_ab.py` 的非观测源码入口，与 eager hook 数据分开记录。

可复用的无 hook 生产 graph 测试已整理进 [`tests/pto_attn/precision`](../../../tests/pto_attn/precision/README.md)，含模型层加载、原生 metadata/cache fixture、ULP、A/B/A 和 Event 计时；没有把私有服务器路径写死。入口及环境模板见该目录 README，可对历史 checkout 传入不同 `--variant-dir` 进行复测。

静态与 CPU 检查包括：初轮 adapter 合约回归 30 项通过，补充本轮回归后最终 33 项通过；512-block gather 的 1,228 组 32/128-page 边界枚举通过；更新的数学 golden 覆盖 B1/S1、B1/S6、短历史、跨页和全无效；sqrt 整数中点舍入算法 30,640 个 CPU 用例通过。上述检查不代替 NPU 编译、同步和精度验收。

最终增量 `pre-commit` 与 `bash format.sh ci` 全部通过。首轮因本机缺少 gitleaks、wget 和 shellcheck 未能完成两个工具检查；随后将官方 gitleaks 8.30.1 和 shellcheck 0.11.0 放入任务工具目录，校验 gitleaks 发布 SHA256 后重跑通过，没有跳过检查或修改系统环境。独立复核覆盖 QKV/O-proj 依赖、稀疏事件配对、尾块及共享 Hadamard 初始化，未发现阻断问题。

最初一次 CPU 合约启动器因 import 顺序触发已有循环导入，调整启动器先加载 ops 后通过；它没有产生 kernel 精度结论。NPU 排队时间也不计为执行时间。已有 v1/v2/v3 结果保留，不用最终版覆盖此前失败或较差数值。
