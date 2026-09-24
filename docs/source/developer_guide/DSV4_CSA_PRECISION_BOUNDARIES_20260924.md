# DSV4 CSA 数值边界单因素对拍（2026-09-24）

## 测试范围与复现条件

- 仓库分支：`dev/pypto-dsv4-csa-v0.25.1rc1-cann9.0.1`。
- 被测运行时代码基线：`778bf158b53e3a7dbd86065d775115e3e6d01d3e`。测试开始时分支 HEAD 为 `cf271b359be3ae3c72f883733721aaa296f69955`；该提交仅增加测试历史文档，不改运行时代码。
- 环境：CANN/NNAL 9.0.1、Python 3.12.14、vLLM 0.25.1、vLLM-Ascend 0.25.1rc1、Torch 2.10.0+cpu、Torch-NPU 2.10.0.post2、PyPTO `54957491`、Simpler `166852bf`、PTOAS 0.63；Ascend A3。
- 权重：`DeepSeek-V4-Flash-0731-w8a8` 第 2 层的真实 attention 权重。不是三层裁剪模型；每个候选都单独运行这一层。
- 每个候选：TP1、B4、S6、起始 position 8191；使用固定合成 hidden 和约 8K 历史 cache。使用 eager 单层 kernel，不开 graph，不测性能，不代表整模型生成。
- 原生参照：相同 attention 实例和权重执行 `AscendDSAImpl.forward`；CSA 候选从同一实现基线独立生成，只加入一个舍入/量化因素和结果观测输出。每次重置 hidden/cache。
- hidden SHA256：`7d1aba3522c00ce722a262000d154b8e90902700aca562b3e75d3f81c21f6550`。三份初始 cache SHA256 在所有成功候选中完全一致；模型权重测试前后未变。
- 记录 QR、QR scale、Q、raw KV、attention heads、top-k、最终输出，以及 compressed/raw/main state/inner state/index key/index scale 六类 cache。阈值固定为 `torch.allclose(rtol=1e-2, atol=1e-2)`；relative-L2 定义为 `||candidate-native||₂ / ||native||₂`。

生成的测试 overlay 在运行时基线 `778bf158` 上增加了中间张量观测签名。除单因素所在文件外，候选源文件 hashes 相同；下表列出关键文件 SHA256。它们是隔离测试 overlay 的 hash，不是生产文件 hash。

| 候选 | `qkv_proj_rope.py` | `decode_csa.py` | `decode_o_proj.py` |
|---|---|---|---|
| control | `b6cfc1a9f3d6c3b190d71acf4658d5b08269bfb050ce67f4282557237f09e4ae` | `c1f089ac52eb1dea9cde2b14545610ab1664d789ef39a77764afa34bb0a1c0cb` | `0be385e878931c96348dbb23259aeed7fff93c4990a84ae074bc2717c34bcb93` |
| Q-A BF16 | `337579850e1c45259531b7d4c56e791beb41e8284a658f808b91db401c3c3bbc` | `c1f089ac52eb1dea9cde2b14545610ab1664d789ef39a77764afa34bb0a1c0cb` | `0be385e878931c96348dbb23259aeed7fff93c4990a84ae074bc2717c34bcb93` |
| Q-B BF16 | `412bcbf62ded401d143f1a915389c41396575c133d6ff7f33e4f0cd66b6bb07d` | `c1f089ac52eb1dea9cde2b14545610ab1664d789ef39a77764afa34bb0a1c0cb` | `0be385e878931c96348dbb23259aeed7fff93c4990a84ae074bc2717c34bcb93` |
| Q RMS BF16 | `094f86103456fb270a1aefb9c35b238be7e9a8d090bc53e035e1186ae611c467` | `c1f089ac52eb1dea9cde2b14545610ab1664d789ef39a77764afa34bb0a1c0cb` | `0be385e878931c96348dbb23259aeed7fff93c4990a84ae074bc2717c34bcb93` |
| KV projection BF16 | `76ff951cf82a30008987d503aa9b4bf7b48c8fb51766993ad0815054db33283f` | `c1f089ac52eb1dea9cde2b14545610ab1664d789ef39a77764afa34bb0a1c0cb` | `0be385e878931c96348dbb23259aeed7fff93c4990a84ae074bc2717c34bcb93` |
| KV RMS BF16 | `304c03799b81e7941060a2536b5ba230b2d161be570ec73cfdc8da744b5400a2` | `c1f089ac52eb1dea9cde2b14545610ab1664d789ef39a77764afa34bb0a1c0cb` | `0be385e878931c96348dbb23259aeed7fff93c4990a84ae074bc2717c34bcb93` |
| O-A BF16 | `b6cfc1a9f3d6c3b190d71acf4658d5b08269bfb050ce67f4282557237f09e4ae` | `0370809b17eca1b104199b92d6d4b13be603fee97818bab3cfb9f510923ee3f6` | `d29dfcfda486b3bb56192ec02cce86b0d3a5c2dc12731e42f3044f08f5b98aa4` |
| O-B full row | `b6cfc1a9f3d6c3b190d71acf4658d5b08269bfb050ce67f4282557237f09e4ae` | `0370809b17eca1b104199b92d6d4b13be603fee97818bab3cfb9f510923ee3f6` | `6c12ef12c6d907cab0af126ae9dccbd1f9846967fd4573fb3e0fc350ffa6ebb4` |

## 单因素结果

表中各误差均相对原生输出。`同 heads O-proj` 是把同一份 CSA heads 交给原生 O-proj 后，对照候选 CSA O-proj 输出，用于隔离输出投影误差。输出列即使进程成功退出，只要未达到 allclose 仍记作未通过。

| 候选 | Q | raw KV | heads | 最终输出 | 同 heads O-proj | 相对 control 的 cache 变化 |
|---|---:|---:|---:|---:|---:|---|
| control | 0.6758% | 0.2810% | 1.9196% | 2.2210% | 1.5741% | 基线；index key 对原生为 0.7068%，未过阈值 |
| Q-A 累加后 BF16 round | 0.2872% | 0.2810% | 0.7586% | 1.7320% | 1.5782% | 六类 cache 不变 |
| Q-B dequant 后 BF16 round | 0.6854% | 0.2810% | 1.9195% | 2.2243% | 1.5766% | 六类 cache 不变 |
| Q RMS 到 RoPE 前 BF16 round | 0.6765% | 0.2810% | 1.9194% | 2.2193% | 1.5714% | 六类 cache 不变 |
| KV projection 到 RMS 前 BF16 round | 0.6758% | **0.0936%** | 1.8993% | 2.1959% | 1.5760% | 仅 raw KV cache 改善到 0.0936%；另外五类不变 |
| KV RMS 到 RoPE 前 BF16 round | 0.6758% | 0.2787% | 1.9189% | 2.2163% | 1.5804% | 仅 raw KV cache 有变化；另外五类不变 |
| O-A 累加后 BF16 round，固定同一 heads | 0.6758% | 0.2810% | 1.9196% | 2.2225% | 1.5102% | 六类 cache 不变 |
| O-B 完整 8192 维共享 amax/scale，固定同一 heads 和 O-A raw | 0.6758% | 0.2810% | 1.9196% | **2.0639%** | **0.5943%（allclose 通过）** | 六类 cache 不变 |

Q-A BF16 单因素让 heads relative-L2 从 1.9196% 降至 0.7586%，最终输出降至 1.7320%，但最终输出依然未通过 `1e-2` 阈值。Q-B 和 Q-RMS 在本固定输入下没有实质改善。KV projection BF16 明显改善 raw KV，但 heads/最终输出只小幅改变。KV RMS BF16 对 raw KV 的改善很小。

### O-proj 两项独立诊断

两次 O-proj 测试都锁定同一批 attention heads。O-A BF16 和 O-B 全行量化是两条独立候选，没有把两项叠加。

- **O-A BF16：**舍入后的 O-A 中间张量与原生 O-A 输出逐元素完全相同（relative-L2 0）。Q、raw KV、heads、top-k 与六类 cache 和 control 逐元素相同。固定同 heads 的 O-proj 输出误差由 control 的 1.5741% 微降到 1.5102%，仍未通过。与 native full-row O-proj 量化比，原来每组 1024 维各算 scale 的量化有 170,130/196,608 个 int8 值不同，scale relative-L2 为 2.517；O-A 的中间值已对齐，不足以单独解释 O-proj 数值差异。
- **O-B 全行 8192：**从同一 O-A raw FP32 和同一 heads 独立执行，仅把原来每组 1024 维的 amax/scale 改为整行 8192 维共享。kernel 输出的 int8 codes 和 scales 与 FP32 显式参考公式逐元素完全一致。对原生 O-proj 的同 heads 对拍误差为 0.5943%，通过阈值；与原生量化相比，196,608 个 int8 值中只有 2,079 个相差 1 个码，scale relative-L2 为 0.5118%。O-A raw FP32 与 native BF16 O-A 的误差为 0.1653%。
- 因而，**O-B 量化范围是当前 O-proj 与原生表示之间的一项主要差异**：单独改为整行 scale 后，固定 heads 的 O-proj 误差从 1.5741% 降到 0.5943%。但最终整层输出仍为 2.0639%，因为上游 attention heads 误差仍为 1.9196%。尚未测试“O-A BF16 + O-B 整行量化”的组合，不能把单项结果外推成组合版的最终精度。

## 任务和失败记录

| 因素 | 任务 ID | 设备 | 结果 |
|---|---|---:|---|
| control | `task_20260924_165129_16889987557` | 0 | 完成 |
| Q-A BF16 | `task_20260924_165853_279324730931` | 0 | 完成 |
| Q-B BF16 | `task_20260924_170353_2949312820` | 0 | 完成 |
| Q RMS BF16 | `task_20260924_171638_3539375883` | 0 | 完成 |
| KV projection BF16 | `task_20260924_172335_406238521975` | 0 | 完成 |
| KV RMS BF16 | `task_20260924_173021_21319224126` | 0 | 完成 |
| O-A BF16（修正参考后） | `task_20260924_175136_106088715122` | 3 | 完成 |
| O-B full row | `task_20260924_175710_1267592503` | 0 | 完成 |

以下尝试没有产生有效精度结论：`task_20260924_173246_2613438306` 的 O-A kernel 已完成，但 runner 在指标阶段把 FP32 传入 CANN 9.0.1 不支持 FP32 的 `npu_dynamic_quant`，故保留失败日志，未计入结果；修正为 CPU 端明确复现 kernel 量化公式后，`task_20260924_173948_5254456718` 在 600 秒默认队列超时前未分到设备并被取消；没有开始测试。更早的 control 两次尝试分别是调试签名传参错误和漏掉权重后处理初始化，也未进入有效的 kernel 对拍。所有有效候选均重置输入/cache，且实际设备锁在进程结束后释放。

## 结论和限制

1. 当前 CSA kernel 与原生 DSA 仍有明确精度差距；本轮 control 的 attention heads error 为 1.9196%，最终输出为 2.2210%，均未通过阈值。
2. 单因素边界确认了两条有效方向：Q-A BF16 舍入显著改善 heads；O-B 的整行 8192 维 scale 使固定-heads O-proj 对拍通过。KV projection BF16 主要改善 raw KV/cache，而非最终输出。
3. O-B 不会修复 attention heads 之前的误差，最终输出仍未通过。尚未测 O-A BF16 与 O-B full-row 合并后的整层结果。
4. 这是单层、合成 hidden/history、TP1、B4/S6、eager 的归因实验；没有测 graph 性能、真实服务请求、完整模型精度或端到端吞吐，不可外推为这些场景通过。
