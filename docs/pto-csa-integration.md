# v0.25.1rc1：PyPTO DSV4 CSA 原生接口实现

## 来源与目标

基线为官方 `v0.25.1rc1`（`9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`），vLLM 0.25.1、A3、CANN 9.0.1、Torch 2.10.0、TorchNPU 2.10.0.post2。

此前迁移自 sunkaixuan CSA 分支及 PR #5（`72120920`）。本次 kernel 的 native slots、compact metadata 和物理页寻址参考 nalinaly `dsv4-flash-pto-v0.25.1rc1` 的 `8a4c4e6f01fc862fdd458c9238520946f46e22f3`。原生接口及 metadata 语义以官方 0.25.1rc1 为准。

## 入口和数据所有权

`VLLM_ASCEND_PYPTO_DSV4_CSA=1` 使 DSA backend 选择 `PyptoDSAImpl`，它继承 `AscendDSAImpl`。`forward` 与原生签名相同，原地写入调用方 `output` 并返回同一 tensor。默认关闭；CP backend 优先级保持不变。外层 `dsa_forward` 恢复原生实现，不增加调用参数、模型别名或独立 substitute/build_args 层。

| 数据 | 新实现处理 |
| --- | --- |
| 五组 metadata / 六项 KV tuple | 按原生顺序直接解包 |
| slots、query offsets、seq lengths、positions | 直接传原生 tensor，不重建 token mask 或 slot-to-page 表 |
| compressor / indexer compact RoPE 和 slots | 分别调用原生 `_compute_compressor_metadata` |
| 主 state / compressed KV | 共享原生分配；使用覆盖完整物理页的零拷贝 view |
| index state / key / scale | 保留原生共享页；kernel 消费物理页 descriptor |
| padded block table | 用 stride 建立零拷贝 view，检查最后一行实际存储边界 |
| 权重 | 原生 post-load 后准备转置和连续视图；不反量化或重新量化权重 |
| 输出 | kernel 直接写调用方 output，返回值与原生一致 |

内层 kernel 当前为 52 参数 ABI，包含必要权重及 scratch。它是实现内部调用，不增加原生 forward 参数。HC、输入 RMSNorm、MoE、缓存分配和调度仍使用原生代码。

## 当前支持范围

当前 kernel 专化为 A3、TP1、C4、均匀 S6、DSpark 出5验6、eager、32-token KV 页及2-row state 页。仅支持输入 Q/KV 投影为 BF16、后续指定投影为 INT8 的权重组合；输入投影量化的 checkpoint 保留原生执行。

prefill、profiling、ragged、dummy padding、其他 query 长度、graph、其他页规格、draft 非因果窗口、KV transfer、CP、LoRA、特殊 output projection TP、index cache 均在 kernel 写缓存前回退原生。不修改 allocator 来迎合 kernel，也不把回退生成成功算作 CSA 命中。

kernel 成功提交后通知 cache write 并保存；失败直接抛出，禁止再次原生执行。融合调用内部没有独立的 prolog/attention 通信重叠边界，不能宣称与原生重叠性能相同。

## 验证与限制

CPU 回归：`tests/pto_attn/test_decode_contract_cpu.py`，检查签名、返回值、指针别名、原生 producer、回退、生命周期与写后异常传播。使用独立 PyPTO feat 组合：`54957491`（含 PR #2867）与 Simpler `166852bf`，须重编译原生扩展并核对 revision。

新路径命中标志为 `[pto-native-csa] ... seq=6`。真实完整模型、S6 接受/拒绝后的多步 cache 状态、单层数值对拍必须重新验收。旧 40 参数版本的 S1、S6、graph 结果不能证明此版本通过。历史单层精度未通过，不放宽阈值；生成成功也不等于精度通过。

## 本次验证结果

- 原生接口 CPU 回归：23 项通过；完整 kernel lowering 通过；增量 pre-commit 通过。
- 独立审查后补齐输入投影量化时原生回退、融合调用生命周期通知及对应测试。
- 完整 `DeepSeek-V4-Flash-0731-w8a8`：TP1、DP/EP16、DSpark5、EPLB、eager、32-token 页；每rank提交4个短自然语言请求，各生成64 tokens，16个rank全部完成，确认 `[pto-native-csa]` S6命中，正常退出。
- 上述请求总数不能证明固定decode GBS64；CSA 128K、固定并发、graph及数值精度仍未验收。
- PyPTO `54957491` / Simpler `166852bf`：138项相关CPU回归、单卡eager和capture/replay冒烟通过。运行时graph冒烟不代表本CSA实现支持graph。
