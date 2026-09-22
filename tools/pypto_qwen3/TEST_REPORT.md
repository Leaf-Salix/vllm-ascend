# Qwen3-14B Attention-only 测试记录

当前路径只替换 `Qwen3Attention.forward`；前后 Add/RMSNorm、residual 和 MLP
保留原生。PyPTO 直接借用 vLLM 的 Device KV cache、block table、slot
mapping、seq_lens 和 query_start_loc；不分配或复制 KV 页。

## 已验证

- Qwen3-14B BF16、TP1、2-token prompt、2-token 输出的 eager 和 PIECEWISE
  ACLGraph 短测；PyPTO 与原生 token IDs 均为 `[198, 43256]`。
- Graph 日志出现 `Replaying aclgraph`。完整 trace 中两次 forward 共 80 个
  PyPTO AICore 事件、160 个原生 AddRmsNorm 事件，没有原生 FIA 事件。
- 相关 Host UT 8 项通过；包含原生 DecoderLayer.forward 保留，以及
  PyPTO 借用 vLLM paging metadata 的检查。

本轮更新 PyPTO/Simpler feat 后的四份 Perfetto JSON、版本、运行条件和
性能观察记录于工作区的
`reports/vllm-qwen-kernel/perfetto-attention-feat-02c-6354-20260921/`。
旧 residual-block 单算子测试不作为当前路径的精度证明。

## 尚未验证

- 多请求、跨页、长 prefill、不同 capture bucket 的 attention-only
  精度和 replay；目前的两个短 token 输出不能外推至这些场景。
- 同条件下的长期性能稳定性与内存生命周期。

性能 trace 里约 9 ms 的 `vllm::pypto_qwen3_attention_only` 是 CPU 算子
范围，不是 Device kernel 执行时间。第一次 `npu::get_npu_format` 出现在
范围开始约 8.5 ms 后；格式查询本身很短。KV cache 只借用原有 storage
并创建 view，没有按层重建。该 Host 空档的更细分归因尚需 Python
分段计时或采样，不能将其直接称为 KV cache 重建或重复编译。

PyPTO feat 的 registered-op ST 已覆盖 eager、`torch.compile` 和
NPUGraph capture/replay，断言同一 specialization 的 compile/prepare
计数仍各为 1；该测试证明复用正确性，不是每次 Host 查找开销的性能测试。
