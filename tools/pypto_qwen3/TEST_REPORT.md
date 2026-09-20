# Qwen3-14B Attention-only PyPTO kernel 测试报告

本分支只替换 Qwen3 DecoderLayer 的 Attention residual block：input Add/RMSNorm、
QKV、Q/K RMSNorm、RoPE、KV cache 更新、paged attention、o_proj 和
post-attention Add/RMSNorm。MLP、KV 页所有权、调度和采样保持原生。

## 固定环境

- vLLM：`0.20.2+empty`
- vLLM Ascend：`0.20.2rc1` 基线
- PyPTO：`b172b5e0`
- Simpler：PyPTO gitlink `4f162da0`
- ptoas：`0.54`
- CANN：`9.0.0`
- 模型：Qwen3-14B BF16、TP1、KV block size 128

PyPTO `b172b5e0` 生成的 `pto.load_scalar` 与 ptoas 0.54 配套。系统 ptoas 0.64
已经移除该指令；混用会在编译 paged attention 时失败，不能据此判断算子实现错误。

固定 CANN 9.0 不提供 `aclnnAddRmsNormBias`。整网脚本仅关闭依赖该新算子的四个可选
vLLM-Ascend graph fusion pass；原生 MLP 和 ACLGraph 仍保持启用。这是测试环境兼容配置，
不是 PyPTO attention 的 fallback。

## 已通过

Host UT：模式校验、配置 hash、qkv/o_proj ND 权重布局、attention-only 模式不关闭
原生 MLP fusion，以及 capture warmup metadata 策略，共 `7 passed`。

真实 NPU 单 block（task-submit）：

- 首层和普通层 eager；
- capture 后两次 replay，并在地址不变时更新输入内容、slot mapping 和 block table；
- 129-token 跨两个物理页；
- 两请求 packed prefill 使用不同物理页；
- `slot_mapping=-1` 不写 KV cache；
- graph reset 和正常退出。

通过标志：

```text
QWEN3_ATTENTION_RESIDUAL_BLOCK PASS
```

最大观测绝对误差为 `0.03125`，属于 BF16 舍入量级。golden 使用独立 Torch 公式实现
RoPE、分页寻址与 causal attention，不依赖 ATB attention 实现。

## 整网命令

```bash
VLLM_ASCEND_PYPTO_QWEN3_MODE=attention_block \
python tools/pypto_qwen3/run_attention_block_e2e.py \
  --model /data/models/Qwen3-14B --enforce-eager
```

去掉 `--enforce-eager` 后重启进程执行 ACLGraph。本分支已经完成真实整网 Graph 复验：

- 启动配置的 `splitting_ops` 不含 PyPTO attention block；
- capture 前生成 first/regular 两个 PyPTO JIT artifact；
- 两个 bucket 捕获成功，graph memory 为约 0.05 GiB；
- 请求期日志出现 `Replaying aclgraph`；
- 两个 prompt 的 token IDs 与 eager/原生基线一致；
- 正常 shutdown，最终退出码为 0。

```text
Hello -> [25, 358, 614, 264] -> ": I have a"
The capital of France is -> [12095, 13, 3555, 374] -> " Paris. What is"
```

PyPTO capture 遇到未提前 prepare 的 callable 会 fail-closed，因此上述捕获成功同时证明
所需 callable 已在 capture 前的 eager warmup 完成 prepare，而不是在 capture 内补编译。

## 尚未完成

- 当前 kernel 是接口和正确性基线；还没有移植 PyPTO-Lib 最新 decode 的 4×128 page stack、
  24 mixed-core groups 及 QK/softmax/PV 流水。
- prefill 已按 vLLM packed-token 契约实现正确性路径，但尚未移植 PyPTO-Lib 的独立
  packed prefill specialization。
- 长稳、性能 profile 和 PyPTO-Lib 高性能 prefill/decode specialization 尚未完成；本报告的
  Graph 结论只覆盖 2 prompts、每个 4 output tokens 的正确性短测。
