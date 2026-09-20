# Qwen3-14B PyPTO Attention Block

This experimental path replaces only the Qwen3 decoder attention residual
block with a PyPTO L2-kernel callable:

```text
input Add/RMSNorm -> QKV -> Q/K norm -> RoPE -> KV cache update
-> paged attention -> output projection -> post-attention Add/RMSNorm
```

The vLLM-Ascend MLP, request scheduler, KV block manager, sampler, and public
interfaces remain unchanged.

Enable it with:

```bash
export VLLM_ASCEND_PYPTO_QWEN3_MODE=attention_block
```

The initial supported configuration is Qwen3-14B BF16, TP1, 128-token KV
pages, no quantization, no speculative decoding, and no chunked prefill.

Run `test_attention_residual_block.py` through the hwServer `task-submit`
scheduler. It covers first/regular layer semantics, eager execution, capture
and replay, a cross-page decode, packed multi-request prefill, and the
negative-slot no-write rule. See [TEST_REPORT.md](TEST_REPORT.md) for the
fixed toolchain and current evidence.

Run the Qwen3-14B integration with:

```bash
VLLM_ASCEND_PYPTO_QWEN3_MODE=attention_block \
python tools/pypto_qwen3/run_attention_block_e2e.py \
  --model /data/models/Qwen3-14B --enforce-eager
```

Omit `--enforce-eager` in a fresh process for ACLGraph.
