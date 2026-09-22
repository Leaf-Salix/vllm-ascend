# Qwen3-14B PyPTO Attention-only 实验

`VLLM_ASCEND_PYPTO_QWEN3_MODE=attention_only` 只替换原生
`Qwen3Attention.forward` 中的 QKV、Q/K RMSNorm、RoPE、KV cache 写入、
paged attention 和 o_proj。DecoderLayer 的前后 Add/RMSNorm、residual、MLP，
以及 vLLM 的 KV 页分配、调度和采样仍保持原生。prefill 和 decode 均使用
PyPTO L2 kernel；不支持的配置直接报错，不隐式回退。

支持范围：Qwen3-14B BF16、TP1、128-token KV page、上下文长度不超过
512；不支持量化、推测解码和 chunked prefill。

通过 hwServer `task-submit` 分别运行 eager 和 ACLGraph 整网短测：

```bash
VLLM_ASCEND_PYPTO_QWEN3_MODE=attention_only \
python tools/pypto_qwen3/run_attention_only_e2e.py \
  --model /data/models/Qwen3-14B --enforce-eager

VLLM_ASCEND_PYPTO_QWEN3_MODE=attention_only \
python tools/pypto_qwen3/run_attention_only_e2e.py \
  --model /data/models/Qwen3-14B
```

两条命令应在不同进程运行。旧 attention residual-block 单算子脚本已从
本分支移除；其代码和历史结果保存在旧提交及实验备份分支中，不能当作
当前 attention-only 算子的验收结果。当前证据和未覆盖范围见
[TEST_REPORT.md](TEST_REPORT.md)。
