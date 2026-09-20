# Qwen3-14B PyPTO kernel 实验测试报告

本文冻结当前实验分支中 Qwen3-14B PyPTO L2 kernel 接入的测试范围、复现方式和已知限制。
该分支同时保留三种实验粒度，便于比较和回归：

| 模式 | PyPTO 替换范围 | 目的 |
| --- | --- | --- |
| `partial` | 仅 q/k RMSNorm | 验证最小算子接线和 ACLGraph 生命周期 |
| `attention_block` | DecoderLayer 的完整 Attention residual block | 验证 KV cache 写入、分页读取和 Attention 大函数边界 |
| `full` | 主要模型计算 | 验证全模型 PyPTO kernel 接线的功能上限 |

这些模式默认关闭，不改变原生 vLLM Ascend 路径。实现只修改 vLLM Ascend；没有修改 vLLM、
PyPTO 或 Simpler。

## 1. 测试基线与边界

- vLLM：`0.20.2+empty`
- vLLM Ascend：基于 `0.20.2rc1`
- 模型：Qwen3-14B，BF16，TP1，单 NPU
- 最大上下文：512
- KV block size：128
- 请求：两个不同短 prompt，每个生成 4 token，`temperature=0`
- eager：`enforce_eager=True`
- ACLGraph：`enforce_eager=False`
- 禁用 chunked prefill 和 speculative decode

本实验不支持量化、TP>1、MRoPE、dual-chunk attention 或同一 PyPTO context 的跨流并发。
任何超出固定结构、dtype 或布局的输入均应 fail-closed，不能静默回退原生主计算。

## 2. `attention_block` 的接口契约

`attention_block` 是后续收敛方向。它替换一层 DecoderLayer 中除 MLP 之外的完整 Attention
residual block：

```text
Input Add/RMSNorm
→ QKV projection
→ Q/K RMSNorm
→ RoPE
→ 根据 slot_mapping 写入 vLLM KV cache
→ 根据 block_tables/seq_lens/query_start_loc 读取历史 KV
→ causal paged attention
→ output projection
→ Residual Add + Post-Attention RMSNorm
→ 原生 vLLM Ascend MLP
```

PyPTO block 返回两个 `[T, 5120]` BF16 tensor：

- `mlp_input`：post-attention norm 的归一化输出，交给原生 MLP；
- `updated_residual`：Attention 输出与原 residual 相加后的未归一化状态，传给下一层。

vLLM 继续拥有 KV cache、页分配和回收、block table、请求调度及采样。PyPTO 只借用以下
Device tensor：

| 输入 | 作用 |
| --- | --- |
| `slot_mapping` | 决定本轮 K/V 写入的物理 cache row；`-1` 表示 padding，不写 cache |
| `key_cache` / `value_cache` | vLLM 分配的 paged KV cache，PyPTO 原位更新 |
| `block_tables` | 逻辑 KV 页到物理页的映射 |
| `seq_lens` | 每个 request 的当前总序列长度 |
| `query_start_loc` | 本轮拼接 token 中各 request 的区间 |

首层和普通层使用不同的固定 callable 签名。首层直接保存原始 `hidden_states` 作为 residual；
普通层执行 Add/RMSNorm。两者后半段共用相同 Attention 计算。权重由 vLLM 持有，PyPTO
不复制或堆叠 40 层权重。只有 qkv/o_proj 保持 PyPTO 可读取的 ND 布局；原生 MLP 的权重转换
和 fusion pass 保持不变。

## 3. 测试命令

### 3.1 Host UT

```bash
pytest -q \
  tests/ut/ops/test_layernorm.py::test_pypto_qwen3_mode \
  tests/ut/ops/test_layernorm.py::test_pypto_qwen3_mode_rejects_unknown_value \
  tests/ut/test_platform.py::TestNPUPlatform::test_pypto_qwen3_mode_is_part_of_config_hash_inputs \
  tests/ut/test_platform.py::TestNPUPlatform::test_pypto_qwen3_mode_rejects_invalid_or_conflicting_config \
  tests/ut/compilation/test_graph_fusion_pass_manager.py
```

覆盖模式枚举、配置 hash、冲突配置拒绝、layernorm 路由，以及 `attention_block` 不关闭原生
MLP fusion pass。当前结果为 `10 passed`。

### 3.2 单 Attention residual block

在已安装本分支、PyPTO kernel runtime 和 torch_npu 的 NPU 环境中执行：

```bash
python tools/pypto_qwen3/test_attention_residual_block.py --device 0
```

脚本与 torch_npu 原生 golden 对照并验证：

- 首层和普通层 eager；
- capture 后两次 replay，并在地址不变的情况下更新输入内容；
- 两次 replay 使用不同物理页；
- 129-token 跨越两个 128-token 物理页；
- `slot_mapping=-1` 不修改任何 KV cache 字节；
- graph reset 和进程退出。

通过标志为：

```text
QWEN3_ATTENTION_RESIDUAL_BLOCK PASS
```

### 3.3 Qwen3-14B 整网

以下脚本可用于 `off`、`partial`、`attention_block` 和 `full`。`<MODEL_DIR>` 必须指向完整的
Qwen3-14B BF16 权重目录。

```bash
VLLM_ASCEND_PYPTO_QWEN3_MODE=attention_block \
PYPTO_QWEN_EAGER=1 \
MODEL_DIR=<MODEL_DIR> \
python - <<'PY'
import os

from vllm import LLM, SamplingParams

prompts = ["Hello", "The capital of France is"]
llm = LLM(
    model=os.environ["MODEL_DIR"],
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=1,
    max_model_len=512,
    max_num_seqs=len(prompts),
    enable_chunked_prefill=False,
    gpu_memory_utilization=0.75,
    enforce_eager=bool(int(os.environ["PYPTO_QWEN_EAGER"])),
)
results = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=4))
for result in results:
    print(result.outputs[0].token_ids, result.outputs[0].text)
PY
```

将 `PYPTO_QWEN_EAGER` 改为 `0` 后重启进程执行 ACLGraph 测试。Graph 通过不能只依据 capture
完成；日志中还必须出现实际 `NPUGraph.replay()` 对应的 replay 记录。

## 4. 功能验证结果

### 4.1 原有四格实验

`partial` 和 `full` 均完成了 eager 与 ACLGraph 整网验证：

| 替换范围 | Eager | ACLGraph |
| --- | --- | --- |
| q/k RMSNorm（`partial`） | 通过 | 两个 bucket capture 并实际 replay，通过 |
| 主要模型计算（`full`） | 通过 | 两个 bucket capture 并实际 replay，通过 |

四格与原生基线的 token IDs 一致：

```text
Hello                         -> [25, 358, 614, 264]
The capital of France is      -> [12095, 13, 3555, 374]
```

`full` 图节点审计确认主计算节点使用 PyPTO kernel；vLLM 仍负责调度、KV 页所有权和采样。

### 4.2 `attention_block`

| 用例 | Eager | ACLGraph |
| --- | --- | --- |
| 单 block 原生 golden | 通过 | capture 后两次 replay，通过 |
| 两个不同物理 KV 页 | 通过 | 通过 |
| 129-token 跨页 | 通过 | 单 block eager 覆盖 |
| padding slot 不写 cache | 通过 | 单 block eager 覆盖 |
| Qwen3-14B 两请求整网 | 通过 | batch 1/2 图已捕获并实际 replay，通过 |

整网 token IDs 与上表原生、`partial` 和 `full` 结果一致。收窄 fusion 隔离范围、恢复原生 MLP
优化后已重新执行两请求 graph，结果仍一致。

## 5. 性能观察

下表是早期统一 workload 下的稳态结果：两个请求、每个生成 8 token、一次 warmup 后取三次中位数。
它用于说明实验实现的优化方向，不是性能承诺。

| 模式 | Eager 延迟 | ACLGraph 延迟 | 相对同执行模式原生 |
| --- | ---: | ---: | ---: |
| 原生 vLLM Ascend | 0.484 s | 0.237 s | 1.00x |
| `partial` | 1.399 s | 0.610 s | 慢 2.89x / 2.57x |
| `full` | 7.842 s | 7.166 s | 慢 16.21x / 30.17x |

后续优化将 `full` 的 callable 粒度由大量小算子收敛到较大的融合单元，但当前实现仍明显慢于
原生。Profile 表明 Graph 主要消除了 Host 提交开销，不能修复 PyPTO AICore kernel 和细粒度
callable 边界的设备侧成本。因此不能仅凭功能四格通过宣称具备生产替换价值。

`attention_block` 目前完成的是接口、KV cache 和 graph 生命周期验证；尚未形成可发布的稳定
性能结论。

## 6. 生命周期与所有权结论

- `profile_run` 发生在 KV cache 分配之前，明确使用原生路径做显存估算；真实请求缺少 metadata
  时 fail-closed。
- callable 在模型初始化或 capture 前 warmup 阶段准备；capture/replay 不新增 callable。
- replay 读取稳定地址中的新 tensor 内容，不在 capture 内分配 workspace、执行 H2D 或同步。
- KV cache 与 paging metadata 的生命周期由 vLLM 管理，PyPTO 不建立第二套 block manager。
- 当前依赖单 model-execution stream 的顺序执行，不支持同一 context 的跨流并发。
- shutdown 使用 PyPTO 已有 torch_npu teardown，不增加插件级 `atexit`。

## 7. 未覆盖范围

- TP>1、量化、chunked prefill、speculative decode、MRoPE、dual-chunk attention；
- 长上下文、长稳、多 Engine 或多 context 并发；
- 完整资源泄漏工具检查和异常恢复矩阵；
- `attention_block` 的稳定性能与 callable 数量 profile；
- 复杂 prompt 的模型质量评测。

当前结论只适用于本文固定的 Qwen3-14B BF16、TP1、单 NPU、512 上下文实验范围。
