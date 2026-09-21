# PyPTO CSA 算子接入 vLLM 的对接说明

这条分支把 DeepSeek-V4 某一层的 `attention.forward` 整段换成 PyPTO 写的 CSA 算子,
在 vLLM 的真实推理服务路径上跑。本文给 pypto-lib 侧同事看,不假设读者了解 vLLM 这边。

内容分两部分:一是**已经确定的事实**(版本、代码位置、参数、维度关系、性能归因),
二是**仍待补的信息**(几处只能从活跃张量上读的布局细节)。每条都标了来源。

## 1. 版本

| 仓 | commit | 分支 | 链接 |
| --- | --- | --- | --- |
| vLLM | `bc150f50299199599673614f80d12a196f377655` | detached | [vllm-project/vllm@bc150f50](https://github.com/vllm-project/vllm/commit/bc150f50299199599673614f80d12a196f377655) |
| vLLM Ascend 上游基线 | `367b8e62da799870a7476ce34f5f7658589a8aad` | v0.20.2rc1 | [vllm-project/vllm-ascend@367b8e62](https://github.com/vllm-project/vllm-ascend/commit/367b8e62da799870a7476ce34f5f7658589a8aad) |
| **本分支(CSA 适配)** | `213933829aededf9e75359fa1de6e14426b81c5f` | `feat/csa-attn-cut-20260920` | 本 PR |
| pypto | `6b49cfd58de65f8a325339a30d6fa45e1973c237` | `feat/kernel-mode-integration-test` | [hw-native-sys/pypto@6b49cfd5](https://github.com/hw-native-sys/pypto/commit/6b49cfd58de65f8a325339a30d6fa45e1973c237) |
| simpler | `17ea300256e2a6db5af397ce619a7d480b595d80` | `feat/kernel-mode-integration-test`(tip) | [hw-native-sys/simpler@17ea3002](https://github.com/hw-native-sys/simpler/commit/17ea300256e2a6db5af397ce619a7d480b595d80) |
| pypto-lib | `675f027f1bf9ce8f79740f0b2ed042b0c4b2c8b3` | — | [hw-native-sys/pypto-lib@675f027f](https://github.com/hw-native-sys/pypto-lib/commit/675f027f1bf9ce8f79740f0b2ed042b0c4b2c8b3) |

本分支相对上游基线领先 18 个提交。pypto 与 simpler 必须**按 commit 取**:
pypto 的本地分支名是 `feat/kernel-mode-20260920`,但提交实际属于
`feat/kernel-mode-integration-test`;simpler 停在 detached HEAD。

**复现时的一个坑**:开发机上默认 `pip` 解析到的 pypto 和 simpler 是另外两个目录,
不是上表这套。我们靠启动脚本的 `PYTHONPATH` 覆盖。不设 `PYTHONPATH` 会拿到别的版本,
症状通常是 `Kernel ABI requires Simpler <hash>; native binding is <other>` —— 那是
pypto 与 simpler 原生 binding 的指纹不匹配,与 pypto-lib 无关。

## 2. 代码位置

| 文件 | 行数 | 作用 |
| --- | ---: | --- |
| `vllm_ascend/attention/pto_attn.py` | 1018 | 转换层:把 vLLM 的运行时状态翻译成算子的 46 个参数 |
| `vllm_ascend/ops/dsa.py` | 309 | 钩子:在 `dsa_forward` 里 `_build_kv_cache` 之后、`impl.forward` 之前分流 |
| `vllm_ascend/attention/pto_kernels/dspark/` | 11k+ | 从 pypto-lib 剥离的算子副本 |

剥离副本相对 pypto-lib 原版的真实差异是 **+222 / −54 行(34 处)**,绝大部分是
绝对 import 改相对。实质分叉只有一处:新增 `decode_csa_attn_tp1`(46 参数,只做 attention)
与其 `@pl.jit` 入口 `decode_csa_attn_tp1_test`,原来的 `decode_csa_tp1` 改成
`hc_pre_norm → attn → hc_post` 的薄包装以保持 51 参数契约。

> 用 `diff` 直接比会报「整个文件都变了」,那是行尾差异造成的假象,先 `tr -d '\r'`。

这两个入口**在 pypto-lib 里不存在**,是本分支 fork 出来的。

## 3. 运行配置

```text
vLLM        tensor_parallel_size=1  data_parallel_size=1  pipeline_parallel_size=1
            单个 device (ASCEND_RT_VISIBLE_DEVICES=15)
            dtype=bfloat16  quantization=ascend
            max_model_len=8704  max_num_batched_tokens=4096
            gpu-memory-utilization=0.60
            cudagraph_mode=FULL_DECODE_ONLY  cudagraph_capture_sizes=[1,2,4]
负载        模型 official-l3（DeepSeek-V4 截成 3 层，MTP 层已去除）
            batch=4  prompt=1024 token  max_tokens=32
算子        PTO_ATTN_TP=4   PTO_ATTN_SEQ=1
```

`cudagraph_capture_sizes` 每项必须 ≤ 算子的 `B`,超出的 descriptor 会被转换层
decline 并回落原生路径。

## 4. 注册入口与参数绑定

```python
# vllm_ascend/attention/pto_attn.py
def _registered():
    global _OP
    if _OP is None:
        from pypto.torch import init, register
        kcsa, _ = kernel()
        init()
        _OP = register(kcsa.decode_csa_attn_tp1_test, "pypto_csa::attention_csa")
    return _OP
```

绑定顺序(`ARG_ORDER`,共 46 项):

```python
ARG_ORDER = (
    "x_normed",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "cmp_freqs_cos", "cmp_freqs_sin",
    "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
    "compress_state", "compress_state_block_table",
    "idx_wq_b", "idx_wq_b_scale", "weights_proj", "hadamard_idx",
    "inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w",
    "inner_compress_state", "inner_compress_state_block_table",
    "kv_cache", "cmp_kv", "cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table",
    "ori_slot_mapping", "window_swa_indices",
    "cmp_slot_mapping", "idx_slot_mapping",
    "state_slot_mapping", "inner_state_slot_mapping",
    "position_ids", "kv_seq_lens", "attn_sink",
    "wo_a", "wo_b", "wo_b_scale",
    "attn_out",
)
```

参数值在 `build_args()` 里组装,钩子在 `ops/dsa.py::dsa_forward` 调用
`pto_attn.substitute()`;返回 `True` 表示已接管,不再走 `impl.forward`。

## 5. 维度关系

看到 `B=16` 与服务 `bs=4` 并存不必困惑,两者不是一回事。

`decode_csa.py:132` 是 `B = DECODE_BATCH // TP_SIZE`,`DECODE_BATCH=64`,
所以 `PTO_ATTN_TP=4` 得到 `B=16`。这是**算子实例的批容量**,不是实际批。
4 个真实请求摊进 `T = B × S` 的矩形,padding 车道置 `-1`。

**注意 `TP_SIZE=4` 在本部署里是容量选择器,不是四路并行。** 我们在单 device 上跑,
注册的入口用全局权重常量(`O_GROUPS=8`),没有跨 rank 通信;那些
`pld.DistributedTensor` 参数属于别的入口。用 `--tp 4` 的原因是 `--tp 1` 时
`T=512` 会让 ring heap 分配失败(`FATAL: Task Allocator Deadlock - Heap Exhausted`),
而 kernel 模式没有 ring 尺寸的配置入口。代价是这个实例最多接 16 个并发请求。

本实例化下的编译期常量:

```text
B=16  S=8  T=T_PAD=128  BLOCK_SIZE=32  HEAD_DIM=512  IDX_HEAD_DIM=128
WIN=128  COMPRESS_RATIO=4
MAIN_STATE_BLOCK_SIZE=2   MAIN_STATE_MAX_BLOCKS=8   MAIN_STATE_DIM=2048
MAIN_STATE_STORAGE_LEN=16
INNER_STATE_BLOCK_SIZE=2  INNER_STATE_MAX_BLOCKS=8  INNER_STATE_DIM=512
CMP_MAX_BLOCKS=IDX_MAX_BLOCKS=8192
```

## 6. 入参格式与 vLLM 实际布局的差异

vLLM 用 `as_strided` 在一块 padded 缓冲上切出多份逻辑 cache,所以其中四份是
非连续视图;而 PyPTO 绑定层拒收非连续张量(`pypto/torch/interop.py:141` 直接
`raise ValueError("... requires a contiguous strided tensor; no copy is made")`)。

| 错配 | 算子这边 | vLLM 那边 | 逼出的代价 |
| --- | --- | --- | --- |
| 压缩器 state 的组织 | 每请求 16 行私有环 + 自带块表(页 2 行 × 8 页) | 按自己的分配器分页,页 8 行 | 每步现建环 + 写回 |
| indexer key / scale | 两个独立张量,scale 要 FP32 | 同一个 16640 字节页,key 占前 128 行、scale 在尾部 | key 非连续 + 类型转换 |
| 页大小 | 32 行 | 128 行 | **已零开销解决**(4 倍,块表列号换算) |
| 槽号空间 | 按算子自己的页编号 | 按 vLLM 的 `(block, intra)` | 每步翻译 |

页大小那条说明这类差异**可以被消除**:只要是整数倍关系,换算块表即可,不需要搬数据。
滑窗 KV 与压缩 KV 走的就是这条路,实测搬运开销为 0。

## 7. 性能现状

aclgraph `FULL_DECODE_ONLY` 下 8 个 decode 步的实测,每步设备耗时:

| 项 | us/步 | 说明 |
| --- | ---: | --- |
| 搬运(去 stride) | 20,811 | 转换层 |
| 索引换算 | 2,070 | 转换层,183 次算子启动/步 |
| **CSA 算子本体** | **976** | AICPU 派发段与 AICore 执行段的并集 |
| 图内其余模型算子 | 3,031 | 两轮都有 |
| 图外 eager 通道 | 1,507 | metadata |
| 合计 | 28,395 | 原生对照 4,737 |

算子耗时**不能把 AICPU 段与 AICore 段相加**:后者在全部 8 步里都严格嵌套于前者,
相加会把 976 us 说成 1924 us。

**转换层占了 80%,算子本体占 3.4%。**

搬运那 20.8 ms 技术上可以在转换层内修掉(改成不触发去 stride),修完约 7.2 ms/步。
但即使做完,索引换算里仍有约 1.0~1.5 ms/步**消不掉** —— 两套寻址约定之间必须有人翻译,
且必须做在图内,因为图重放时 Python 不执行,翻译结果没法预先算好塞进去。

**我们决定不做这部分优化。** 两个原因:对齐版本出来之后这些代码全都要删掉,现在投入是废功;
而且做完离原生仍有距离。现有转换层功能上够用(能跑通、能进图),性能等对齐版本解决。

希望 lib 侧提供的是一个**入参与 vLLM 实际布局对齐**的 CSA 算子版本,
让转换层只做参数传递 —— 不搬移数据、不改数据格式、不做索引换算。
那样每步可以从 28.4 ms 直接降到约 5.6 ms,对原生的 4.7 ms 约 1.19 倍,
剩下的差距只剩算子本身的计算时间。

## 8. 布局与语义的详细答复

六份 cache 的布局、五类 slot mapping 的语义、维度关系等详细问答见
[pto-csa-layout-qa.md](pto-csa-layout-qa.md)。三条最要紧的结论:

- **state block table 记的是整段历史的绝对逻辑页号**,`[B, 1088]` INT32,
  列 j 对应 position ∈ [8j, 8j+8)。ring 语义只存在于算子这一侧,vLLM 侧不 wrap。
- **inner state 与 indexer key 共享同一块 allocation**,inner state 页尾那
  256 字节是 indexer scale,**不是 padding,算子绝不能往那里写**。
- **indexer 的三份 cache 只存在于三层里的一层**(`ops/dsa.py:284` 按
  `compress_ratio == 4` 门控),所有 indexer 的实测只对 layer 2 成立。

仍待补的是几处只能从活跃张量上读的:各张量的 `data_ptr` 关系、各块表的实际示例值,
以及一份覆盖边界的 fixture(8 行逻辑页边界、16 行环回绕、128 行 KV 页边界、
indexer 页切换、inactive token、多请求不同物理页)。需要再跑一轮带数值转储的运行。

## 9. 复现

```bash
export PTO_ATTN_REPLACE=1 PTO_ATTN_SEQ=1 PTO_ATTN_TP=4 PYPTO_CACHE=1
export SERVE_EXTRA='--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
```

`SERVE_EXTRA` 的 JSON **不能有空格**:启动脚本按空格分词。

判据不要用客户端的 `output_token_fingerprints`,它是对每请求哈希取
`sorted({...})`,丢掉了请求身份,同配置多轮会给出不同结果。可用的判据是三条同时成立:
捕获进度跑满且服务健康;每个 capture descriptor 有两次替换事件、第二次带
`capturing=True`;重放期每步延迟在 PTO 量级而非原生量级。再加 profile 里
`simpler_aicpu_kernel_exec` 每步出现一次作为直接证据。
