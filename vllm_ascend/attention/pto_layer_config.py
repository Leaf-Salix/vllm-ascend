# SPDX-License-Identifier: Apache-2.0
"""Full BSH CSA configuration contract, pinned to the reproduced reference."""

QUERY_TOKENS = 6


def validate_configuration(config):
    import torch
    from vllm.config import CUDAGraphMode

    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.utils import enable_dsa_cp, oproj_tp_enable

    hf = config.model_config.hf_config
    expected = {
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "sliding_window": 128,
        "rms_norm_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
    }
    mismatches = {
        key: (getattr(hf, key, None), value) for key, value in expected.items() if getattr(hf, key, None) != value
    }
    if hf.model_type != "deepseek_v4" or mismatches:
        raise ValueError(f"PTO CSA requires the DeepSeek V4 Flash configuration: {mismatches}")
    if config.model_config.dtype != torch.bfloat16 or config.model_config.quantization != "ascend":
        raise ValueError("PTO CSA requires Native ModelSlim W8A8 loading with BF16 activations")
    parallel = config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("PTO CSA currently requires TP=1 and PP=1")
    if enable_dsa_cp() or oproj_tp_enable():
        raise ValueError("PTO CSA does not support DSA context parallelism or O-projection TP")
    # Release Native selects INT8 indexer storage directly on A3; its upstream
    # AttentionConfig does not accept the main-branch indexer_kv_dtype='int8'.
    # NativeCSACall validates the actual key/scale dtype and shared storage.
    if config.cache_config.block_size != 32:
        raise ValueError("PTO CSA requires 32-token cache blocks")
    # mode=1 使量化权重采用 NZ，mode=2 再启用 BF16 NZ。四张根矩阵方向与 Native
    # 加载后保持一致，prepare_weights 直接借用已匹配的格式 29 存储；pl.NZ 声明相同字节。
    # Native 因当前 CANN 算子能力保留 ND 的目标权重，只在初始化时转换一次所需格式。
    #
    # enable_kv_nz 仍然拒绝：它改的是 KV cache 的页布局，而 PTO 的 cache 读取路径
    # （尤其是 indexer 的整页搬运）是按 Native 的 ND 页布局写死的，不是换个格式就行。
    if get_ascend_config().weight_nz_mode not in (0, 1, 2) or get_ascend_config().enable_kv_nz:
        raise ValueError("PTO CSA requires weight_nz_mode in (0, 1, 2) and enable_kv_nz=false")
    from .pto_kernels.dspark_layer.nz_mode import validate_weight_nz_mode

    validate_weight_nz_mode(get_ascend_config().weight_nz_mode)
    if getattr(hf, "use_index_cache", False) or config.lora_config is not None:
        raise ValueError("PTO CSA does not support IndexCache reuse or LoRA")
    spec = config.speculative_config
    if spec is None or spec.method != "dspark" or spec.num_speculative_tokens != QUERY_TOKENS - 1:
        raise ValueError("PTO CSA requires target DSpark decoding with five speculative tokens")
    if config.use_v2_model_runner:
        raise ValueError("PTO CSA service graph dispatch currently requires Model Runner V1")
    graph_mode = config.compilation_config.cudagraph_mode
    if graph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY):
        raise ValueError("PTO CSA supports eager or FULL_DECODE_ONLY graph mode")
    if graph_mode != CUDAGraphMode.NONE:
        if config.compilation_config.cudagraph_num_of_warmups < 1:
            raise ValueError("PTO CSA requires a warmup call before each graph capture")
