import torch
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3DecoderLayer
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
    pos_embed_interpolate_native,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding


def tensor_parallel_wrap(func):
    def wrap(*args, **kwargs):
        deepstack_input_embeds = func(*args, **kwargs)
        if deepstack_input_embeds is None:
            return deepstack_input_embeds
        try:
            flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
        except (AssertionError, AttributeError, KeyError):
            flash_comm_v1_enabled = False
        if flash_comm_v1_enabled:
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            deepstack_input_embeds.tensors = {
                k: v.chunk(tp_size)[tp_rank] for k, v in deepstack_input_embeds.tensors.items()
            }
        return deepstack_input_embeds

    return wrap


def forward_with_split_qkv_rmsnorm_mrope(self, positions: torch.Tensor, hidden_states: torch.Tensor):
    if getattr(self, "_pypto_attention_only_enabled", False):
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.pypto_qwen3_attention_only(
            positions,
            hidden_states,
            self.qkv_proj.weight,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            self.o_proj.weight,
            output,
            _encode_layer_name(self.attn.layer_name),
        )
        return output
    qkv, _ = self.qkv_proj(hidden_states)
    if isinstance(self.rotary_emb, AscendMRotaryEmbedding):
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        if cos_sin.device != qkv.device:
            cos_sin = cos_sin.to(qkv.device)
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(qkv.dtype)
        q, k, v, _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            cos_sin=cos_sin,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            eps=self.q_norm.variance_epsilon,
            mrope_section=self.rotary_emb.mrope_section,
            is_interleaved=self.rotary_emb.mrope_interleaved,
            rope_dim=self.rotary_emb.rotary_dim,
        )
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


Qwen3Attention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3MoeAttention.forward = forward_with_split_qkv_rmsnorm_mrope


def _pypto_qwen3_attention_only(
    positions: torch.Tensor,
    normalized_hidden: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    # Keep this opaque op in the compiled model while KV cache allocation is
    # being profiled. Capture invokes it again after vLLM binds the real cache.
    if _EXTRA_CTX.in_profile_run:
        output.copy_(normalized_hidden)
        return

    from vllm.model_executor.layers.attention.attention import get_attention_context

    from vllm_ascend.ops import pypto_qwen3_attention

    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if attn_metadata is None or attn_layer is None or attn_layer.layer_name != layer_name:
        raise RuntimeError(f"PyPTO Qwen3 attention could not resolve layer context: {layer_name!r}")
    if not isinstance(kv_cache, (torch.Tensor, list, tuple)) or len(kv_cache) < 2:
        raise RuntimeError("PyPTO Qwen3 attention requires a bound vLLM KV cache")
    key_cache, value_cache = kv_cache[0], kv_cache[1]
    if key_cache.ndim != 4 or key_cache.shape[1:] != (128, 8, 128) or value_cache.shape != key_cache.shape:
        raise ValueError("PyPTO Qwen3 attention received an incompatible vLLM KV cache layout")
    rows = normalized_hidden.shape[0]
    slot_mapping = layer_slot_mapping if layer_slot_mapping is not None else attn_metadata.slot_mapping
    if slot_mapping is None or slot_mapping.numel() < rows:
        raise ValueError("PyPTO Qwen3 attention slot mapping is shorter than the token dimension")
    pypto_qwen3_attention.attention_only(
        positions=positions,
        normalized_hidden=normalized_hidden,
        qkv_weight=qkv_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos_sin_cache=cos_sin_cache,
        o_proj_weight=o_proj_weight,
        slot_mapping=slot_mapping[:rows],
        key_cache=key_cache.view(-1, 1024),
        value_cache=value_cache.view(-1, 1024),
        block_table=attn_metadata.block_tables,
        seq_lens=attn_metadata.seq_lens_device,
        query_start_loc=attn_metadata.query_start_loc,
        output=output,
    )


def _pypto_qwen3_attention_only_fake(
    positions: torch.Tensor,
    normalized_hidden: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="pypto_qwen3_attention_only",
    op_func=_pypto_qwen3_attention_only,
    fake_impl=_pypto_qwen3_attention_only_fake,
    mutates_args=["output"],
    dispatch_key="PrivateUse1",
)


_original_qwen3_decoder_layer_init = Qwen3DecoderLayer.__init__


def _patched_qwen3_decoder_layer_init(self, *args, **kwargs) -> None:
    _original_qwen3_decoder_layer_init(self, *args, **kwargs)
    from vllm_ascend import envs

    attention_only_enabled = envs.VLLM_ASCEND_PYPTO_QWEN3_MODE == "attention_only"
    self.self_attn._pypto_attention_only_enabled = attention_only_enabled
    if not attention_only_enabled:
        return

    config = args[0] if args else kwargs["config"]
    cache_config = kwargs.get("cache_config", args[1] if len(args) > 1 else None)
    quant_config = kwargs.get("quant_config", args[2] if len(args) > 2 else None)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if (
        config.hidden_size != 5120
        or config.num_hidden_layers != 40
        or config.num_attention_heads != 40
        or config.num_key_value_heads != 8
        or head_dim != 128
        or config.rms_norm_eps != 1e-6
        or getattr(config, "attention_bias", False)
        or not getattr(config, "is_causal", True)
        or self.self_attn.dual_chunk_attention_config is not None
        or isinstance(self.self_attn.rotary_emb, AscendMRotaryEmbedding)
        or get_tensor_model_parallel_world_size() != 1
        or get_pp_group().world_size != 1
        or quant_config is not None
    ):
        raise ValueError(
            "PyPTO Qwen3 attention-only mode requires unquantized BF16 Qwen3-14B "
            "decoder attention (40 query heads, 8 KV heads, head size 128, TP1)"
        )
    if cache_config is None or cache_config.block_size != 128 or cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("PyPTO Qwen3 attention-only mode requires a BF16 128-token KV cache block")

    from vllm.config import get_current_vllm_config

    from vllm_ascend.ops import pypto_qwen3_attention

    vllm_config = get_current_vllm_config()
    if (
        vllm_config.model_config.dtype != torch.bfloat16
        or vllm_config.model_config.max_model_len > 512
        or vllm_config.speculative_config is not None
        or vllm_config.scheduler_config.enable_chunked_prefill
    ):
        raise ValueError(
            "PyPTO Qwen3 attention-only mode requires BF16, max_model_len <= 512, "
            "and does not support speculative decoding or chunked prefill"
        )
    self.self_attn.qkv_proj._pypto_qwen3_attention_weight = True
    self.self_attn.o_proj._pypto_qwen3_attention_weight = True
    pypto_qwen3_attention.init()
    pypto_qwen3_attention.registered_attention_only_op()


Qwen3DecoderLayer.__init__ = _patched_qwen3_decoder_layer_init
Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = tensor_parallel_wrap(
    Qwen3VLForConditionalGeneration._get_deepstack_input_embeds
)


def _fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
    outputs = []
    for t, h, w in grid_thw:
        outputs.append(
            pos_embed_interpolate_native(
                self.pos_embed.weight,
                t,
                h,
                w,
                self.num_grid_per_side,
                self.spatial_merge_size,
                self.dtype,
            )
        )
    return torch.cat(outputs, dim=0)


Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate


def patch_qwen3_vl_moe_pp_layer_range():
    try:
        from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM
    except Exception:
        return

    if not hasattr(Qwen3MoeLLMForCausalLM, "start_layer"):
        Qwen3MoeLLMForCausalLM.start_layer = property(lambda self: self.model.start_layer)

    if not hasattr(Qwen3MoeLLMForCausalLM, "end_layer"):
        Qwen3MoeLLMForCausalLM.end_layer = property(lambda self: self.model.end_layer)


patch_qwen3_vl_moe_pp_layer_range()
