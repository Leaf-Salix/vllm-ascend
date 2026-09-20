import os

import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3ForCausalLM, Qwen3MLP
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
    pos_embed_interpolate_native,
)

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding

_original_qwen3_load_weights = Qwen3ForCausalLM.load_weights
_original_qwen3_forward = Qwen3ForCausalLM.forward


def _profile_limited_qwen3_load_weights(self, weights):
    loaded = _original_qwen3_load_weights(self, weights)
    requested = int(os.getenv("VLLM_ASCEND_PYPTO_QWEN3_PROFILE_LAYERS", "0"))
    if requested:
        local_layers = self.model.end_layer - self.model.start_layer
        if requested < 1 or requested > local_layers:
            raise ValueError("VLLM_ASCEND_PYPTO_QWEN3_PROFILE_LAYERS must fit in the local pipeline stage")
        self.model.end_layer = self.model.start_layer + requested
    return loaded


Qwen3ForCausalLM.load_weights = _profile_limited_qwen3_load_weights


def _pypto_swimlane_tensor_signature(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))


def _profiled_qwen3_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors=None,
    inputs_embeds: torch.Tensor | None = None,
):
    from vllm_ascend import envs

    if envs.VLLM_ASCEND_PYPTO_QWEN3_MODE == "off" or not envs.VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_LEVEL:
        return _original_qwen3_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)

    from vllm_ascend.ops.pypto_swimlane import eager_model_swimlane

    signature = (
        _pypto_swimlane_tensor_signature(input_ids),
        _pypto_swimlane_tensor_signature(positions),
        _pypto_swimlane_tensor_signature(inputs_embeds),
    )
    with eager_model_swimlane(
        "qwen3_model_forward",
        signature,
        graph_enabled=not self.vllm_config.model_config.enforce_eager,
    ):
        return _original_qwen3_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)


Qwen3ForCausalLM.forward = _profiled_qwen3_forward


def _prepare_pypto_qkv_frontend(self) -> None:
    from vllm_ascend import envs

    self._pypto_qkv_norm_rope_op = None
    if envs.VLLM_ASCEND_PYPTO_QWEN3_MODE != "full" or isinstance(
        self.rotary_emb,
        AscendMRotaryEmbedding,
    ):
        return

    from vllm_ascend.ops import pypto_qwen3_full

    pypto_qwen3_full.init()
    self._pypto_qkv_norm_rope_op = pypto_qwen3_full.registered_ops()["qkv_norm_rope"]


_original_qwen3_attention_init = Qwen3Attention.__init__


def _patched_qwen3_attention_init(self, *args, **kwargs) -> None:
    _original_qwen3_attention_init(self, *args, **kwargs)
    _prepare_pypto_qkv_frontend(self)


_original_qwen3_moe_attention_init = Qwen3MoeAttention.__init__


def _patched_qwen3_moe_attention_init(self, *args, **kwargs) -> None:
    _original_qwen3_moe_attention_init(self, *args, **kwargs)
    _prepare_pypto_qkv_frontend(self)


Qwen3Attention.__init__ = _patched_qwen3_attention_init
Qwen3MoeAttention.__init__ = _patched_qwen3_moe_attention_init


_original_qwen3_mlp_init = Qwen3MLP.__init__
_original_qwen3_mlp_forward = Qwen3MLP.forward


def _patched_qwen3_mlp_init(self, *args, **kwargs) -> None:
    _original_qwen3_mlp_init(self, *args, **kwargs)
    from vllm_ascend import envs

    self._pypto_mlp_op = None
    if envs.VLLM_ASCEND_PYPTO_QWEN3_MODE == "full":
        from vllm_ascend.ops import pypto_qwen3_full

        pypto_qwen3_full.init()
        self._pypto_mlp_op = pypto_qwen3_full.registered_ops()["mlp"]


def _patched_qwen3_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    if self._pypto_mlp_op is None:
        return _original_qwen3_mlp_forward(self, x)
    x_2d = x.view(-1, x.shape[-1])
    out = torch.empty_like(x_2d)
    out = self._pypto_mlp_op(
        x_2d,
        self.gate_up_proj.weight,
        self.down_proj.weight,
        out,
    )
    return out.view(x.shape)


Qwen3MLP.__init__ = _patched_qwen3_mlp_init
Qwen3MLP.forward = _patched_qwen3_mlp_forward


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
    from vllm_ascend import envs

    if envs.VLLM_ASCEND_PYPTO_QWEN3_MODE == "full" and not isinstance(
        self.rotary_emb,
        AscendMRotaryEmbedding,
    ):
        if self._pypto_qkv_norm_rope_op is None:
            raise RuntimeError("PyPTO Qwen3 fused QKV front-end was not prepared")
        rows = hidden_states.shape[0]
        q = torch.empty((rows, self.q_size), dtype=hidden_states.dtype, device=hidden_states.device)
        k = torch.empty((rows, self.kv_size), dtype=hidden_states.dtype, device=hidden_states.device)
        v = torch.empty_like(k)
        q, k, v = self._pypto_qkv_norm_rope_op(
            hidden_states,
            self.qkv_proj.weight,
            self.q_norm.weight.view(1, self.head_dim),
            self.k_norm.weight.view(1, self.head_dim),
            positions,
            self.rotary_emb.cos_sin_cache,
            q,
            k,
            v,
        )
    else:
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
    elif envs.VLLM_ASCEND_PYPTO_QWEN3_MODE != "full":
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
