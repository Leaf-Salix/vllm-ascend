# SPDX-License-Identifier: Apache-2.0
"""Opaque service forward boundary; Native remains the fallback implementation."""

import torch
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


def _native_attention_half(wrapper, hidden_states, positions, output) -> None:
    """按 release 原样走一遍 attention 半边：mHC pre + input_layernorm + attention + mHC post。

    PTO 现在接管的是整个半边而不只是 attention，所以回退也必须覆盖同样的范围，
    否则 mHC 的门控就没人算了。这里照抄 `DeepseekV4DecoderLayer.forward` 的前半段，
    不改变任何 Native 的调用顺序与参数。
    """
    layer = wrapper._pto_csa_layer[0]  # tuple 包装，见 install_csa_forward 的说明
    residual = hidden_states.clone()
    mixed, post, comb = layer.hc_pre(
        hidden_states, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
    )
    normed = layer.input_layernorm(mixed)
    attn = layer.self_attn(positions=positions, hidden_states=normed, llama_4_scaling=None)
    output.copy_(layer.hc_post(attn, residual, post, comb))


def dsv4_csa_forward(hidden_states: torch.Tensor, positions: torch.Tensor,
                     output: torch.Tensor, layer_name: str) -> None:
    from vllm_ascend.ops.dsa import _build_kv_cache

    context = get_forward_context()
    wrapper = context.no_compile_layers[layer_name]
    runtime = getattr(wrapper, "_pto_csa_runtime", None)
    if runtime is None or not runtime.eligible(context, hidden_states, positions):
        _native_attention_half(wrapper, hidden_states, positions, output)
        return
    runtime(context, hidden_states, positions, output, _build_kv_cache(wrapper, context))


def dsv4_csa_forward_fake(hidden_states: torch.Tensor, positions: torch.Tensor,
                          output: torch.Tensor, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name="dsv4_csa_forward", op_func=dsv4_csa_forward,
    mutates_args=["output"], fake_impl=dsv4_csa_forward_fake, dispatch_key="PrivateUse1",
)
