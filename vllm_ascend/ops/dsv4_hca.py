# SPDX-License-Identifier: Apache-2.0
"""HCA 的单次服务调用边界，未支持的场景交回 Native。"""

import torch
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


def native_attention_half(wrapper, hidden, positions, output):
    layer = wrapper._pto_hca_layer[0]
    residual = hidden.clone()
    mixed, post, comb = layer.hc_pre(hidden, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
    normalized = layer.input_layernorm(mixed)
    attention = layer.self_attn(positions=positions, hidden_states=normalized, llama_4_scaling=None)
    output.copy_(layer.hc_post(attention, residual, post, comb))


def dsv4_hca_forward(hidden: torch.Tensor, positions: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    from vllm_ascend.ops.dsa import _build_kv_cache

    context = get_forward_context()
    wrapper = context.no_compile_layers[layer_name]
    runtime = getattr(wrapper, "_pto_hca_runtime", None)
    record_route = getattr(type(runtime), "record_graph_route", None)
    if record_route is not None:
        record_route(runtime, context, hidden, positions)
    empty_call = getattr(type(runtime), "handle_empty_call", None)
    if empty_call is not None and empty_call(runtime, context, hidden, positions, output):
        return
    if runtime is None or not runtime.eligible(context, hidden, positions):
        native_attention_half(wrapper, hidden, positions, output)
        return
    runtime(context, hidden, positions, output, _build_kv_cache(wrapper, context))


def dsv4_hca_forward_fake(hidden: torch.Tensor, positions: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name="dsv4_hca_forward", op_func=dsv4_hca_forward,
    mutates_args=["output"], fake_impl=dsv4_hca_forward_fake, dispatch_key="PrivateUse1",
)
