# SPDX-License-Identifier: Apache-2.0
"""实际Native半层的QLI只读hook；最小TopK控制与详细输入观测相互校验。"""

import torch
from dsv4_csa_single_layer import collect_state, guard_checks, restore


def capture_native_qli(layer, fixture, config):
    # Native ops/model initialization must precede these imports to avoid the
    # circular device_op initialization observed by the earlier Q probe.
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.dsv4_csa import _native_attention_half

    wrapper = layer.self_attn.dsa_attn
    assert getattr(wrapper, "_pto_csa_runtime", None) is None, "Native control must not dispatch to a PTO service"
    original = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer
    states, guards, detail = [], [], {}
    for detailed in (False, True):
        restore(fixture)
        output = torch.full_like(fixture["hidden"], float("nan"))
        captured = {}

        def record_qli(*inputs, _captured=captured, _detailed=detailed, **kwargs):
            assert not inputs, "capture the verified keyword-only Native QLI boundary"
            assert "topk" not in _captured, "expected one actual CSA QLI call"
            assert kwargs["return_value"] is False
            if _detailed:
                # Clone before the actual consumer; retaining aliases could hide
                # a later mutation. No QLI argument or scalar option is changed.
                _captured["inputs"] = {
                    name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                    for name, value in kwargs.items()
                }
            value = original(*inputs, **kwargs)
            _captured["topk"] = value[0].detach().clone()
            return value

        torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = record_qli
        try:
            with set_ascend_forward_context(
                fixture["metadata"], config, num_tokens=fixture["tokens"], num_actual_tokens=fixture["tokens"]
            ):
                _native_attention_half(wrapper, fixture["hidden"], fixture["positions"], output)
            torch.npu.synchronize()
            assert "topk" in captured, "actual Native QLI was not reached"
            states.append(collect_state(fixture, output.reshape(fixture["tokens"], -1), captured["topk"]))
            guards.append(guard_checks(fixture))
            if detailed:
                detail = {
                    name: value.cpu() if isinstance(value, torch.Tensor) else value
                    for name, value in captured["inputs"].items()
                }
        finally:
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original
    return states, guards, detail
