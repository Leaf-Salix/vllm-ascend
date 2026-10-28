# SPDX-License-Identifier: Apache-2.0
"""实际Native半层的QLI只读hook；最小TopK控制与详细输入观测相互校验。"""

import sys
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import torch
from dsv4_csa_single_layer import collect_state, guard_checks, restore


@contextmanager
def query_stage_capture(impl, tokens):
    """只读观测真实Native query的BF16边界，退出时恢复全部原函数。"""
    module = sys.modules[type(impl).__module__]
    original_rope = torch.ops._C_ascend.inplace_partial_rotary_mul
    original_linear, original_scale = module.hadamard_linear, module.hadamard_scale
    stages = {}

    def query_tensor(value):
        return isinstance(value, torch.Tensor) and value.shape[-1] == 128 and value.numel() == tokens * 64 * 128

    def save_stage(name, value):
        assert name not in stages, "expected one actual Native query stage"
        stages[name] = value.detach().clone()

    def record_rope(value, *args, **kwargs):
        is_query = query_tensor(value)
        if is_query:
            save_stage("projection", value.reshape(tokens, 64, 128))
        result = original_rope(value, *args, **kwargs)
        if is_query:
            save_stage("rope", value.reshape(tokens, 64, 128))
        return result

    def record_linear(value, *args, **kwargs):
        result = original_linear(value, *args, **kwargs)
        if query_tensor(value):
            save_stage("hadamard_linear", result[0])
        return result

    def record_scale(value, *args, **kwargs):
        result = original_scale(value, *args, **kwargs)
        if query_tensor(value):
            save_stage("scaled", result)
        return result

    torch.ops._C_ascend.inplace_partial_rotary_mul = record_rope
    module.hadamard_linear, module.hadamard_scale = record_linear, record_scale
    try:
        yield stages
    finally:
        torch.ops._C_ascend.inplace_partial_rotary_mul = original_rope
        module.hadamard_linear, module.hadamard_scale = original_linear, original_scale


@contextmanager
def weight_stage_capture(impl):
    """观测实际权重投影及prepare输入，保留静态方法描述符并自动恢复。"""
    module = sys.modules[type(impl).__module__]
    operator = module.DeviceOperator
    original_prepare = operator.prepare_dsa_indexer_weights
    stages = {}

    def save(name, value):
        assert name not in stages
        assert isinstance(value, torch.Tensor)
        stages[name] = value.detach().clone()

    def record_projection(_layer, inputs, result):
        assert len(inputs) == 1
        save("input", inputs[0])
        save("projection", result)
        # None preserves the actual forward result object.

    def record_prepare(value, *args, **kwargs):
        save("scaled", value)
        result = original_prepare(value, *args, **kwargs)
        save("consumer", result)
        return result

    handle = impl.weights_proj.register_forward_hook(record_projection)
    try:
        with patch.object(operator, "prepare_dsa_indexer_weights", new=staticmethod(record_prepare)):
            yield stages
    finally:
        handle.remove()


@contextmanager
def output_stage_capture(impl):
    """只读观测实际O-proj入口及输出，复用原绑定方法和原返回对象。"""
    original = impl._forward_o_proj
    original_decode = impl._forward_decode
    operator = sys.modules[type(impl).__module__].DeviceOperator
    original_factory = operator.get_dsa_sparse_attn_op
    stages = {}

    def record(heads, output, *args, **kwargs):
        assert "heads" not in stages, "expected one actual Native O-proj call"
        stages["heads"] = heads.detach().clone()
        result = original(heads, output, *args, **kwargs)
        stages["projection"] = output.detach().clone()
        return result

    def record_decode(*args, **kwargs):
        assert "attention" not in stages, "expected one actual Native decode call"
        result = original_decode(*args, **kwargs)
        stages["attention"] = result.detach().clone()
        return result

    def record_wo_b_input(_layer, inputs):
        assert len(inputs) == 1 and "wo_a" not in stages, "expected one actual Native WO-B input"
        stages["wo_a"] = inputs[0].detach().clone()
        # A None pre-hook result preserves the original arguments.

    def sparse_factory():
        original_sparse = original_factory()

        def record_sparse(query, **kwargs):
            assert "sparse_inputs" not in stages, "expected one actual Native sparse consumer"
            stages["sparse_inputs"] = {
                name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for name, value in {"query": query, **kwargs}.items()
            }
            result = original_sparse(query, **kwargs)
            stages["sparse_output"] = result[0].detach().clone()
            return result

        return record_sparse

    handle = impl.wo_b.register_forward_pre_hook(record_wo_b_input)
    try:
        with (
            patch.object(operator, "get_dsa_sparse_attn_op", new=staticmethod(sparse_factory)),
            patch.object(impl, "_forward_o_proj", new=record),
            patch.object(impl, "_forward_decode", new=record_decode),
        ):
            yield stages
    finally:
        handle.remove()


def capture_native_qli(layer, fixture, config):
    # Native ops/model initialization must precede these imports to avoid the
    # circular device_op initialization observed by the earlier Q probe.
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.dsv4_csa import _native_attention_half

    wrapper = layer.self_attn.dsa_attn
    assert getattr(wrapper, "_pto_csa_runtime", None) is None, "Native control must not dispatch to a PTO service"
    original = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer
    states, guards, detail, detailed_stages, detailed_weights, detailed_output = [], [], {}, {}, {}, {}
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
            impl = wrapper.dsa_attn.impl
            stage_context = query_stage_capture(impl, fixture["tokens"]) if detailed else nullcontext({})
            weight_context = weight_stage_capture(impl) if detailed else nullcontext({})
            output_context = output_stage_capture(impl) if detailed else nullcontext({})
            with (
                stage_context as stages,
                weight_context as weights,
                output_context as output_stages,
                set_ascend_forward_context(
                    fixture["metadata"], config, num_tokens=fixture["tokens"], num_actual_tokens=fixture["tokens"]
                ),
            ):
                _native_attention_half(wrapper, fixture["hidden"], fixture["positions"], output)
            torch.npu.synchronize()
            assert "topk" in captured, "actual Native QLI was not reached"
            states.append(collect_state(fixture, output.reshape(fixture["tokens"], -1), captured["topk"]))
            guards.append(guard_checks(fixture))
            if detailed:
                assert set(stages) == {"projection", "rope", "hadamard_linear", "scaled"}
                assert all(value.dtype == torch.bfloat16 for value in stages.values())
                detailed_stages = {name: value.cpu() for name, value in stages.items()}
                assert set(weights) == {"input", "projection", "scaled", "consumer"}
                assert all(weights[name].dtype == torch.bfloat16 for name in ("input", "projection", "scaled"))
                assert weights["consumer"].dtype == torch.float16
                detailed_weights = {name: value.cpu() for name, value in weights.items()}
                assert set(output_stages) == {
                    "attention",
                    "heads",
                    "projection",
                    "sparse_inputs",
                    "sparse_output",
                    "wo_a",
                }
                output_names = ("attention", "heads", "projection", "sparse_output", "wo_a")
                assert all(output_stages[name].dtype == torch.bfloat16 for name in output_names)
                detailed_output = {name: output_stages[name].cpu() for name in output_names}
                detailed_output["sparse_inputs"] = {
                    name: value.cpu() if isinstance(value, torch.Tensor) else value
                    for name, value in output_stages["sparse_inputs"].items()
                }
                detail = {
                    name: value.cpu() if isinstance(value, torch.Tensor) else value
                    for name, value in captured["inputs"].items()
                }
        finally:
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original
    return states, guards, detail, detailed_stages, detailed_weights, detailed_output
