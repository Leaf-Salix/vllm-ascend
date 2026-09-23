"""Keep the native dsa_forward caller-output and profiling contracts."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops import dsa


@pytest.fixture
def call(monkeypatch):
    impl = SimpleNamespace(forward=Mock())
    layer = SimpleNamespace(prefix="layer", dsa_attn=SimpleNamespace(impl=impl, layer_name="layer.attn"))
    metadata = [object()]
    context = SimpleNamespace(no_compile_layers={"layer": layer}, attn_metadata={"layer.attn": metadata[0]})
    cache = tuple(object() for _ in range(6))
    monkeypatch.setattr(dsa, "get_forward_context", lambda: context)
    monkeypatch.setattr(dsa, "_build_kv_cache", Mock(return_value=cache))
    monkeypatch.setenv("VLLM_ASCEND_PYPTO_DSV4_CSA", "1")
    return SimpleNamespace(
        impl=impl,
        layer=layer,
        metadata=metadata,
        context=context,
        cache=cache,
        hidden=torch.ones(1, 4),
        output=torch.zeros(1, 4),
    )


def assert_native(call, gather=False):
    call.impl.forward.assert_called_once_with("layer.attn", call.hidden, call.cache, call.metadata, gather, call.output)


def test_disabled_preserves_native_call(call, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_PYPTO_DSV4_CSA", "0")
    with patch("vllm_ascend.attention.pto_attn.substitute") as substitute:
        assert dsa.dsa_forward(call.hidden, False, call.output, "layer") is None
    substitute.assert_not_called()
    assert_native(call)


def test_success_writes_caller_output_once(call):
    def write_output(layer, hidden, cache, metadata, output):
        assert output is call.output
        output.fill_(3)
        return True

    with patch("vllm_ascend.attention.pto_attn.substitute", side_effect=write_output) as substitute:
        assert dsa.dsa_forward(call.hidden, False, call.output, "layer") is None
    substitute.assert_called_once_with(call.layer, call.hidden, call.cache, call.metadata, call.output)
    call.impl.forward.assert_not_called()
    assert torch.equal(call.output, torch.full_like(call.output, 3))


def test_declined_call_uses_native_args(call):
    with patch("vllm_ascend.attention.pto_attn.substitute", return_value=False):
        dsa.dsa_forward(call.hidden, False, call.output, "layer")
    assert_native(call)


def test_gather_call_stays_native(call):
    with patch("vllm_ascend.attention.pto_attn.substitute") as substitute:
        dsa.dsa_forward(call.hidden, True, call.output, "layer")
    substitute.assert_not_called()
    assert_native(call, gather=True)


def test_profiling_preserves_native_collective_path(call):
    call.context.attn_metadata = None
    with patch("vllm_ascend.attention.pto_attn.substitute") as substitute:
        dsa.dsa_forward(call.hidden, True, call.output, "layer")
    substitute.assert_not_called()
    dsa._build_kv_cache.assert_not_called()
    call.impl.forward.assert_called_once_with("layer.attn", call.hidden, None, None, True, call.output)


def test_kernel_failure_does_not_run_native_after_mutation(call):
    with (
        patch("vllm_ascend.attention.pto_attn.substitute", side_effect=RuntimeError("launch failed")),
        pytest.raises(RuntimeError, match="launch failed"),
    ):
        dsa.dsa_forward(call.hidden, False, call.output, "layer")
    call.impl.forward.assert_not_called()
