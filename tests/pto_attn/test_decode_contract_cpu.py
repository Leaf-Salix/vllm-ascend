"""Guard the serving boundary before the CSA kernel can mutate native caches."""

import builtins
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention import pto_attn


def decode_call():
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        speculative_config=None,
        kv_transfer_config=None,
    )
    impl = SimpleNamespace(
        vllm_config=config,
        compress_ratio=4,
        skip_topk=False,
        use_index_cache=False,
    )
    metadata = [
        SimpleNamespace(decode=object(), num_prefills=0, num_decodes=4, num_decode_tokens=4, num_actual_tokens=4)
        for _ in range(5)
    ]
    return impl, metadata, (None,) * 6


def test_single_token_decode_is_supported():
    assert pto_attn.supports_decode(*decode_call())


@pytest.mark.parametrize("group", range(5))
def test_mixed_prefill_decode_declines_whole_call(group):
    impl, metadata, cache = decode_call()
    metadata[group].num_prefills = 1
    assert not pto_attn.supports_decode(impl, metadata, cache)


@pytest.mark.parametrize("flag", ["skip_topk", "use_index_cache"])
def test_index_cache_reuse_declines(flag):
    impl, metadata, cache = decode_call()
    setattr(impl, flag, True)
    assert not pto_attn.supports_decode(impl, metadata, cache)


@pytest.mark.parametrize(
    "axis", ["tensor_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size"]
)
def test_distributed_attention_declines(axis):
    impl, metadata, cache = decode_call()
    setattr(impl.vllm_config.parallel_config, axis, 2)
    assert not pto_attn.supports_decode(impl, metadata, cache)


@pytest.mark.parametrize("config_name", ["speculative_config", "kv_transfer_config"])
def test_unvalidated_lifecycle_declines(config_name):
    impl, metadata, cache = decode_call()
    setattr(impl.vllm_config, config_name, object())
    assert not pto_attn.supports_decode(impl, metadata, cache)


@pytest.mark.parametrize("tokens", [0, 3, 8])
def test_nonuniform_or_multitoken_decode_declines(tokens):
    impl, metadata, cache = decode_call()
    metadata[0].num_decode_tokens = tokens
    metadata[0].num_actual_tokens = tokens
    assert not pto_attn.supports_decode(impl, metadata, cache)


@pytest.mark.parametrize("groups,caches", [(4, 6), (5, 7), (0, 0)])
def test_other_cache_abis_decline(groups, caches):
    impl, metadata, _ = decode_call()
    assert not pto_attn.supports_decode(impl, metadata[:groups], (None,) * caches)


def test_rope_uses_0251_full_cache_without_copy():
    import sys
    import types

    rope = types.ModuleType("vllm_ascend.ops.rope_dsv4")
    table = torch.randn(32, 1, 1, 64, dtype=torch.float32)
    rope._ROPE_STATE = SimpleNamespace(
        layer_info={"layer": ("config", ["default"])},
        full_rope_cache={"config": (table, table)},
    )
    with patch.dict(sys.modules, {rope.__name__: rope}):
        cos, sin = pto_attn._native_rope_tables("layer")
    assert cos.shape == (32, 64)
    assert cos.data_ptr() == sin.data_ptr() == table.data_ptr()


@pytest.mark.parametrize("fails", [False, True])
def test_kernel_import_pins_tp1_and_restores_argv(monkeypatch, fails):
    original = ["vllm", "--tp", "4", "--port", "8000"]
    monkeypatch.setattr(sys, "argv", original)
    real_import = builtins.__import__
    imported = SimpleNamespace(config=object(), decode_csa=object())

    def import_kernel(name, *args, **kwargs):
        if name == "pto_kernels.dspark":
            assert sys.argv == ["vllm", "--tp", "1"]
            if fails:
                raise ImportError("kernel unavailable")
            return imported
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_kernel)
    if fails:
        with pytest.raises(ImportError, match="kernel unavailable"):
            pto_attn._import_kernel()
    else:
        assert pto_attn._import_kernel() == (imported.decode_csa, imported.config)
    assert sys.argv is original


def test_kernel_layout_exception_propagates_after_launch(monkeypatch):
    impl, metadata, cache = decode_call()
    metadata[0].decode = SimpleNamespace(input_positions=torch.arange(4))
    layer = SimpleNamespace(dsa_attn=SimpleNamespace(impl=impl, layer_name="layer"))
    utils = ModuleType("vllm_ascend.utils")
    utils.AscendDeviceType = SimpleNamespace(A3="A3")
    utils.get_ascend_device_type = lambda: "A3"
    utils.enable_dsa_cp = utils.olora_tp_enable = utils.oproj_tp_enable = lambda: False
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setattr(pto_attn, "kernel", lambda: (SimpleNamespace(S=6, B=64), None))
    monkeypatch.setattr(pto_attn, "capture_active", lambda: True)
    monkeypatch.setattr(pto_attn, "_tally", lambda *args: None)
    monkeypatch.setattr(pto_attn, "build_args", lambda *args, **kwargs: ((), (None, None, 4)))

    def launched_kernel():
        raise pto_attn.NativeLayoutError("failure after launch")

    monkeypatch.setattr(pto_attn, "_registered", lambda: launched_kernel)
    with pytest.raises(pto_attn.NativeLayoutError, match="failure after launch"):
        pto_attn.substitute(layer, torch.zeros(4, 4), cache, metadata, torch.zeros(4, 4))
