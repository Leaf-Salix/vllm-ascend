# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for full-layer dispatch and native storage ownership.

Only framework/NPU services are stubbed. Execute the actual dispatch/binding
functions and the actual decoder forward body with CPU tensors.
"""

import ast
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

ATTENTION = Path(__file__).resolve().parents[2] / "vllm_ascend/attention"


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_compressor_keeps_native_nd_weights(compress_ratio):
    tree = ast.parse((ATTENTION.parent / "models/deepseek_v4.py").read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Compressor")
    definition.body = [
        node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    namespace = {
        "nn": torch.nn,
        "torch": torch,
        "ReplicatedLinear": lambda *args, **kwargs: NS(),
        "get_ascend_device_type": lambda: "A3",
        "AscendDeviceType": NS(A5="A5"),
        "RMSNorm": lambda *args, **kwargs: NS(),
        "AscendCompressorStateCache": lambda **kwargs: NS(**kwargs),
        "_dsv4_block_sizes": lambda: {32: [[0, 0, 2, 8]]},
    }
    module = ast.Module(body=[definition], type_ignores=[])
    # Postpone framework annotations; execute the production constructor body.
    import __future__

    exec(compile(module, "compressor_constructor", "exec", flags=__future__.annotations.compiler_flag), namespace)
    compressor = namespace["Compressor"](
        NS(),
        NS(hidden_size=4096, qk_rope_head_dim=64, rms_norm_eps=1e-6),
        compress_ratio=compress_ratio,
        cache_config=NS(block_size=32),
    )
    assert compressor.wkv.keep_weight_nd is True
    assert compressor.wgate.keep_weight_nd is True


@pytest.fixture
def modules(monkeypatch):
    package = types.ModuleType("csa_cpu")
    package.__path__ = [str(ATTENTION)]
    monkeypatch.setitem(sys.modules, "csa_cpu", package)
    storage = importlib.import_module("csa_cpu.pto_layer_storage")
    weights = importlib.import_module("csa_cpu.pto_layer_weights")
    for name, module in (
        ("vllm_ascend.attention.pto_layer_storage", storage),
        ("vllm_ascend.attention.pto_layer_weights", weights),
        ("vllm.forward_context", NS(get_forward_context=Mock())),
        ("vllm.utils.torch_utils", NS(direct_register_custom_op=Mock())),
        (
            "vllm_ascend.attention.utils",
            NS(
                **{
                    name: Mock()
                    for name in (
                        "maybe_save_kv_layer_to_connector",
                        "notify_kv_cache_written",
                        "wait_for_kv_layer_from_connector",
                    )
                }
            ),
        ),
        ("vllm_ascend.memcache_comm_fence", NS(record_attention_compute_start=Mock())),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("csa_cpu.pto_layer", None)
    host = importlib.import_module("csa_cpu.pto_layer")
    yield NS(host=host, storage=storage)
    for name in list(sys.modules):
        if name.startswith("csa_cpu."):
            sys.modules.pop(name)


def test_native_page_and_indexer_aliases(modules):
    owner = torch.zeros(2 * 4160, dtype=torch.int8)
    key = owner.as_strided((2, 32, 1, 128), (4160, 128, 128, 1))
    scale = owner.view(torch.float16).as_strided((2, 32, 1, 1), (2080, 1, 1, 1), 2048)
    pages = modules.storage.indexer_storage(key, scale)
    assert pages.data_ptr() == owner.data_ptr() and pages.shape == (2, 4160)
    pages[1, 4096] = 17
    assert owner[8256] == 17
    with pytest.raises(ValueError, match="share"):
        modules.storage.indexer_storage(key, scale.clone())
    partial = torch.zeros(22).as_strided((2, 2, 1, 3), (16, 3, 3, 1))
    with pytest.raises(ValueError, match="final page"):
        modules.storage.physical_pages(partial)
    table_owner = torch.arange(16, dtype=torch.int32).view(2, 8)
    view = modules.storage.table_storage(table_owner[:, :3])
    assert view.data_ptr() == table_owner.data_ptr() and view.shape == (2, 8)


def test_graph_gate_retains_actual_s6_count(modules):
    gate = modules.host.can_replay_csa_graph
    assert gate(num_tokens=18, num_reqs=3, uniform_decode=True, padded_tokens=24)
    assert not gate(num_tokens=17, num_reqs=3, uniform_decode=False, padded_tokens=24)
    assert not gate(num_tokens=24, num_reqs=4, uniform_decode=False, padded_tokens=24)
    assert gate(num_tokens=1, num_reqs=1, uniform_decode=True, padded_tokens=1)


def decoder_forward():
    source = (ATTENTION.parent / "models/deepseek_v4.py").read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2DecoderLayer")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "native_decoder_forward", "exec"), scope)
    return scope["forward"]


@pytest.mark.parametrize("enabled,scaling", [(False, None), (True, None), (True, 3.0)])
def test_decoder_preserves_native_arguments_and_tuple(monkeypatch, enabled, scaling):
    calls = []

    def pre(x, fn, scale, base):
        calls.append(fn)
        return x + 1, "post", "comb"

    def attention(**kwargs):
        assert kwargs["llama_4_scaling"] == scaling
        calls.append("native")
        return kwargs["hidden_states"] * 2

    layer = NS(
        _pto_csa_enabled=enabled,
        hc_pre=pre,
        hc_post=lambda x, residual, post, comb: x + residual,
        hc_attn_fn="attn",
        hc_attn_scale=1,
        hc_attn_base=1,
        hc_ffn_fn="ffn",
        hc_ffn_scale=1,
        hc_ffn_base=1,
        input_layernorm=lambda x: x,
        post_attention_layernorm=lambda x: x,
        mlp=lambda x: x * 3,
        self_attn=Mock(side_effect=attention),
    )
    layer.self_attn.dsa_attn = NS(prefix="target")
    hidden = torch.ones(2, 4, 3)
    positions = torch.arange(2)

    def fused(x, p, out, name):
        assert p is positions and name == "target"
        calls.append("fused")
        out.copy_((x + 1) * 2 + x)

    monkeypatch.setattr(torch.ops.vllm, "full_csa_forward", fused, raising=False)
    result, residual = decoder_forward()(layer, positions, hidden, None, scaling)
    assert torch.equal(hidden, torch.ones_like(hidden))
    assert torch.equal(residual, torch.full_like(hidden, 5))
    assert torch.equal(result, torch.full_like(hidden, 23))
    assert calls == (["fused", "ffn"] if enabled and scaling is None else ["attn", "native", "ffn"])


@pytest.mark.parametrize("field,value", [(None, None), ("hc_mult", 8), ("hc_sinkhorn_iters", 10), ("hc_eps", 1e-5)])
def test_full_hc_configuration_is_validated(modules, monkeypatch, field, value):
    mode = NS(NONE=0, FULL_DECODE_ONLY=1)
    monkeypatch.setitem(sys.modules, "vllm.config", NS(CUDAGraphMode=mode))
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ascend_config", NS(get_ascend_config=lambda: NS(weight_nz_mode=2, enable_kv_nz=False))
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.utils", NS(enable_dsa_cp=lambda: False, oproj_tp_enable=lambda: False)
    )
    monkeypatch.setitem(
        sys.modules, "csa_cpu.pto_kernels.dspark_layer.nz_mode", NS(validate_weight_nz_mode=lambda _: None)
    )
    validate = importlib.import_module("csa_cpu.pto_layer_config").validate_configuration
    hf = NS(
        model_type="deepseek_v4",
        hidden_size=4096,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=8,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        sliding_window=128,
        rms_norm_eps=1e-6,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )
    config = NS(
        model_config=NS(hf_config=hf, dtype=torch.bfloat16, quantization="ascend"),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
        cache_config=NS(block_size=32),
        lora_config=None,
        speculative_config=NS(method="dspark", num_speculative_tokens=5),
        use_v2_model_runner=False,
        compilation_config=NS(cudagraph_mode=mode.NONE),
    )
    if field:
        setattr(hf, field, value)
        with pytest.raises(ValueError, match=field):
            validate(config)
    else:
        validate(config)


def test_native_metadata_bound_without_rebuilding_slots(modules):
    main_owner = torch.zeros(2 * 32768, dtype=torch.uint8)
    compressed = main_owner.view(torch.bfloat16).view(2, 32, 1, 512)
    state = main_owner.view(torch.float32).as_strided((2, 2, 1, 2048), (8192, 2048, 2048, 1))
    raw = torch.zeros_like(compressed)
    index_owner = torch.zeros(2 * 4160, dtype=torch.int8)
    key = index_owner.as_strided((2, 32, 1, 128), (4160, 128, 128, 1))
    scale = index_owner.view(torch.float16).as_strided((2, 32, 1, 1), (2080, 1, 1, 1), 2048)
    inner = index_owner.view(torch.float32).as_strided((2, 2, 1, 512), (1040, 512, 512, 1))
    metadata = {}
    for name in ("swa", "compressed", "state", "indexer", "indexer_state"):
        req = NS(
            block_table=torch.zeros(4, 3, dtype=torch.int32),
            seq_lens=torch.ones(4, dtype=torch.int32),
            query_start_loc=torch.arange(0, 25, 6, dtype=torch.int32),
            slot_mapping=torch.zeros(24, 2, dtype=torch.int32),
            cos={"layer": torch.ones(24, 64)},
            sin={"layer": torch.zeros(24, 64)},
        )
        metadata[name] = NS(decode=req)
    compact = {
        name: (torch.ones(8, 64), torch.zeros(8, 64), torch.zeros(8, 2, dtype=torch.int32))
        for name in ("compressed", "indexer")
    }
    names = (
        "x_hc",
        "x_out",
        "kv_cache",
        "cmp_kv",
        "compress_state",
        "inner_compress_state",
        "idx_native_kv_cache",
        "ori_slot_mapping",
        "state_slot_mapping",
        "inner_state_slot_mapping",
        "cmp_slot_mapping",
        "idx_slot_mapping",
        "cmp_query_start_loc",
        "cmp_seq_lens",
        "kv_seq_lens",
    )
    layer = NS(
        _pto_csa_weights={},
        _pto_csa_param_names=names,
        _pto_csa_scores=torch.empty(24, 512),
        _pto_csa_topk=torch.empty(24, 512, dtype=torch.int32),
        self_attn=NS(dsa_attn=NS(dsa_attn=NS(layer_name="layer"))),
    )
    hidden = torch.zeros(24, 4, 4096, dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    values = modules.host._kernel_arguments(
        layer, hidden, torch.arange(24), output, metadata, compact, (compressed, raw, state, inner, key, scale)
    )
    bound = dict(zip(names, values))
    for name, expected in {
        "x_hc": hidden,
        "x_out": output,
        "kv_cache": raw,
        "cmp_kv": compressed,
        "ori_slot_mapping": metadata["swa"].decode.slot_mapping,
        "state_slot_mapping": metadata["state"].decode.slot_mapping,
        "inner_state_slot_mapping": metadata["indexer_state"].decode.slot_mapping,
        "cmp_slot_mapping": compact["compressed"][2],
        "idx_slot_mapping": compact["indexer"][2],
        "cmp_query_start_loc": metadata["compressed"].decode.query_start_loc,
        "cmp_seq_lens": metadata["compressed"].decode.seq_lens,
        "kv_seq_lens": metadata["indexer"].decode.seq_lens,
    }.items():
        assert bound[name] is expected
    assert bound["compress_state"].untyped_storage().data_ptr() == main_owner.data_ptr()
    assert bound["inner_compress_state"].untyped_storage().data_ptr() == index_owner.data_ptr()
    assert bound["idx_native_kv_cache"].untyped_storage().data_ptr() == index_owner.data_ptr()
    context = NS(additional_kwargs={})
    producer = Mock(return_value=compact["compressed"])
    layer.self_attn.dsa_attn.dsa_attn.impl = NS(_compute_compressor_metadata=producer)
    req = metadata["compressed"].decode
    assert modules.host._compact_metadata(layer, context, req) is compact["compressed"]
    assert modules.host._compact_metadata(layer, context, req) is compact["compressed"]
    assert producer.call_count == 1
    modules.host._compact_metadata(layer, NS(additional_kwargs={}), req)
    assert producer.call_count == 2
