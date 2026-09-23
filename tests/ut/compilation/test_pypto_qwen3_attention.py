from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager
from vllm_ascend.patch.worker import patch_qwen3vl


def test_pypto_qwen3_attention_preserves_native_mlp_fusion_passes():
    sentinel = object()
    manager = GraphFusionPassManager()
    manager.passes.append(sentinel)
    config = SimpleNamespace(
        additional_config={
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
                "fuse_qknorm_rope": False,
                "fuse_allreduce_rms": False,
                "fuse_muls_add": False,
            }
        },
        compilation_config=SimpleNamespace(pass_config=SimpleNamespace(enable_sp=False)),
    )

    with patch("vllm_ascend.envs.VLLM_ASCEND_PYPTO_QWEN3_MODE", "attention_only"):
        manager.configure(config)

    assert manager.passes == [sentinel]


def test_pypto_qwen3_keeps_native_decoder_layer_forward():
    assert Qwen3DecoderLayer.forward.__module__ == "vllm.model_executor.models.qwen3"


def test_pypto_qwen3_attention_only_routes_decode_and_prefill_separately():
    hidden = torch.empty((1, 5120), dtype=torch.bfloat16)
    qkv = torch.empty((1, 7168), dtype=torch.bfloat16)
    qkv_proj = Mock(return_value=(qkv, None))
    qkv_proj.weight = torch.empty(0)
    o_proj = Mock(return_value=(hidden, None))
    o_proj.weight = torch.empty(0)
    attention = SimpleNamespace(
        _pypto_attention_only_enabled=True,
        qkv_proj=qkv_proj,
        q_norm=Mock(side_effect=lambda tensor: tensor, weight=torch.empty(0)),
        k_norm=Mock(side_effect=lambda tensor: tensor, weight=torch.empty(0)),
        rotary_emb=Mock(return_value=(hidden, hidden)),
        o_proj=o_proj,
        attn=Mock(return_value=hidden),
        q_size=5120,
        kv_size=1024,
        head_dim=128,
    )
    attention.attn.layer_name = "model.layers.0.self_attn.attn"
    attention.rotary_emb.cos_sin_cache = torch.empty(0)
    positions = torch.zeros(1, dtype=torch.int64)

    with (
        patch.object(patch_qwen3vl, "get_attention_context") as context,
        patch.object(torch.ops.vllm, "pypto_qwen3_attention_only", create=True) as pypto_op,
    ):
        for state, actual_tokens, use_pypto in (
            ("PrefillNoCache", 1, False),
            ("DecodeOnly", 0, True),
            ("DecodeOnly", 1, True),
        ):
            context.return_value = (
                SimpleNamespace(attn_state=SimpleNamespace(name=state), num_actual_tokens=actual_tokens),
                None,
                None,
                None,
            )
            qkv_proj.reset_mock()
            pypto_op.reset_mock()
            result = patch_qwen3vl.forward_with_split_qkv_rmsnorm_mrope(attention, positions, hidden)
            assert result.shape == hidden.shape
            assert pypto_op.called is use_pypto
            assert qkv_proj.called is not use_pypto

        # A decode bucket with more than one row is outside the first kernel's
        # one-token contract and must stay on the native path.
        two_rows = torch.empty((2, 5120), dtype=torch.bfloat16)
        qkv_proj.return_value = (torch.empty((2, 7168), dtype=torch.bfloat16), None)
        attention.attn.return_value = two_rows
        o_proj.return_value = (two_rows, None)
        context.return_value = (SimpleNamespace(attn_state=SimpleNamespace(name="DecodeOnly")), None, None, None)
        qkv_proj.reset_mock()
        pypto_op.reset_mock()
        result = patch_qwen3vl.forward_with_split_qkv_rmsnorm_mrope(attention, positions, two_rows)
        assert result.shape == two_rows.shape
        assert qkv_proj.called
        assert not pypto_op.called


def test_pypto_qwen3_decode_graph_has_no_native_attention_update_handles():
    with (
        patch("vllm_ascend.envs.VLLM_ASCEND_PYPTO_QWEN3_MODE", "attention_only"),
        patch("vllm_ascend.attention.attention_v1.get_graph_params", return_value=None),
        patch("vllm_ascend.attention.attention_v1._EXTRA_CTX", SimpleNamespace(is_draft_model=False)),
    ):
        assert AscendAttentionBackendImpl.update_graph_params(None, None, 1, None) is None


def test_pypto_qwen3_attention_only_borrows_vllm_paging_metadata():
    layer_name = "model.layers.0.self_attn.attn"
    cache = torch.empty((2, 1, 128, 8, 128), dtype=torch.bfloat16)
    slot_mapping = torch.zeros(1, dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=slot_mapping,
        block_tables=torch.zeros((1, 4), dtype=torch.int32),
        seq_lens_device=torch.ones(1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
    )
    args = (
        torch.zeros(1, dtype=torch.int64),
        torch.empty((1, 5120), dtype=torch.bfloat16),
        torch.empty((7168, 5120), dtype=torch.bfloat16, device="meta"),
        torch.empty(128, dtype=torch.bfloat16),
        torch.empty(128, dtype=torch.bfloat16),
        torch.empty((512, 128), dtype=torch.bfloat16),
        torch.empty((5120, 5120), dtype=torch.bfloat16, device="meta"),
        torch.empty((1, 5120), dtype=torch.bfloat16),
        layer_name,
    )
    with (
        patch(
            "vllm.model_executor.layers.attention.attention.get_attention_context",
            return_value=(metadata, SimpleNamespace(layer_name=layer_name), cache, slot_mapping),
        ),
        patch.object(patch_qwen3vl, "_EXTRA_CTX", SimpleNamespace(in_profile_run=False)),
        patch("vllm_ascend.ops.pypto_qwen3_attention.attention_only") as kernel,
    ):
        patch_qwen3vl._pypto_qwen3_attention_only(*args)

    call = kernel.call_args.kwargs
    assert call["normalized_hidden"] is args[1]
    assert call["block_table"] is metadata.block_tables
    assert call["seq_lens"] is metadata.seq_lens_device
    assert call["query_start_loc"] is metadata.query_start_loc
    assert call["slot_mapping"].data_ptr() == slot_mapping.data_ptr()
    assert call["key_cache"].data_ptr() == cache[0].data_ptr()
    assert call["value_cache"].data_ptr() == cache[1].data_ptr()
    assert call["output"] is args[7]
