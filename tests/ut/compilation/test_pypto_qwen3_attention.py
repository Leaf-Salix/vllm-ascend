from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

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
