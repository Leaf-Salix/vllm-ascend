from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager


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

    with patch("vllm_ascend.envs.VLLM_ASCEND_PYPTO_QWEN3_MODE", "attention_block"):
        manager.configure(config)

    assert manager.passes == [sentinel]
