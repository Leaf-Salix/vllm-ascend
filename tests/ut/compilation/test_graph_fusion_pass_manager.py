from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager


@pytest.mark.parametrize("mode", ["partial", "full"])
def test_pypto_qwen3_mode_disables_native_fusion_passes(mode):
    manager = GraphFusionPassManager()
    manager.passes.append(object())

    with patch("vllm_ascend.envs.VLLM_ASCEND_PYPTO_QWEN3_MODE", mode):
        # PyPTO modes return before reading the vLLM config. This makes the
        # fail-closed rule explicit: no native pass may replace a PyPTO node.
        manager.configure(object())

    assert manager.passes == []


def test_pypto_qwen3_attention_block_preserves_native_mlp_fusion_passes():
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
