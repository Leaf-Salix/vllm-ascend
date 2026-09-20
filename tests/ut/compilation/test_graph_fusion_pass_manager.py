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
