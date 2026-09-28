# SPDX-License-Identifier: Apache-2.0
"""CPU检查实际LLM构造参数：两侧对齐部署模板，eager诊断不能冒充静态编译。"""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from offline_pd import run


@pytest.mark.parametrize("backend", ["native", "pto"])
@pytest.mark.parametrize("graph_mode", ["full_decode_only", "eager"])
def test_decode_llm_uses_template_optimizations(backend, graph_mode, monkeypatch, tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "plan.json").write_text(json.dumps({
        "model": run.FORMAL_MODEL, "decode": {"speculative_tokens": 5},
        "cases": [{"history": 131072, "p_dp_rank": 0}],
    }))
    supplied = {}

    class CapturedLLM(Exception):
        pass

    def llm(**kwargs):
        supplied.update(kwargs)
        raise CapturedLLM

    vllm = ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = llm, Mock()
    config = ModuleType("vllm.config")
    config.KVTransferConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = Mock()
    for module in (vllm, config, platforms):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(run.signal, "signal", Mock())
    monkeypatch.setattr(sys, "argv", [
        "run.py", "decode", "--bank", str(bank), "--output", str(tmp_path / "out"),
        "--rank", "0", "--backend", backend, "--graph-mode", graph_mode,
        "--batch", "24", "--capture-sizes", "144",
    ])
    with pytest.raises(CapturedLLM):
        run.main()

    additional = supplied["additional_config"]
    compilation = additional["ascend_compilation_config"]
    assert compilation["enable_npugraph_ex"] == (graph_mode != "eager")
    assert compilation["enable_static_kernel"] == (graph_mode != "eager")
    assert compilation["fuse_norm_quant"] is True
    assert additional["enable_cpu_binding"] is True
    assert additional["multistream_overlap_shared_expert"] is True
    assert additional["recompute_scheduler_enable"] is False
    assert additional["weight_nz_mode"] == 2
    assert additional["eplb_config"]["dynamic_eplb"] is False
    assert supplied["async_scheduling"] is True
    assert supplied["disable_hybrid_kv_cache_manager"] is False
    # 固定权重、独立KV和测试档位是用户已确认的负载；不照搬模板的另一份权重与B32。
    assert supplied["model"] == run.FORMAL_MODEL
    assert supplied["enable_prefix_caching"] is False
    assert supplied["max_num_seqs"] == 24
    assert supplied["gpu_memory_utilization"] == 0.95
    assert supplied["model_loader_extra_config"]["num_threads"] == 128
    assert supplied["speculative_config"] == {
        "method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True,
    }
    if graph_mode != "eager":
        assert supplied["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    else:
        assert supplied["enforce_eager"] is True
