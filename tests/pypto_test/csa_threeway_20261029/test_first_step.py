# SPDX-License-Identifier: Apache-2.0
"""227 CPU验证字节比较及诊断门禁；不启动模型、NPU或本机测试进程树。"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from csa_threeway_20261029.first_step_observer import (
    CACHE_NAMES,
    FirstStepObserver,
    bit_metrics,
    check_complete,
    snapshot,
)


def test_bit_comparison_preserves_signed_zero_and_nan():
    actual = torch.tensor([0.0, float("nan")])
    expected = actual.clone()
    assert bit_metrics(actual, expected)["bit_equal"]
    expected[0] = -0.0
    assert not bit_metrics(actual, expected)["bit_equal"]


def test_snapshot_preserves_full_alias_and_independent_history():
    value = torch.arange(16, dtype=torch.int8)
    copies = snapshot({"page": value, "alias": value.view(16), "tail": value[8:]})
    value.zero_()
    assert copies["page"] is copies["alias"]
    assert copies["page"].tolist() == list(range(16))
    assert copies["tail"].tolist() == list(range(8, 16))


def test_diagnostic_cannot_pass_with_missing_layers_or_full_replay(tmp_path):
    spec = importlib.util.spec_from_file_location("first_step_launch_test", Path(__file__).with_name("launch.py"))
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    layers = list(range(2, 43, 2))
    report = {
        "status": "first_step_complete",
        "first_forward": {"graph_mode": "NONE"},
        "actual_pto_config": {
            "ring_task_window": [2048] * 4,
            "ring_dep_pool": None,
            "ring_heap": [256 << 20, 512 << 20, 256 << 20, 256 << 20],
        },
        "layers": [
            {
                "layer": i,
                "restored_exact": True,
                "input_unchanged": True,
                "output_unchanged": True,
                "use_index_cache": False,
                "skip_topk": False,
                "metadata_restored_exact": True,
                "changed_readonly_inputs": [],
                "output": {"bit_equal": True},
                "caches": {
                    name: {"bit_equal": True}
                    for name in ("cache_rows", "main_cache", "main_state", "idx_state", "idx_cache")
                },
                "shared_topk": {"bit_equal": True},
            }
            for i in layers
        ],
    }
    assert check_complete(report, layers)
    (tmp_path / "FIRST_STEP_DIAGNOSTIC.json").write_text(json.dumps({"kind": "first_step_complete", "layers": layers}))

    def passed():
        (tmp_path / "same_input_first_step.json").write_text(json.dumps(report))
        return launch.first_step_passed(tmp_path)

    assert passed()
    report["first_forward"]["graph_mode"] = "FULL"
    assert not check_complete(report, layers) and not passed()
    report["first_forward"]["graph_mode"] = "NONE"
    report["layers"][0]["output"]["bit_equal"] = False
    assert not check_complete(report, layers) and not passed()
    report["layers"][0]["output"]["bit_equal"] = True
    report["layers"][0]["shared_topk"]["bit_equal"] = False
    assert not check_complete(report, layers) and not passed()
    report["layers"][0]["shared_topk"]["bit_equal"] = True
    report["actual_pto_config"]["ring_dep_pool"] = [2048] * 4
    assert not passed()
    report["actual_pto_config"]["ring_dep_pool"] = None
    report["layers"].pop()
    assert not check_complete(report, layers) and not passed()
    report["layers"].append({**report["layers"][-1], "layer": 42})
    assert passed()
    (tmp_path / "FIRST_DIFFERENCE.json").write_text("{}")
    assert not passed()  # Another rank's earlier difference must never be hidden by diagnostic completion.


@pytest.mark.parametrize("native_raises", [False, True])
def test_native_shadow_restores_pto_state_even_after_exception(tmp_path, monkeypatch, native_raises):
    observer = object.__new__(FirstStepObserver)
    observer._first_step_root = tmp_path
    observer._first_step_report = {"layers": []}
    hidden, positions = torch.arange(4.0).reshape(2, 2), torch.tensor([3, 4])
    output = torch.empty_like(hidden)
    caches = {name: torch.full((2, 4), float(i)) for i, name in enumerate(CACHE_NAMES)}
    topk = torch.full((2, 2), 7, dtype=torch.int32)
    impl = SimpleNamespace(topk_indices_buffer=topk, use_index_cache=False, skip_topk=False)
    runtime = SimpleNamespace(wrapper=SimpleNamespace(dsa_attn=SimpleNamespace(impl=impl)), prefixes={"swa": "p"})
    observer._first_step_indices = {id(runtime): 2}
    factory = lambda: None
    req = SimpleNamespace(
        sas_metadata=None, qli_metadata=None, native_metadata_factory=factory, query_start_loc=torch.tensor([0, 1, 2])
    )
    context = SimpleNamespace(attn_metadata={"p": SimpleNamespace(decode=req)})
    native_inputs = {"x_hc": hidden, "position_ids": positions}
    call = SimpleNamespace(buffers=caches, native=native_inputs, policy=SimpleNamespace(constexprs=lambda: {}))

    def pto(_runtime, _context, actual_hidden, _positions, actual_output, _cache):
        initial = snapshot({**caches, "shared_topk": topk})
        observer._first_step_pending = (call, initial, snapshot(native_inputs))
        for cache in caches.values():
            cache.add_(1)
        actual_output.copy_(actual_hidden + 2)
        return "original PTO return"

    def native(_wrapper, actual_hidden, _positions, actual_output):
        for cache in caches.values():
            cache.add_(1)
        req.sas_metadata = torch.tensor([99])
        req.native_metadata_factory = None
        req.query_start_loc.add_(20)
        if native_raises:
            topk.zero_()
            raise RuntimeError("injected Native failure")
        actual_output.copy_(actual_hidden + 2)

    monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.dsv4_csa", SimpleNamespace(_native_attention_half=native))
    if native_raises:
        with pytest.raises(RuntimeError, match="injected Native failure"):
            observer._first_step_pair(runtime, context, hidden, positions, output, (), pto)
    else:
        assert observer._first_step_pair(runtime, context, hidden, positions, output, (), pto) == "original PTO return"
        assert observer._first_step_report["layers"][0]["metadata_restored_exact"]
    assert torch.equal(output, hidden + 2)
    assert all(torch.equal(cache, torch.full_like(cache, float(i + 1))) for i, cache in enumerate(caches.values()))
    assert torch.equal(topk, torch.full_like(topk, 7))
    assert req.sas_metadata is None and req.native_metadata_factory is factory
    assert req.query_start_loc.tolist() == [0, 1, 2]
