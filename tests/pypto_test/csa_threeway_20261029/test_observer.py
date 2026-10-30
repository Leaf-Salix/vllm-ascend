# SPDX-License-Identifier: Apache-2.0
"""不依赖本机vLLM/NPU，检查Native计时、过滤和同步发生的时机。"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def test_native_timing_is_prewarmed_and_only_resolved_at_end(monkeypatch):
    calls = []

    class Event:
        def record(self):
            calls.append("record")

        def elapsed_time(self, _other):
            calls.append("elapsed")
            return 1.0

    class Base:
        def offline_begin_observation(self):
            self._offline_csa_layer_names = ["c4"]
            return {"begin": True}

        def offline_end_observation(self):
            return {"end": True}

    context = SimpleNamespace(
        capturing=False,
        is_draft_model=False,
        attn_metadata={
            "c4": SimpleNamespace(num_prefills=0, num_actual_tokens=184, decode=SimpleNamespace(num_reqs_actual=40))
        },
        cudagraph_runtime_mode=SimpleNamespace(name="FULL"),
    )
    modules = {
        "torch": {"npu": SimpleNamespace(synchronize=lambda: calls.append("synchronize"))},
        "csa_threeway_20261029.runtime": {"configure_runtime": lambda: None},
        "csa_tnd_model_validation_20261023.observer": {"TNDObserver": Base},
        "offline_pd.forward_timing": {"prewarm_timing_events": lambda n: [(Event(), Event()) for _ in range(n)]},
        "vllm.forward_context": {"get_forward_context": lambda: context},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("threeway_timing_test", Path(__file__).with_name("observer.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observer = module.TimedObserver()

    def original(_tokens):
        calls.append("forward")

    class Positions:
        def __getitem__(self, _key):
            return self

        def tolist(self):
            return list(range(184))

    observer.model_runner = SimpleNamespace(_model_forward=original, _dsa_positions_cpu_buf=Positions())

    def dummy_run():
        observer.model_runner._model_forward(6)
        raise RuntimeError("dummy-control")

    observer.model_runner._dummy_run = dummy_run
    observer.offline_begin_observation()
    try:
        observer.model_runner._dummy_run()
    except RuntimeError as error:
        assert str(error) == "dummy-control"
    assert not observer._threeway_in_dummy_batch
    for _ in range(28):
        observer.model_runner._model_forward(240)
    # No PTO runtime is needed for the Native arm to be timed.
    assert len(observer._threeway_samples) == 20
    assert calls.count("record") == 40
    assert "elapsed" not in calls and "synchronize" not in calls
    context.capturing = True
    observer.model_runner._model_forward(240)
    context.capturing = False
    context.is_draft_model = True
    observer.model_runner._model_forward(240)
    context.is_draft_model = False
    observer.model_runner._model_forward(6)
    assert observer._threeway_seen == 28
    result = observer.offline_end_observation()
    assert result["threeway_timing"]["complete"]
    assert len(result["threeway_timing"]["samples"]) == 20
    assert calls.count("synchronize") == 1 and calls.count("elapsed") == 39
    assert observer.model_runner._model_forward is original
    assert observer.model_runner._dummy_run is dummy_run
    assert any(row["is_dummy_batch"] and row["steps"] == 1 for row in result["policy_descriptors"])
