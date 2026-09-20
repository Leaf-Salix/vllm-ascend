# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.ops import pypto_swimlane


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(pypto_swimlane, "_CONTROLLER", None)
    monkeypatch.setenv("VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_LEVEL", "4")
    monkeypatch.setenv("VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_MAX_CAPTURES", "1")

    events = []
    state = {}
    npu = SimpleNamespace(current_device=lambda: 0, is_current_stream_capturing=Mock(return_value=False))
    monkeypatch.setattr(pypto_swimlane.torch, "npu", npu, raising=False)

    def init(**kwargs):
        state.update(kwargs)
        state["window"] = 0
        events.append("init")

    def end():
        events.append("end")
        output = state["output_dir"]
        if state["window"]:
            output = output / f"window_{state['window']}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "chip_swimlane_records.json").write_text(json.dumps({"aicore_tasks": [[0, 0, 0, 1, 2]]}))
        (output / "deps.json").write_text("{}")
        state["window"] += 1

    adapter = ModuleType("pypto.torch")
    adapter.init = Mock(side_effect=init)
    adapter.begin_dfx = Mock(side_effect=lambda: events.append("begin"))
    adapter.end_dfx = Mock(side_effect=end)
    monkeypatch.setitem(sys.modules, "pypto.torch", adapter)

    def convert(command, **kwargs):
        events.append("convert")
        assert command[:3] == [sys.executable, "-m", "simpler_setup.tools.swimlane_converter"]
        assert Path(command[3]).is_file()
        assert command[4] == "--deps-json" and Path(command[5]).is_file()
        assert command[6] == "-o"
        assert kwargs == {"check": True, "timeout": 60}
        Path(command[-1]).write_text(json.dumps({"traceEvents": [{"name": "Qwen3", "ph": "X", "args": {"taskId": 0}}]}))

    converter = Mock(side_effect=convert)
    monkeypatch.setattr(pypto_swimlane.subprocess, "run", converter)
    yield SimpleNamespace(adapter=adapter, converter=converter, events=events, npu=npu, state=state)
    monkeypatch.setattr(pypto_swimlane, "_CONTROLLER", None)


def test_eager_warms_live_forward_then_collects_without_rerun(runtime):
    pypto_swimlane.init_pypto(runtime.adapter.init)
    for _ in range(3):
        with pypto_swimlane.eager_model_swimlane("qwen3", ((2, 4),), graph_enabled=False):
            runtime.events.append("forward")

    assert runtime.events == ["init", "forward", "begin", "forward", "end", "convert", "forward"]
    runtime.adapter.init.assert_called_once()
    assert runtime.state["enable_chip_swimlane"] == 4
    assert runtime.state["enable_dep_gen"] is True
    assert Path(runtime.state["output_dir"]).parent == Path(pypto_swimlane.envs.VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_DIR)


def test_graph_enabled_run_reserves_budget_for_replay(runtime):
    pypto_swimlane.init_pypto(runtime.adapter.init)
    with pypto_swimlane.eager_model_swimlane("qwen3", ((2, 4),), graph_enabled=True):
        runtime.events.append("warmup")
    with pypto_swimlane.graph_replay_swimlane("FULL:batch=2"):
        runtime.events.append("replay")

    assert runtime.events == ["init", "warmup", "begin", "replay", "end", "convert"]
    metadata = json.loads((runtime.state["output_dir"] / "capture.json").read_text())
    assert metadata["mode"] == "graph_replay"
    assert metadata["graph"] == "FULL:batch=2"


def test_capture_never_opens_a_dfx_window(runtime):
    pypto_swimlane.init_pypto(runtime.adapter.init)
    runtime.npu.is_current_stream_capturing.return_value = True
    with pypto_swimlane.eager_model_swimlane("qwen3", ((2, 4),), graph_enabled=False):
        runtime.events.append("capture")
    with pypto_swimlane.graph_replay_swimlane("nested"):
        runtime.events.append("nested_replay")

    assert runtime.events == ["init", "capture", "nested_replay"]
    runtime.adapter.begin_dfx.assert_not_called()


def test_failed_replay_closes_window_and_consumes_budget(runtime):
    pypto_swimlane.init_pypto(runtime.adapter.init)
    with pytest.raises(RuntimeError, match="replay failed"), pypto_swimlane.graph_replay_swimlane("broken"):
        raise RuntimeError("replay failed")
    with pypto_swimlane.graph_replay_swimlane("not_retried"):
        runtime.events.append("replay")

    assert runtime.events == ["init", "begin", "end", "replay"]
    runtime.converter.assert_not_called()


def test_empty_graph_capture_is_diagnostic_only(runtime):
    pypto_swimlane.init_pypto(runtime.adapter.init)
    end = runtime.adapter.end_dfx.side_effect

    def empty_end():
        end()
        output = runtime.state["output_dir"]
        (output / "chip_swimlane_records.json").write_text(json.dumps({"aicore_tasks": []}))
        (output / "deps.json").unlink()

    runtime.adapter.end_dfx.side_effect = empty_end
    with pypto_swimlane.graph_replay_swimlane("without_pypto"):
        pass

    runtime.converter.assert_not_called()
    metadata = json.loads((runtime.state["output_dir"] / "capture.json").read_text())
    assert metadata["status"] == "no_pypto_tasks"


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_LEVEL", "5"),
        ("VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_LEVEL", "-1"),
        ("VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_DIR", ""),
        ("VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_MAX_CAPTURES", "0"),
    ],
)
def test_invalid_configuration_is_rejected(runtime, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    monkeypatch.setattr(pypto_swimlane, "_CONTROLLER", None)
    with pytest.raises(ValueError):
        pypto_swimlane.init_pypto(runtime.adapter.init)
    runtime.adapter.init.assert_not_called()


def test_conversion_failure_is_visible(runtime):
    pypto_swimlane.init_pypto(runtime.adapter.init)
    runtime.converter.side_effect = subprocess.CalledProcessError(1, "converter")
    with pytest.raises(subprocess.CalledProcessError), pypto_swimlane.graph_replay_swimlane("broken_conversion"):
        pass


def test_v2_updates_parameters_before_the_replay_window_closes(monkeypatch):
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    from vllm_ascend.worker.v2 import aclgraph_utils

    events = []
    output = object()

    @contextmanager
    def replay_window(_name):
        events.append("begin")
        try:
            yield
        finally:
            events.append("end")

    monkeypatch.setattr(
        aclgraph_utils.envs,
        "VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_LEVEL",
        1,
    )
    monkeypatch.setattr(pypto_swimlane, "graph_replay_swimlane", replay_window)
    monkeypatch.setattr(
        ModelCudaGraphManager,
        "run_fullgraph",
        lambda _self, _desc: events.append("replay") or output,
    )
    monkeypatch.setattr(aclgraph_utils, "set_forward_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(aclgraph_utils, "get_forward_context", object)
    monkeypatch.setattr(
        aclgraph_utils,
        "update_full_graph_params",
        lambda *args, **kwargs: events.append("update"),
    )

    manager = object.__new__(aclgraph_utils.ModelAclGraphManager)
    manager.device = aclgraph_utils.torch.device("cpu")
    manager.vllm_config = SimpleNamespace()
    manager.model_runner = SimpleNamespace(
        input_buffers=SimpleNamespace(positions=aclgraph_utils.torch.zeros(4)),
        dp_size=1,
        model_state=SimpleNamespace(attn_metadata=None),
        attn_backends={"test": object()},
        update_stream=None,
        speculative_config=None,
    )

    assert manager.run_fullgraph(SimpleNamespace(num_tokens=4, cg_mode="FULL")) is output
    assert events == ["begin", "replay", "update", "end"]


def test_v1_wraps_only_the_live_graph_replay(monkeypatch):
    from vllm.config import CUDAGraphMode

    from vllm_ascend.compilation import acl_graph

    events = []
    output = object()
    batch = "batch=4"

    @contextmanager
    def replay_window(name):
        assert name == f"FULL:{batch}"
        events.append("begin")
        try:
            yield
        finally:
            events.append("end")

    monkeypatch.setattr(
        acl_graph.ascend_envs,
        "VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_LEVEL",
        1,
    )
    monkeypatch.setattr(acl_graph, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False))
    monkeypatch.setattr(
        acl_graph,
        "get_forward_context",
        lambda: SimpleNamespace(batch_descriptor=batch, cudagraph_runtime_mode=CUDAGraphMode.FULL),
    )
    monkeypatch.setattr(pypto_swimlane, "graph_replay_swimlane", replay_window)

    wrapper = object.__new__(acl_graph.ACLGraphWrapper)
    wrapper.runtime_mode = CUDAGraphMode.FULL
    wrapper.concrete_aclgraph_entries = {
        batch: acl_graph.ACLGraphEntry(
            batch_descriptor=batch,
            aclgraph=SimpleNamespace(replay=lambda: events.append("replay")),
            output=output,
        )
    }
    wrapper.is_debugging_mode = False
    wrapper.enable_enpu = True
    wrapper.use_eagle = False
    wrapper.runnable = Mock()

    assert wrapper() is output
    assert events == ["begin", "replay", "end"]
    wrapper.runnable.assert_not_called()
