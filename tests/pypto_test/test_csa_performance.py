# SPDX-License-Identifier: Apache-2.0
"""CPU 回归：性能窗口拒绝错误档位，设备时间与主机时间分别记录。"""

import gc
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu
from offline_pd.observer import OfflineCSAObserver
from offline_pd.performance import compare_worker_configs, layer_intervals


def test_worker_event_difference_is_reported_without_ignoring_other_configuration():
    native = {"deterministic_level": 0, "dynamic_eplb": False, "cann_event_work_mode": 0,
              "scheduler": {"max_num_batched_tokens": 256}}
    pto = {**native, "cann_event_work_mode": 1}
    assert compare_worker_configs(native, pto) == {"native": 0, "pto": 1}
    with pytest.raises(ValueError, match="worker 配置不同"):
        compare_worker_configs(native, {**pto, "scheduler": {"max_num_batched_tokens": 400}})
    with pytest.raises(ValueError, match="观测覆盖不同"):
        compare_worker_configs(native, {key: value for key, value in pto.items() if key != "cann_event_work_mode"})


def scheduler(tokens, requests):
    return SimpleNamespace(total_num_scheduled_tokens=tokens,
                           num_scheduled_tokens=dict.fromkeys(range(requests), 6))


@pytest.fixture
def observer(monkeypatch):
    instance = OfflineCSAObserver()
    instance.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=0))
    instance.model_runner = SimpleNamespace(
        execute_model=Mock(return_value="output"),
        sample_tokens=Mock(return_value=SimpleNamespace(sampled_token_ids=[[1, 2] for _ in range(16)])),
    )
    monkeypatch.setattr(torch.npu, "synchronize", Mock())
    return instance


@pytest.mark.parametrize("failure", [None, "shape_gap", "missing_forward"])
@pytest.mark.parametrize("host_diagnostics", [False, True])
def test_forward_excludes_prepare_and_postprocess(observer, monkeypatch, failure, host_diagnostics):
    class Event:
        clock = 0

        def __init__(self, **kwargs):
            self.stamp = None

        def record(self):
            self.stamp = Event.clock

        def recorded_time(self):
            return self.stamp

        def elapsed_time(self, end):
            return (end.stamp - self.stamp) / 1000

    def forward():
        Event.clock += 100
        return "hidden"

    @contextmanager
    def synchronize_input_prep():
        Event.clock += 9000  # Existing event wait stays outside forward.
        yield

    def prepare_inputs():
        Event.clock += 1000
        return "prepared"

    def execute(*_):
        with observer.model_runner.synchronize_input_prep():
            assert observer.model_runner._prepare_inputs() == "prepared"
        if host_diagnostics:
            gc.collect(0)
        if failure != "missing_forward":
            assert observer.model_runner._model_forward() == "hidden"
        Event.clock += 20000  # logits 等后置操作不属于 forward。
        return "output"

    observer.model_runner.execute_model = execute
    observer.model_runner._model_forward = forward
    observer.model_runner.synchronize_input_prep = synchronize_input_prep
    observer.model_runner._prepare_inputs = prepare_inputs
    observer.model_runner._dsa_positions_cpu_buf = torch.arange(24)
    monkeypatch.setattr(torch.npu, "Event", Event)
    monkeypatch.setattr(torch.npu, "reset_peak_memory_stats", Mock())
    monkeypatch.setattr(torch.npu, "max_memory_allocated", Mock(return_value=10))
    monkeypatch.setattr(torch.npu, "max_memory_reserved", Mock(return_value=20))
    callbacks_before = list(gc.callbacks)
    observer.offline_begin_forward(2, 24, 4, 10, host_diagnostics)
    shapes = [(4, 4)] + [(24, 4)] * 15
    if failure == "shape_gap":
        shapes.insert(6, (18, 3))
    for shape in shapes:
        observer.model_runner.execute_model(scheduler(*shape))
    result = observer.offline_end_forward()
    assert result["sufficient"] is (failure is None)
    assert result["measured_steps"] == 10
    assert result["steady_step_indices"] == list(range(2, 12))
    if failure != "missing_forward":
        assert result["forward"]["samples_us"] == [100] * 10
        assert result["forward"]["mean_us"] == 100
    assert observer.model_runner.execute_model is execute
    assert observer.model_runner._model_forward is forward
    assert observer.model_runner.synchronize_input_prep is synchronize_input_prep
    assert observer.model_runner._prepare_inputs is prepare_inputs
    assert gc.callbacks == callbacks_before
    if host_diagnostics:
        host = result["host_diagnostics"]
        assert len(host["steps"]) == 10
        assert any(e["phase"] == "stop" for e in host["gc_events"])
        for step in host["steps"]:
            assert step["input_sync_begin"]["monotonic_ns"] <= step["input_sync_end"]["monotonic_ns"]
            assert step["inputs_begin"]["monotonic_ns"] <= step["inputs_end"]["monotonic_ns"]
            assert step["inputs_end"]["monotonic_ns"] <= step["input_prep_end"]["monotonic_ns"]
        if failure != "missing_forward":
            for step in host["steps"]:
                assert step["execute_entry"]["monotonic_ns"] <= step["forward_entry"]["monotonic_ns"]
                assert step["forward_submitted"]["monotonic_ns"] <= step["execute_return"]["monotonic_ns"]
    else:
        assert "host_diagnostics" not in result


@pytest.mark.parametrize("last_shape, sufficient", [((96, 16), True), ((84, 14), False)])
def test_profile_requires_full_shape_and_contiguous_window(observer, monkeypatch, tmp_path, last_shape, sufficient):
    profiler = Mock()
    monkeypatch.setattr(torch_npu.profiler, "profile", Mock(return_value=profiler))
    observer.offline_begin_profile(str(tmp_path), 1, 2, 96, 16, 0)
    # 首步前缀恢复及被调度器拆分的 batch 都不能消耗稳态步序号。
    for shape in [(16, 16), (96, 16), (84, 14), (96, 16), last_shape]:
        assert observer.model_runner.execute_model(scheduler(*shape)) == "output"
    result = observer.offline_end_profile()
    assert result["sufficient"] is sufficient
    assert result["profiled_steps"] == 2
    assert result["window"][0] == {"steady_step_index": 1, "scheduled_tokens": 96, "requests": 16}
    profiler.start.assert_called_once()
    profiler.stop.assert_called_once()


@pytest.mark.parametrize("sample_complete, shape_gap", [(True, False), (False, False), (True, True)])
def test_steady_covers_sampling_and_rejects_unfinished_cycles(observer, monkeypatch, sample_complete, shape_gap):
    class Event:
        clock = 0

        def __init__(self, **kwargs):
            self.stamp = None

        def record(self):
            Event.clock += 100
            self.stamp = Event.clock

        def recorded_time(self):
            return self.stamp

        def elapsed_time(self, end):
            return (end.stamp - self.stamp) / 1000

    monkeypatch.setattr(torch.npu, "Event", Event)
    monkeypatch.setattr(torch.npu, "reset_peak_memory_stats", Mock())
    monkeypatch.setattr(torch.npu, "max_memory_allocated", Mock(return_value=10))
    monkeypatch.setattr(torch.npu, "max_memory_reserved", Mock(return_value=20))
    observer.offline_begin_steady(2, 96, 16)
    shapes = [(16, 16), (84, 14)] + [(96, 16)] * 27
    if shape_gap:
        # 中途出现非满档步时，不能把它两侧的起点拼成一个满档周期。
        shapes.insert(12, (84, 14))
    for shape in shapes:
        observer.model_runner.execute_model(scheduler(*shape))
        if sample_complete:
            # 模拟 execute 之后的采样/草稿，旧计时不会覆盖这段时间。
            Event.clock += 700
            observer.model_runner.sample_tokens(None)
    result = observer.offline_end_steady()
    assert result["sufficient"] is (sample_complete and not shape_gap)
    assert result["measured_steps"] == 11
    assert result["step_tokens"] == [96] * 11
    assert result["device"]["samples_us"] == [100] * 11
    assert len(set(result["device"]["start_timestamps_raw"])) == 11
    if sample_complete:
        count = 9 if shape_gap else 10
        assert result["decode_cycle"]["samples_us"] == [1000] * count
        assert result["decode_cycle"]["actual_output_tokens"] == [32] * count
        if not shape_gap:
            assert result["decode_cycle"]["actual_output_tokens_per_second"] == 32000
        assert result["decode_cycle"]["mean_us"] == 1000
        assert result["execute_sample_device"]["samples_us"] == [900] * 11
    else:
        assert result["decode_cycle"]["samples_us"] == []
        assert result["incomplete_sample_steps"] == 11
    assert "tokens_per_second" not in result


@pytest.mark.parametrize("missing_worker", [False, True])
@pytest.mark.parametrize("kernel_names", [False, True])
def test_layer_mapping_counts_overlap_once_and_rejects_missing_tasks(missing_worker, kernel_names):
    rows = []

    def task(name, time, duration, model=49):
        if kernel_names and name in ("HcPre", "HcPost", "CompressorMetadata"):
            name += "_c12a9d608e5abdbd1728e849208025ff_0"
        rows.append({"name": name, "start_ns": time * 1000, "duration_ns": duration * 1000,
                     "model": model, "task": len(rows)})

    for layer in range(43):
        start = layer * 40
        if layer >= 2 and layer % 2 == 0:
            if layer == 2:
                task("CompressorMetadata", start - 4, 2)
                task("CompressorMetadata", start - 2, 2)
            task("simpler_aicpu_kernel_exec_example", start, 10, None)
            if not (missing_worker and layer == 4):
                task("aicore_kernel_mode_0_mix_aic", start + 1, 8, None)
        else:
            task("HcPre", start, 2)
            task("HcPost", start + 8, 2)
        task("HcPre", start + 20, 2)
        task("HcPost", start + 28, 2)
    rows.sort(key=lambda row: row["start_ns"])
    if missing_worker:
        with pytest.raises(ValueError, match="runtime/worker 数量不符"):
            layer_intervals(rows, "pto", steps=1)
    else:
        result = layer_intervals(rows, "pto", steps=1)["intervals"]
        assert len(result) == 21
        assert result[0]["us"] == 14
        assert result[0]["leading_metadata_tasks"] == 2
        assert all(row["body_us"] == 10 for row in result)
        assert all(row["us"] == 10 for row in result[1:])
