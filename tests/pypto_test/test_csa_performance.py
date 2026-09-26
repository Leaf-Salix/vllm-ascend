# SPDX-License-Identifier: Apache-2.0
"""CPU 回归：性能窗口拒绝错误档位，设备时间与主机时间分别记录。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu
from offline_pd.observer import OfflineCSAObserver
from offline_pd.performance import layer_intervals


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
    assert result["measured_steps"] == 21
    assert result["step_tokens"] == [96] * 21
    assert result["device"]["samples_us"] == [100] * 21
    assert len(set(result["device"]["start_timestamps_raw"])) == 21
    if sample_complete:
        count = 19 if shape_gap else 20
        assert result["decode_cycle"]["samples_us"] == [1000] * count
        assert result["decode_cycle"]["actual_output_tokens"] == [32] * count
        if not shape_gap:
            assert result["decode_cycle"]["actual_output_tokens_per_second"] == 32000
        assert result["execute_sample_device"]["samples_us"] == [900] * 21
    else:
        assert result["decode_cycle"]["samples_us"] == []
        assert result["incomplete_sample_steps"] == 21
    assert "tokens_per_second" not in result


@pytest.mark.parametrize("missing_worker", [False, True])
def test_layer_mapping_counts_overlap_once_and_rejects_missing_tasks(missing_worker):
    rows = []

    def task(name, time, duration, model=49):
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
