# SPDX-License-Identifier: Apache-2.0
"""CPU 回归：性能窗口拒绝错误档位，设备时间与主机时间分别记录。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu
from offline_pd.observer import OfflineCSAObserver


def scheduler(tokens, requests):
    return SimpleNamespace(total_num_scheduled_tokens=tokens,
                           num_scheduled_tokens=dict.fromkeys(range(requests), 6))


@pytest.fixture
def observer(monkeypatch):
    instance = OfflineCSAObserver()
    instance.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=0))
    instance.model_runner = SimpleNamespace(execute_model=Mock(return_value="output"))
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


def test_steady_uses_fresh_device_events_and_skips_partial_batch(observer, monkeypatch):
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
    for shape in [(16, 16), (84, 14)] + [(96, 16)] * 22:
        observer.model_runner.execute_model(scheduler(*shape))
    result = observer.offline_end_steady()
    assert result["sufficient"]
    assert result["measured_steps"] == 20
    assert result["step_tokens"] == [96] * 20
    assert result["device"]["samples_us"] == [100] * 20
    assert len(set(result["device"]["start_timestamps_raw"])) == 20
    assert "tokens_per_second" not in result
