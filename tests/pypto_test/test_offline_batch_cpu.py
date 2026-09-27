# SPDX-License-Identifier: Apache-2.0
"""Batch admission must not run early requests or change model memory."""

from types import SimpleNamespace

from offline_pd.batch import generate_aligned_batch


def test_all_requests_are_queued_before_resume():
    calls = []
    outputs = [object(), object()]

    class Engine:
        def sleep(self, *, level, mode):
            assert (level, mode) == (0, "keep")
            calls.append("pause")

        def enqueue(self, prompts, params, *, use_tqdm):
            assert calls == ["pause"] and len(prompts) == 2
            calls.append("enqueue_all")

        def collective_rpc(self, method):
            assert calls == ["pause", "enqueue_all"] and method == "offline_batch_barrier"
            calls.append("dp_ready")

        def wake_up(self, *, tags):
            assert calls == ["pause", "enqueue_all", "dp_ready"] and tags == ["scheduling"]
            calls.append("resume")

        def wait_for_completion(self, *, use_tqdm):
            assert calls[-1] == "resume"
            return outputs

    assert generate_aligned_batch(Engine(), ["a", "b"], object()) is outputs


def test_actual_level_zero_does_not_call_memory_sleep_or_wakeup():
    from vllm.v1.engine.core import EngineCore

    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("A scheduling-only pause touched model memory")

    engine = SimpleNamespace(
        pause_scheduler=lambda **kwargs: calls.append(("pause", kwargs)),
        resume_scheduler=lambda: calls.append(("resume", {})),
        model_executor=SimpleNamespace(sleep=forbidden, wake_up=forbidden, is_sleeping=False),
    )
    EngineCore.sleep(engine, level=0, mode="keep")
    EngineCore.wake_up(engine, tags=["scheduling"])
    assert calls == [("pause", {"mode": "keep", "clear_cache": False}), ("resume", {})]
