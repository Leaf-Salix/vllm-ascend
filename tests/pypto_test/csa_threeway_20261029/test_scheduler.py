# SPDX-License-Identifier: Apache-2.0
"""验证首token/事件首差以及请求结束时缺少事件尾部都会终止。"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def scheduler(monkeypatch, tmp_path):
    class Base:
        def _update_request_with_output(self, request, new_token_ids):
            request.output_token_ids.extend(new_token_ids)
            if request.stopped:
                self.acceptance_events.pop(request.request_id, None)
            return new_token_ids, request.stopped

        def make_spec_decoding_stats(
            self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens, num_invalid_spec_tokens, request_id
        ):
            self.acceptance_events.setdefault(request_id, []).append([num_draft_tokens, num_accepted_tokens])
            return spec_decoding_stats

    dependency = types.ModuleType("low_acceptance_20261001.scheduler")
    dependency.AcceptanceScheduler = Base
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    spec = importlib.util.spec_from_file_location("checked_scheduler_test", Path(__file__).with_name("scheduler.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    obj = module.CheckedScheduler.__new__(module.CheckedScheduler)
    obj.parallel_config = SimpleNamespace(data_parallel_rank=0)
    obj.log_stats = True
    obj._threeway_output = tmp_path
    obj._threeway_reference = {"q0": [10, 20]}
    obj._threeway_events = {"q0": [[5, 3], [5, 2]]}
    obj.acceptance_events = {"req": [[5, 3]]}
    request = SimpleNamespace(
        request_id="req", kv_transfer_params={"offline_key": "q0"}, output_token_ids=[], stopped=False
    )
    obj.requests = {"req": request}
    return obj, request, tmp_path


def test_first_token_stops(scheduler):
    obj, request, root = scheduler
    with pytest.raises(RuntimeError, match="THREEWAY_FIRST_DIFFERENCE"):
        obj._update_request_with_output(request, [11])
    assert (root / "STOP").exists()
    assert json.loads((root / "FIRST_DIFFERENCE.json").read_text())["kind"] == "token"


def test_completed_request_checks_missing_event_tail(scheduler):
    obj, request, root = scheduler
    request.stopped = True
    with pytest.raises(RuntimeError, match="THREEWAY_FIRST_DIFFERENCE"):
        obj._update_request_with_output(request, [10, 20])
    assert json.loads((root / "FIRST_DIFFERENCE.json").read_text())["kind"] == "completed_acceptance_events"


def test_event_prefix_checks_keyword_signature(scheduler):
    obj, _, root = scheduler
    with pytest.raises(RuntimeError, match="THREEWAY_FIRST_DIFFERENCE"):
        obj.make_spec_decoding_stats(
            spec_decoding_stats=None,
            num_draft_tokens=5,
            num_accepted_tokens=1,
            num_invalid_spec_tokens=None,
            request_id="req",
        )
    assert json.loads((root / "FIRST_DIFFERENCE.json").read_text())["kind"] == "acceptance_event"


def test_matching_completion(scheduler):
    obj, request, root = scheduler
    obj.make_spec_decoding_stats(None, 5, 2, None, "req")
    request.stopped = True
    assert obj._update_request_with_output(request, [10, 20]) == ([10, 20], True)
    assert not (root / "STOP").exists()
