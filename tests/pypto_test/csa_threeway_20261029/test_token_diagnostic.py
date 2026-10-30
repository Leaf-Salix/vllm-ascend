# SPDX-License-Identifier: Apache-2.0
"""227 CPU区分接受事件与输出token；原严格postflight绝不能被重写。"""

import json
from types import SimpleNamespace

import pytest
from csa_threeway_20261029 import postflight, token_diagnostic


def test_event_only_diagnostic_preserves_fatal_token_and_length_checks(tmp_path, monkeypatch):
    probe = object.__new__(token_diagnostic.TokenDiagnosticScheduler)
    probe._threeway_output = tmp_path
    probe.parallel_config = SimpleNamespace(data_parallel_rank=0)

    def fatal(_self, kind, *_args):
        raise RuntimeError("strict " + kind)

    monkeypatch.setattr(token_diagnostic.CheckedScheduler, "_fail", fatal)
    probe._fail("acceptance_event", "q04", 35, [5, 5], [5, 4])
    path = tmp_path / "first_acceptance_difference_rank0.json"
    original = path.read_bytes()
    probe._fail("completed_acceptance_events", "q04", 48, [], [])
    assert path.read_bytes() == original
    assert not (tmp_path / "STOP").exists() and not (tmp_path / "FIRST_DIFFERENCE.json").exists()
    for kind in ("token", "output_length"):
        with pytest.raises(RuntimeError, match="strict " + kind):
            probe._fail(kind, "q04", 143, 17, 18)


def test_final_token_comparison_does_not_turn_strict_event_failure_into_pass(tmp_path):
    actual, native = tmp_path / "actual", tmp_path / "native"
    actual.mkdir()
    native.mkdir()
    strict = {
        "passed": False,
        "coverage_errors": [],
        "precision_errors": ["acceptance events differ"],
        "acceptance": {"total_preemptions": 0},
    }
    strict_path = actual / "postflight.json"
    strict_path.write_text(json.dumps(strict))
    strict_bytes = strict_path.read_bytes()
    rows = [{"key": f"q{i}", "finished": True, "output_token_ids": [i] * 192} for i in range(40)]
    for rank in range(8):
        for folder in (actual, native):
            (folder / f"rank{rank}.mixed.json").write_text(json.dumps({"requests": rows}))
            event = [5, 5] if folder == actual else [5, 4]
            (folder / f"request_stats_rank{rank}.jsonl").write_text(
                json.dumps({"key": "q04", "events": [event]}) + "\n"
            )
    result = postflight.summarize_tokens(actual, native, 8)
    assert result["tokens_equal"] and result["compared_tokens"] == 61440
    assert not result["acceptance_events_equal"] and not result["strict_postflight_passed"]
    assert strict_path.read_bytes() == strict_bytes
    rows[4]["output_token_ids"][17] = 999
    (actual / "rank0.mixed.json").write_text(json.dumps({"requests": rows}))
    assert not postflight.summarize_tokens(actual, native, 8)["tokens_equal"]
