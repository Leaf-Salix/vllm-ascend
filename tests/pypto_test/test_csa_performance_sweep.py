# SPDX-License-Identifier: Apache-2.0
"""CPU 回归：复用加载时实际 batch 不串档，DSpark 使用本档计数增量。"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from offline_pd import run


def test_sweep_forwarding_and_actual_batch(monkeypatch, tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "audit.json").write_text('{"status": "PASS"}')
    (bank / "plan.json").write_text(json.dumps({
        "model": run.FORMAL_MODEL, "decode": {"speculative_tokens": 5}}))
    monkeypatch.setenv("TASK_DEVICE", ",".join(map(str, range(16))))
    monkeypatch.setattr(run.signal, "signal", Mock())
    monkeypatch.setattr(run.os, "killpg", Mock())
    commands = []

    def popen(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(pid=123456, returncode=0, poll=lambda: 0, wait=lambda **_: 0)

    monkeypatch.setattr(run.subprocess, "Popen", popen)
    batches = [4, 8, 16, 24, 32, 40]
    monkeypatch.setattr(sys, "argv", [
        "run.py", "performance", "--bank", str(bank), "--output", str(tmp_path / "pto"),
        "--backend", "pto", "--batch", "40", "--decode-tokens", "192", "--max-num-batched-tokens", "400",
        "--sweep-batches", *map(str, batches), "--capture-sizes", *map(str, [b * 6 for b in batches])])
    run.main()
    assert len(commands) == 16
    worker = Mock()
    monkeypatch.setattr(run, "worker", worker)
    for rank, cmd in enumerate(commands):
        monkeypatch.setattr(sys, "argv", cmd[1:])
        run.main()
        parsed = worker.call_args.args[0]
        assert parsed.rank == rank
        assert parsed.max_num_batched_tokens == 400
        cases = list(run.diagnostic_runs(parsed))
        assert [item.batch for item in cases] == batches
        assert all(item.rank_batch == item.batch and item.max_num_seqs == 40 for item in cases)
        assert [item.output for item in cases] == [tmp_path / f"b{b}" / "pto" for b in batches]
        assert parsed.batch == parsed.rank_batch == 40


def test_dspark_budget_preserves_full_verify_batch():
    # 调用实际 vLLM 配置方法，覆盖旧 256 预算只能调度 96 token 的失败原因。
    from vllm.config import SpeculativeConfig, VllmConfig

    speculative = object.__new__(SpeculativeConfig)
    speculative.method = "dspark"
    speculative.parallel_drafting = True
    speculative.num_speculative_tokens = 5
    args = SimpleNamespace(command="performance", batch=40, max_num_batched_tokens=None)
    plan = {"decode": {"speculative_tokens": 5}}
    for budget, scheduled in ((256, 96), (run.token_budget(args, plan), 240)):
        scheduler = SimpleNamespace(max_num_seqs=40, max_num_batched_tokens=budget,
                                    max_num_scheduled_tokens=None)
        config = SimpleNamespace(speculative_config=speculative, scheduler_config=scheduler)
        VllmConfig._set_max_num_scheduled_tokens(config)
        assert scheduler.max_num_scheduled_tokens == scheduled
    args.max_num_batched_tokens = 256
    assert run.token_budget(args, plan) == 256  # 可明确复现已经采集的 128K B4/8/16。


def metrics(drafts, accepted):
    counts = {"num_drafts": drafts, "num_draft_tokens": drafts * 5,
              "num_accepted_tokens": sum(accepted)}
    return [SimpleNamespace(name=f"vllm:spec_decode_{key}", value=value)
            for key, value in counts.items()] + [SimpleNamespace(
                name="vllm:spec_decode_num_accepted_tokens_per_pos", values=accepted)]


def test_sweep_stats_use_delta_and_recompute_rates():
    llm = SimpleNamespace(get_metrics=Mock(side_effect=[[], metrics(10, [8] * 5),
                                                       metrics(30, [26, 26, 26, 20, 20])]))
    empty = run.spec_decode_metrics(llm)
    first = run.spec_decode_metrics(llm, baseline=empty)
    second = run.spec_decode_metrics(llm, baseline=first)
    assert first["counters"]["num_drafts"] == 10
    assert second["counters"] == {"num_drafts": 20, "num_draft_tokens": 100, "num_accepted_tokens": 78}
    assert second["num_accepted_tokens_per_pos"] == [18, 18, 18, 12, 12]
    assert second["acceptance_rate"] == 0.78
    assert first["num_accepted_tokens_per_pos"] == [8] * 5


def test_sweep_rejects_reset_stats():
    llm = SimpleNamespace(get_metrics=Mock(side_effect=[metrics(10, [8] * 5), metrics(5, [4] * 5)]))
    previous = run.spec_decode_metrics(llm)
    with pytest.raises(ValueError, match="累计计数重置"):
        run.spec_decode_metrics(llm, baseline=previous)
