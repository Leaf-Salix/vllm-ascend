# SPDX-License-Identifier: Apache-2.0
"""八卡测试适配只放开指定CLI与rank循环，不修改模型配置或数学。"""

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("eight_card_prepare", Path(__file__).with_name("prepare.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_eight_card_snapshot_adaptation(tmp_path):
    source = Path(__file__).resolve().parents[3]
    for relative in ("offline_pd/run.py", "low_acceptance_20261001/mixed.py"):
        original = source / "tests/pypto_test" / relative
        target = tmp_path / "tests/pypto_test" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
    MODULE.adapt_eight_card_tests(tmp_path)
    runner = (tmp_path / "tests/pypto_test/offline_pd/run.py").read_text()
    mixed = (tmp_path / "tests/pypto_test/low_acceptance_20261001/mixed.py").read_text()
    ast.parse(runner)
    ast.parse(mixed)
    assert "args.decode_dp not in (8, 16)" in runner
    assert 'args.pto_attention not in ("csa", "hca", "both")' in runner
    assert mixed.count("range(8)") == 2 and "range(16)" not in mixed
    # Bank export retains its original DP16 contract; no synthetic expert/model changes.
    assert "args.decode_tokens != 1 or args.decode_dp != 16" in runner
    with pytest.raises(AssertionError):
        MODULE.adapt_eight_card_tests(tmp_path)


def test_capacity_guard_rejects_override_before_allocation():
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

    spec = importlib.util.spec_from_file_location("eight_card_runtime", Path(__file__).with_name("runtime.py"))
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    configs = [
        KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(size=513, shared_by=["layer0"]),
                KVCacheTensor(size=512, shared_by=["layer1"]),
            ],
            kv_cache_groups=[],
        )
    ]
    assert runtime.check_kv_budget(configs, [2048])[0]["planned_bytes"] == 1536
    with pytest.raises(RuntimeError, match="KV_BUDGET_EXCEEDED"):
        runtime.check_kv_budget(configs, [1024])
    with pytest.raises(ValueError):
        runtime.check_kv_budget(configs, [])
    assert runtime.check_kv_budget(configs, [5 << 20], 2 << 20)[0]["planned_bytes"] == 4195840
    with pytest.raises(RuntimeError, match="KV_BUDGET_EXCEEDED"):
        runtime.check_kv_budget(configs, [4 << 20], 2 << 20)


def test_eight_card_script_rejects_unapproved_devices():
    script = Path(__file__).with_name("run_eight_card.sh").read_text()
    assert '[[ "$TASK_DEVICE" == 8,9,10,11,12,13,14,15 ]]' in script
    assert script.index('[[ "$TASK_DEVICE" ==') < script.index('source "$align_root/')
    assert "0,8,9" not in script


def test_forty_inputs_use_five_eight_request_waves(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[3] / "tests/pypto_test/low_acceptance_20261001/mixed.py"
    target = tmp_path / "tests/pypto_test/low_acceptance_20261001/mixed.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    MODULE.adapt_request_waves(tmp_path, 8)
    spec = importlib.util.spec_from_file_location("eight_card_mixed_test", target)
    mixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mixed)
    calls, batches, offsets, saved = [], [], [], []
    mixed.wait_for_generation_peers = lambda output, rank, offset: offsets.append(offset)

    def generate(_llm, prompts, params):
        batches.append(len(prompts))
        return [
            SimpleNamespace(
                request_id=str(len(batches) * 8 + i),
                finished=True,
                outputs=[SimpleNamespace(token_ids=[1] * 192, text="", finish_reason="length", stop_reason=None)],
            )
            for i in range(len(prompts))
        ]

    for name, attributes in {
        "offline_pd.batch": {"generate_aligned_batch": generate},
        "offline_pd.run": {
            "spec_decode_metrics": lambda llm: {},
            "write_json": lambda path, value: saved.append(value),
        },
        "vllm": {"SamplingParams": lambda **kwargs: SimpleNamespace(**kwargs)},
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    class LLM:
        def collective_rpc(self, name):
            calls.append(name)
            return ["observation"] if name == "offline_end_observation" else []

        def get_tokenizer(self):
            return SimpleNamespace(eos_token_id=1)

    (tmp_path / "tokens.json").write_text(json.dumps([1, 2, 3]))
    args = SimpleNamespace(
        save_extended_bank=False,
        batch=40,
        bank=tmp_path,
        rank=0,
        decode_tokens=192,
        output=tmp_path,
        backend="native",
        pto_attention="csa",
        worker_runtime_config={},
        deterministic=True,
    )
    cases = [{"key": str(i), "question_kind": "unit", "tokens": "tokens.json"} for i in range(40)]
    mixed.run_batches(LLM(), args, cases)
    assert batches == [8] * 5 and offsets == [0, 8, 16, 24, 32]
    assert calls.count("offline_begin_observation") == calls.count("offline_end_observation") == 1
    assert len(saved[-1]["requests"]) == 40 and saved[-1]["observation"] == ["observation"]
