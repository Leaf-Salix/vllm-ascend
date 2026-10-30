# SPDX-License-Identifier: Apache-2.0
"""八卡测试适配只放开指定CLI与rank循环，不修改模型配置或数学。"""

import ast
import importlib.util
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
    spec = importlib.util.spec_from_file_location("eight_card_runtime", Path(__file__).with_name("runtime.py"))
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    configs = [SimpleNamespace(tensors=[SimpleNamespace(size=513), SimpleNamespace(size=512)])]
    assert runtime.check_kv_budget(configs, [2048])[0]["planned_bytes"] == 1536
    with pytest.raises(RuntimeError, match="KV_BUDGET_EXCEEDED"):
        runtime.check_kv_budget(configs, [1024])
    with pytest.raises(ValueError):
        runtime.check_kv_budget(configs, [])


def test_eight_card_script_rejects_unapproved_devices():
    script = Path(__file__).with_name("run_eight_card.sh").read_text()
    assert '[[ "$TASK_DEVICE" == 0,8,9,10,11,12,13,15 ]]' in script
    assert script.index('[[ "$TASK_DEVICE" ==') < script.index('source "$align_root/')
    assert "14,15" not in script
