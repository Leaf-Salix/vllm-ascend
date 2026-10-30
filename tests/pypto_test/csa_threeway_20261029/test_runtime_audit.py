# SPDX-License-Identifier: Apache-2.0
"""只在227 CPU检查真实SDK配置字段及审计入口，不能冒充NPU验收。"""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from csa_threeway_20261029 import postflight, runtime_audit

HEAP = [256 << 20, 512 << 20, 256 << 20, 256 << 20]


def test_audit_uses_actual_sdk_configuration(monkeypatch):
    from pypto.runtime.kernel.abi import KernelConfig

    config = KernelConfig("a2a3", "tensormap_and_ringbuffer", 0, ring_heap=tuple(HEAP), ring_task_window=2048)
    monkeypatch.setattr(runtime_audit, "get_process_kernel_state", lambda: SimpleNamespace(config=config))
    actual = runtime_audit.RuntimeAuditObserver.offline_pto_window_config(object())
    assert actual["device_id"] == 0 and actual["ring_task_window"] == [2048] * 4
    assert actual["ring_heap"] == HEAP and actual["ring_dep_pool"] is None
    monkeypatch.setattr(runtime_audit, "get_process_kernel_state", lambda: SimpleNamespace(config=None))
    with pytest.raises(RuntimeError, match="initialized Worker"):
        runtime_audit.RuntimeAuditObserver.offline_pto_window_config(object())


def test_missing_or_different_effective_window_is_rejected():
    actual = {"ring_heap": HEAP, "ring_task_window": [2048] * 4, "ring_dep_pool": None}
    data = {"worker_runtime_config": [{"pto_kernel_config": actual}]}
    assert postflight.runtime_window_errors(data, 2048, HEAP) == []
    actual["ring_task_window"] = [16384] * 4
    assert postflight.runtime_window_errors(data, 2048, HEAP)
    assert postflight.runtime_window_errors({}, 2048, HEAP)
    assert postflight.runtime_window_errors({}, None, HEAP) == []  # Native没有PTO arena。


def test_audit_runner_preserves_original_frozen_source(tmp_path):
    spec = importlib.util.spec_from_file_location("audit_launch_test", Path(__file__).with_name("launch.py"))
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    root = tmp_path / "source"
    original = root / "tests/pypto_test/offline_pd/run.py"
    original.parent.mkdir(parents=True)
    observer = root / "tests/pypto_test/csa_threeway_20261029/observer.py"
    observer.parent.mkdir()
    observer.write_text("# frozen observer\n")
    scheduler = observer.with_name("scheduler.py")
    scheduler.write_text("# frozen strict scheduler\n")
    source = (
        b'config = {"worker_extension_cls": "csa_threeway_20261029.observer.TimedObserver"}\n'
        b'scheduler = {"scheduler_cls": "csa_threeway_20261029.scheduler.CheckedScheduler"}\n'
        b"def run():\n"
        b'    args.worker_runtime_config = llm.collective_rpc("offline_runtime_config")\n'
    )
    original.write_bytes(source)
    (root.parent / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (original, observer, scheduler)
                }
            }
        )
    )
    runner, audit = launch.prepare_runtime_audit_runner(root)
    assert original.read_bytes() == source
    assert runner != original and "offline_pto_window_config" in runner.read_text()
    assert launch.prepare_runtime_audit_runner(root) == (runner, audit)
    diagnostic, receipt = launch.prepare_first_step_runner(root)
    ast.parse(diagnostic.read_text())
    assert "FirstStepObserver" in diagnostic.read_text() and diagnostic != runner
    assert original.read_bytes() == source
    assert launch.prepare_first_step_runner(root) == (diagnostic, receipt)
    token_runner, token_receipt = launch.prepare_token_runner(root)
    ast.parse(token_runner.read_text())
    assert "TokenDiagnosticScheduler" in token_runner.read_text() and original.read_bytes() == source
    assert launch.prepare_token_runner(root) == (token_runner, token_receipt)
    runner.write_text("unexpected modification")
    with pytest.raises(AssertionError, match="refusing to overwrite"):
        launch.prepare_runtime_audit_runner(root)
    assert original.read_bytes() == source
    original.write_bytes(source + b"\n")
    with pytest.raises(AssertionError, match="frozen test source changed"):
        launch.prepare_runtime_audit_runner(root)
