# SPDX-License-Identifier: Apache-2.0
"""CPU 回归：真实启动器传参、导入前绑定、两版根布局及 Native 矩阵方向一致。"""

import builtins
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from offline_pd import run


@pytest.mark.parametrize("mode", [None, 0, 1, 2])
def test_launcher_forwards_mode_to_every_rank(mode, monkeypatch, tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "audit.json").write_text('{"status": "PASS"}')
    monkeypatch.setenv("TASK_DEVICE", ",".join(map(str, range(16))))
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_NZ", "2" if mode != 2 else "0")
    monkeypatch.setattr(run.signal, "signal", Mock())
    monkeypatch.setattr(run.os, "killpg", Mock())
    children = []

    def popen(cmd, **kwargs):
        children.append((cmd, kwargs["env"]))
        return SimpleNamespace(pid=123456, returncode=0, poll=lambda: 0, wait=lambda **_: 0)

    monkeypatch.setattr(run.subprocess, "Popen", popen)
    argv = ["run.py", "decode", "--bank", str(bank), "--output", str(tmp_path / "out")]
    if mode is not None:
        argv += ["--weight-nz-mode", str(mode)]
    monkeypatch.setattr(sys, "argv", argv)
    run.main()
    assert len(children) == 16
    worker = Mock()
    monkeypatch.setattr(run, "worker", worker)
    expected = mode if mode is not None else 0
    for rank, (cmd, env) in enumerate(children):
        assert env["VLLM_ASCEND_ENABLE_NZ"] == str(expected)
        monkeypatch.setattr(sys, "argv", cmd[1:])
        run.main()
        parsed = worker.call_args.args[0]
        assert (parsed.rank, parsed.weight_nz_mode) == (rank, expected)


def test_direct_worker_binds_mode_before_vllm_import(monkeypatch):
    original_import = builtins.__import__
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_NZ", "0")

    class ReachedVllmImport(Exception):
        pass

    def intercepted(name, *args, **kwargs):
        if name == "vllm":
            assert os.environ["VLLM_ASCEND_ENABLE_NZ"] == "2"
            raise ReachedVllmImport
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", intercepted)
    with pytest.raises(ReachedVllmImport):
        run.worker(SimpleNamespace(weight_nz_mode=2, rank=0, backend="pto"))


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_real_roots_match_native_shapes_layouts_and_reject_conflicts(mode):
    # 三个新进程分别导入真实根函数，覆盖模块加载时固定布局的语义；不初始化 NPU。
    code = r'''
import importlib
import os
import torch
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark import nz_mode

mode = int(os.environ["VLLM_ASCEND_ENABLE_NZ"])
nz_mode.validate_weight_nz_mode(mode)
for suffix in ("", "_perf"):
    module = importlib.import_module(f"vllm_ascend.ops.pypto.deepseek_v4_flash_dspark{suffix}.decode_csa")
    root = module._decode_csa_tp1_layer
    expected_nz = ({"wq_a", "wo_a"} if mode == 2 else set())
    if mode >= 1:
        expected_nz.update(("wq_b", "wo_b"))
    layouts = nz_mode.root_weight_layouts(root)
    assert {key for key, value in layouts.items() if value == "NZ"} == expected_nz
    assert nz_mode.root_weight_shapes(root) == {
        "wq_a": (1024, 4096), "wq_b": (1024, 32768),
        "wo_a": (8, 4096, 1024), "wo_b": (8192, 4096),
    }

try:
    nz_mode.validate_weight_nz_mode((mode + 1) % 3)
except ValueError as exc:
    assert "AscendConfig=" in str(exc)
else:
    raise AssertionError("model/layout mismatch was accepted")
os.environ["VLLM_ASCEND_ENABLE_NZ"] = str((mode + 1) % 3)
try:
    nz_mode.validate_weight_nz_mode(mode)
except ValueError as exc:
    assert "imported_layout=" in str(exc)
else:
    raise AssertionError("environment changed after import was accepted")
'''
    env = dict(os.environ, VLLM_ASCEND_ENABLE_NZ=str(mode), TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
