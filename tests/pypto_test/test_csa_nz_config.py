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
    monkeypatch.setenv("DYNAMIC_EPLB", "true")
    monkeypatch.setenv("EXPERT_MAP_RECORD", "true")
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
    expected = mode if mode is not None else 2
    for rank, (cmd, env) in enumerate(children):
        assert env["VLLM_ASCEND_ENABLE_NZ"] == str(expected)
        assert env["DYNAMIC_EPLB"] == env["EXPERT_MAP_RECORD"] == "false"
        assert env["OMP_NUM_THREADS"] == "10"
        assert env["HCCL_BUFFSIZE"] == "1800"
        assert env["HCCL_OP_EXPANSION_MODE"] == "AIV"
        assert env["VLLM_BATCH_INVARIANT"] == "0"
        monkeypatch.setattr(sys, "argv", cmd[1:])
        run.main()
        parsed = worker.call_args.args[0]
        assert (parsed.rank, parsed.weight_nz_mode) == (rank, expected)


def test_direct_worker_binds_mode_before_vllm_import(monkeypatch):
    original_import = builtins.__import__
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_NZ", "0")
    monkeypatch.setenv("DYNAMIC_EPLB", "true")
    monkeypatch.setenv("EXPERT_MAP_RECORD", "true")

    class ReachedVllmImport(Exception):
        pass

    def intercepted(name, *args, **kwargs):
        if name == "vllm":
            assert os.environ["VLLM_ASCEND_ENABLE_NZ"] == "2"
            assert os.environ["DYNAMIC_EPLB"] == os.environ["EXPERT_MAP_RECORD"] == "false"
            raise ReachedVllmImport
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", intercepted)
    with pytest.raises(ReachedVllmImport):
        run.worker(SimpleNamespace(weight_nz_mode=2, rank=0, backend="pto"))


@pytest.mark.parametrize("level", [0, 1])
def test_offline_worker_sets_determinism_before_native_init_in_fresh_process(level):
    # 用真实 torch_npu 进程内设置验证 spawn 边界；替换 Native 构造，完全不初始化 NPU。
    code = r'''
import sys
from types import ModuleType, SimpleNamespace
import torch, torch_npu
level = int(sys.argv[1])
torch_npu.npu.set_deterministic_level(1 - level)
module = ModuleType("vllm_ascend.worker.worker")
class NativeWorker:
    def __init__(self, config):
        assert torch_npu.npu._get_deterministic_level() == level
        assert torch.are_deterministic_algorithms_enabled() == bool(level)
        self.model_runner = SimpleNamespace(dynamic_eplb=False, scheduler_config=SimpleNamespace(
            max_num_seqs=40, max_num_batched_tokens=400, max_num_scheduled_tokens=240))
        self.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=True, disable_hybrid_kv_cache_manager=False),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            compilation_config=SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY"))
module.NPUWorker = NativeWorker
sys.modules[module.__name__] = module
ascend_module = ModuleType("vllm_ascend.ascend_config")
ascend_module.get_ascend_config = lambda: SimpleNamespace(
    ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=True, enable_static_kernel=True, fuse_norm_quant=True),
    enable_cpu_binding=True, multistream_overlap_shared_expert=True,
    scheduler_config=SimpleNamespace(recompute_scheduler_enable=False))
sys.modules[ascend_module.__name__] = ascend_module
from offline_pd.worker import OfflineNPUWorker
worker = OfflineNPUWorker(SimpleNamespace(additional_config={"offline_deterministic_level": level}))
actual = worker.offline_runtime_config()
assert actual["deterministic_level"] == actual["requested_deterministic_level"] == level
assert actual["dynamic_eplb"] is False
assert actual["scheduler"]["max_num_scheduled_tokens"] == 240
assert actual["decode_optimizations"]["ascend_compilation_config"]["enable_static_kernel"] is True
assert actual["decode_optimizations"]["multistream_overlap_shared_expert"] is True
assert actual["engine"]["async_scheduling"] is True
assert not torch_npu.npu.is_initialized()
'''
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run([sys.executable, "-c", code, str(level)], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


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
