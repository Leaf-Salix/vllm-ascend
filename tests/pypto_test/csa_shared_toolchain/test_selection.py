# SPDX-License-Identifier: Apache-2.0
"""公共 SDK 选择应跨主检出和嵌套 worktree 保持一致，不触及设备。"""

import importlib.util
import json
from pathlib import Path

import pytest


def load_selection():
    spec = importlib.util.spec_from_file_location("csa_selection_test", Path(__file__).with_name("selection.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("checkout", ["vllm-ascend-dsv4-pto-0251rc1", ".worktrees/csa-spmd-tnd"])
def test_all_checkouts_select_primary_manifest(tmp_path, monkeypatch, checkout):
    (tmp_path / "env-dsv4-0251rc1.sh").touch()
    relative = Path("tests/pypto_test/csa_shared_toolchain")
    primary = tmp_path / "vllm-ascend-dsv4-pto-0251rc1" / relative
    primary.mkdir(parents=True)
    manifest = primary / "manifest.json"
    data = {
        "release": "shared-current",
        "status": "validated",
        "pypto": {"source": ".worktrees/shared-pypto"},
        "ptoas": {"build": ".worktrees/shared-ptoas/build"},
        "simpler": {"source": ".worktrees/shared-simpler"},
    }
    manifest.write_text(json.dumps(data))
    expected = {
        name: tmp_path / data[name]["build" if name == "ptoas" else "source"] for name in ("pypto", "ptoas", "simpler")
    }
    for path in expected.values():
        path.mkdir(parents=True)
    entry = tmp_path / checkout / relative
    entry.mkdir(parents=True, exist_ok=True)
    if entry != primary:
        (entry / "manifest.json").write_text('{"release": "stale-private-copy"}')
    module = load_selection()
    monkeypatch.setattr(module, "__file__", str(entry / "selection.py"))
    actual, roots, selected = module.selected_toolchain()
    assert actual == data
    assert roots == expected
    assert selected == manifest
    expected["ptoas"].rmdir()
    with pytest.raises(RuntimeError, match="缺少 ptoas"):
        module.selected_toolchain()


def test_missing_workspace_does_not_select_an_unrelated_sdk(tmp_path, monkeypatch):
    module = load_selection()
    monkeypatch.setattr(module, "__file__", str(tmp_path / "selection.py"))
    with pytest.raises(RuntimeError, match="无法.*定位公共 CSA 工作区"):
        module.selected_toolchain()
