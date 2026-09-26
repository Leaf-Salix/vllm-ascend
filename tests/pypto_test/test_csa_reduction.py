# SPDX-License-Identifier: Apache-2.0
"""规约策略必须在导入/编译前固定，不能在图重放期间切换。"""

import runpy
from pathlib import Path

import pytest

POLICY = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/reduction.py"


@pytest.mark.parametrize("mode", [0, 1])
def test_reduction_mode_is_frozen_after_import(monkeypatch, mode):
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD", str(mode))
    policy = runpy.run_path(str(POLICY))
    assert policy["ATOMIC_ADD"] == mode
    policy["validate_reduction_mode"]()
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD", str(1 - mode))
    with pytest.raises(ValueError, match="新进程"):
        policy["validate_reduction_mode"]()


def test_unknown_reduction_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD", "2")
    with pytest.raises(ValueError, match="只接受"):
        runpy.run_path(str(POLICY))
