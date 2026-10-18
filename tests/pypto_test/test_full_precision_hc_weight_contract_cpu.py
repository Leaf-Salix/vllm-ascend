# SPDX-License-Identifier: Apache-2.0
"""CPU门禁覆盖真实失败模式，不以误差阈值代替精确拆分。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_full_precision/hc_pre_contract.py"
)
_SPEC = importlib.util.spec_from_file_location("full_precision_hc_weight_contract", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
qualify_hc_pre_weight = _MODULE.qualify_hc_pre_weight


@pytest.mark.parametrize(
    "value", [0.0, -0.0, 2.0**-48, -(2.0**-48), 2.0**40, -(2.0**40), 3.0231372960543013e-13, 1.916015625]
)
def test_supported_original_weight_is_unchanged(value):
    w = torch.zeros((24, 16384), dtype=torch.float32)
    w[0, 0] = value
    before = w.clone().view(torch.int32)
    q = qualify_hc_pre_weight(w, "L14")
    assert torch.equal(w.view(torch.int32), before)
    assert q["version"] == 1 and len(q["raw_sha256"]) == 64


@pytest.mark.parametrize(
    "bits", [0x7F800000, 0xFF800000, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FC00001, 0x7F7FFFFF, 0x26FFF000, 0x53801000]
)
def test_reject_nonfinite_or_rounded_outside_contract(bits):
    w = torch.zeros((24, 16384), dtype=torch.float32)
    w.numpy().view(np.uint32)[0, 0] = bits
    with pytest.raises(ValueError, match="HC-pre"):
        qualify_hc_pre_weight(w, "L14")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32])
def test_reject_non_native_dtype(dtype):
    with pytest.raises(ValueError):
        qualify_hc_pre_weight(torch.zeros((24, 16384), dtype=dtype), "L14")


def test_reject_shape():
    with pytest.raises(ValueError):
        qualify_hc_pre_weight(torch.zeros((1, 16384)), "L14")
