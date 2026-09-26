# SPDX-License-Identifier: Apache-2.0
"""单层诊断的保护区范围，CPU 即可判定。"""

import pytest
import torch
from dsv4_csa_single_layer import guard_checks, writable_bytes


def test_slot_mask_preserves_page_padding_and_mixed_dtype_guards():
    raw = torch.full((256,), 37, dtype=torch.int8)
    key = raw.as_strided((2, 2, 1, 8), (64, 8, 8, 1), 64)
    scale = raw.view(torch.float16).as_strided((2, 2, 1, 1), (32, 1, 1, 1), 40)
    slots = torch.tensor([[1, 1], [-1, -1]], dtype=torch.int32)
    allowed = writable_bytes(raw, [key, scale], slots)
    assert allowed.nonzero().flatten().tolist() == list(range(136, 144)) + [146, 147]
    initial = raw.clone()
    key[1, 1].fill_(1)
    scale[1, 1].fill_(0.5)
    fixture = {"groups": {"indexer": {"allocation": raw, "initial": initial, "allowed": allowed}}}
    assert guard_checks(fixture)["indexer"]["status"] == "PASS"
    raw[148] = 0
    check = guard_checks(fixture)["indexer"]
    assert check["status"] == "FAIL" and check["first_outside"] == [148]


@pytest.mark.parametrize("slot", [[2, 0], [0, 2]])
def test_invalid_positive_slot_is_rejected(slot):
    raw = torch.zeros(256, dtype=torch.int8)
    view = raw.as_strided((2, 2, 1, 8), (64, 8, 8, 1), 64)
    with pytest.raises(ValueError, match="超出 Native 页视图"):
        writable_bytes(raw, [view], torch.tensor([slot]))
