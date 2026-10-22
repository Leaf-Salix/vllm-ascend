# SPDX-License-Identifier: Apache-2.0
"""真实query上界与padding容量分开；支持判定不读取或规约任何Tensor。"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service_config import (
    native_query_bound,
    supported_descriptor,
)


def supported(bound):
    return supported_descriptor(
        tokens=64,
        requests=17,
        actual_tokens=55,
        max_query_tokens=bound,
        max_seq_len=131078,
        batch_capacity=40,
    )


def test_inactive_tail_does_not_expand_query_bound():
    # 16个真实请求最多S6，额外9行由失活请求承接；不需要任何query数组。
    request = SimpleNamespace(max_seqlen_q=9, max_seqlen_q_actual=6)
    assert native_query_bound(request) == 6
    assert request.max_seqlen_q == 9
    assert supported(native_query_bound(request))


def test_real_query_above_six_is_still_rejected():
    request = SimpleNamespace(max_seqlen_q=9, max_seqlen_q_actual=7)
    assert native_query_bound(request) == 7
    assert not supported(native_query_bound(request))


@pytest.mark.parametrize("actual", (None, "missing"))
def test_legacy_metadata_keeps_conservative_native_bound(actual):
    request = SimpleNamespace(max_seqlen_q=9)
    if actual is None:
        request.max_seqlen_q_actual = None
    assert native_query_bound(request) == 9
    assert not supported(native_query_bound(request))
    request.max_seqlen_q = 6
    assert supported(native_query_bound(request))


@pytest.mark.parametrize("actual", (0, -1, True, 6.0))
def test_invalid_actual_bound_is_not_converted_or_ignored(actual):
    request = SimpleNamespace(max_seqlen_q=6, max_seqlen_q_actual=actual)
    assert native_query_bound(request) is actual
    assert not supported(native_query_bound(request))


def test_tensor_bound_is_rejected_without_reading_values():
    actual = torch.empty((), device="meta", dtype=torch.int32)
    request = SimpleNamespace(max_seqlen_q=6, max_seqlen_q_actual=actual)
    assert native_query_bound(request) is actual
    assert not supported(native_query_bound(request))
