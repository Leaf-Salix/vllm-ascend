# SPDX-License-Identifier: Apache-2.0
"""直接加载两个纯CPU策略文件；图容量不受有效query长度乘积限制。"""

import importlib
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def policy_source(monkeypatch):
    package = types.ModuleType("threeway_padding_policy")
    package.__path__ = [
        str(Path(__file__).resolve().parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd")
    ]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in ("policy", "service_config"):
        monkeypatch.delitem(sys.modules, package.__name__ + "." + name, raising=False)
    return (
        importlib.import_module(package.__name__ + ".policy"),
        importlib.import_module(package.__name__ + ".service_config"),
    )


@pytest.mark.parametrize("capacity,requests,query,actual", [(32, 4, 6, 9), (240, 40, 1, 40), (240, 40, 6, 184)])
def test_padding_capacity_does_not_change_real_query_bound(policy_source, capacity, requests, query, actual):
    policy, config = policy_source
    descriptors = dict(tokens=capacity, requests=requests, max_query_tokens=query, max_seq_len=8198)
    assert config.supported_descriptor(**descriptors, actual_tokens=actual, batch_capacity=40)
    selected = policy.select_policy(**descriptors)
    assert selected.constexprs()


@pytest.mark.parametrize(
    "capacity,requests,query,actual", [(32, 4, 7, 9), (32, 4, 6, 25), (240, 40, 6, 241), (257, 40, 6, 40)]
)
def test_invalid_real_query_and_counts_still_rejected(policy_source, capacity, requests, query, actual):
    _, config = policy_source
    assert not config.supported_descriptor(
        tokens=capacity,
        requests=requests,
        max_query_tokens=query,
        max_seq_len=8198,
        actual_tokens=actual,
        batch_capacity=40,
    )
