# SPDX-License-Identifier: Apache-2.0
"""性能入口默认生效；旧精度配置不得静默使用另一套算术。"""

import pytest

from vllm_ascend.ops.pypto.variant import selected_variant, variant_package


@pytest.mark.parametrize("value", ["spmd_tnd", "SPMD_TND", "pkg:deepseek_v4_flash_csa_spmd_tnd"])
def test_explicit_tnd_variant(monkeypatch, value):
    monkeypatch.setenv("PTO_CSA_VARIANT", value)
    assert selected_variant() == "spmd_tnd"
    assert variant_package() == "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd"


@pytest.mark.parametrize("value", [None, "performance", " PERF "])
def test_default_and_compatible_names_use_performance(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PTO_CSA_VARIANT", raising=False)
    else:
        monkeypatch.setenv("PTO_CSA_VARIANT", value)
    assert selected_variant() == "performance"
    assert variant_package() == "vllm_ascend.ops.pypto.deepseek_v4_flash_csa"


@pytest.mark.parametrize("value", ["precision", " PREC ", "pkg:deepseek_v4_flash_dspark"])
def test_archived_precision_is_rejected(monkeypatch, value):
    monkeypatch.setenv("PTO_CSA_VARIANT", value)
    with pytest.raises(ValueError, match="已封存"):
        selected_variant()


def test_private_performance_copy_is_preserved(monkeypatch):
    monkeypatch.setenv("PTO_CSA_VARIANT", "pkg:csa_perf_probe")
    assert selected_variant() == "performance"
    assert variant_package() == "vllm_ascend.ops.pypto.csa_perf_probe"


@pytest.mark.parametrize("value", ["unknown", "pkg:", "pkg:../precision"])
def test_invalid_selection_is_rejected(monkeypatch, value):
    monkeypatch.setenv("PTO_CSA_VARIANT", value)
    with pytest.raises(ValueError):
        variant_package()
