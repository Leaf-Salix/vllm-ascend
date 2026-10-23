# SPDX-License-Identifier: Apache-2.0
"""CPU-only capture routing, alias and fixed-reduction contracts."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SELECTION = load_file(
    "csa_combined_selection_cpu",
    "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd_combined/selection.py",
)
VARIANT = load_file("csa_combined_variant_cpu", "vllm_ascend/ops/pypto/variant.py")
ENVS = load_file("csa_combined_envs_cpu", "vllm_ascend/envs.py")


@pytest.mark.parametrize("tokens", [1, 4, 6, 16, 17, 21, 126, 144, 192, 240, 256])
def test_combined_routes_one_complete_root(tokens):
    calls = []

    def small(*args, **kwargs):
        calls.append(("small", args, kwargs))
        return "small-output"

    def combined(*args, **kwargs):
        calls.append(("combined", args, kwargs))
        return "combined-output"

    small._schema = combined._schema = "same-64-tensor-schema"
    selected = SELECTION.CaptureAttention(small, combined)
    hidden, tail = SimpleNamespace(shape=(tokens, 4, 4096)), object()
    expected = "small" if tokens <= 16 else "combined"
    assert selected(hidden, tail, marker=7) == expected + "-output"
    assert calls == [(expected, (hidden, tail), {"marker": 7})]
    assert selected._schema == combined._schema


@pytest.mark.parametrize("tokens", [-1, 0, 257])
def test_combined_rejects_out_of_contract_rows(tokens):
    with pytest.raises(ValueError, match="1..256"):
        SELECTION.CaptureAttention(object(), object()).for_tokens(tokens)


def test_combined_has_explicit_alias_and_fixed_reduction(monkeypatch):
    monkeypatch.setenv("PTO_CSA_VARIANT", "tnd_precision_combined")
    monkeypatch.delenv("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD", raising=False)
    assert VARIANT.selected_variant() == "tnd_precision_combined"
    assert VARIANT.variant_package().endswith(".deepseek_v4_flash_csa_tnd_precision_spmd_combined")
    assert ENVS._pto_csa_atomic_add() == 0
    monkeypatch.setenv("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD", "1")
    assert ENVS._pto_csa_atomic_add() == 1  # Runtime initialization rejects this explicit incompatible choice.


def test_combined_keeps_legacy_default(monkeypatch):
    monkeypatch.delenv("PTO_CSA_VARIANT", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD", raising=False)
    assert VARIANT.selected_variant() == "precision"
    assert VARIANT.variant_package().endswith(".deepseek_v4_flash_dspark")
    assert ENVS._pto_csa_atomic_add() == 1
