# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for capture-time prefetch selection and preserved aliases."""

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
    "csa_tuned_selection_cpu",
    "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd_tuned/selection.py",
)
VARIANT = load_file("csa_tuned_variant_cpu", "vllm_ascend/ops/pypto/variant.py")
ENVS = load_file("csa_tuned_envs_cpu", "vllm_ascend/envs.py")


def test_tuned_routes_physical_rows_and_preserves_short_root():
    calls = []
    small, combined = object(), object()

    class Baseline:
        def for_tokens(self, tokens):
            calls.append(tokens)
            return small if tokens <= 16 else combined

    tuned = SimpleNamespace(_schema="same-64-tensor-schema")
    router = SELECTION.CaptureAttention(Baseline(), tuned)
    for tokens in range(1, 257):
        expected = small if tokens <= 16 else combined if tokens < 96 else tuned
        assert router.for_tokens(tokens) is expected
    assert calls == list(range(1, 96))
    assert router._schema == tuned._schema


@pytest.mark.parametrize("tokens", [16, 17, 95, 96, 144, 240])
def test_tuned_dispatches_once_with_original_arguments(tokens):
    calls = []

    def selected(*args, **kwargs):
        calls.append((args, kwargs))
        return "output"

    baseline = SimpleNamespace(for_tokens=lambda rows: selected)
    selected._schema = "same-64-tensor-schema"
    router = SELECTION.CaptureAttention(baseline, selected)
    hidden, tail = SimpleNamespace(shape=(tokens, 4, 4096)), object()
    assert router(hidden, tail, marker=7) == "output"
    assert calls == [((hidden, tail), {"marker": 7})]


@pytest.mark.parametrize("tokens", [-1, 0, 257])
def test_tuned_rejects_rows_outside_accepted_capacity(tokens):
    with pytest.raises(ValueError, match="1..256"):
        SELECTION.CaptureAttention(object(), object()).for_tokens(tokens)


def test_tuned_explicit_alias_keeps_accepted_combined(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD", raising=False)
    for alias, suffix in (
        ("tnd_precision_tuned", "spmd_tuned"),
        ("tnd_precision_combined", "spmd_combined"),
    ):
        monkeypatch.setenv("PTO_CSA_VARIANT", alias)
        assert VARIANT.selected_variant() == alias
        assert VARIANT.variant_package().endswith(suffix)
        assert ENVS._pto_csa_atomic_add() == 0
    monkeypatch.delenv("PTO_CSA_VARIANT")
    assert VARIANT.selected_variant() == "precision"
    assert ENVS._pto_csa_atomic_add() == 1
