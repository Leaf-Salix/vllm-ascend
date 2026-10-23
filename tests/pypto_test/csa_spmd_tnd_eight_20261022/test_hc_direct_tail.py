# SPDX-License-Identifier: Apache-2.0
"""Ensure the tail experiment removes only GM copies, retaining all arithmetic."""

import ast
import copy
from pathlib import Path

import pytest
from test_attention_overlap import function

HERE = Path(__file__).resolve().parent
BASE = HERE / "runs/physical_capacity_production_v1"


def tail_branch(fn, condition):
    nodes = [node for node in ast.walk(fn) if isinstance(node, ast.If) and ast.unparse(node.test) == condition]
    assert len(nodes) == 1
    return nodes[0].orelse


@pytest.mark.parametrize("version", [1, 2])
def test_widen_preserves_fillpad_reduction_order_and_output_coordinates(version):
    source = HERE / f"runs/hc_direct_tail_v{version}"
    old = function(BASE / "hc_stages.py", "hc_widen_rms_range")
    new = function(source / "hc_stages.py", "hc_widen_rms_range")
    before, after = tail_branch(old, "w_rows == T_TILE"), tail_branch(new, "w_rows == T_TILE")
    assert ast.unparse(before[-3].targets[0]) == "widen_tail[0:T_TILE, w_d0:w_d0 + RMS_K_TILE]"
    assert ast.unparse(before[-2].value.func) == "pl.load"
    assert ast.unparse(before[-1]) == "pl.store(w_out, [w_t0, w_d0], x_hc32_flat)"
    expected = (
        "pl.store(w_valid, [w_t0, w_d0], x_hc32_flat)"
        if version == 1
        else "x_hc32_flat = pl.assemble(x_hc32_flat, w_valid, [w_t0, w_d0])"
    )
    assert ast.unparse(after[-1]) == expected
    assert ast.unparse(after[0]) == "w_valid = pl.set_validshape(w_val, w_rows, RMS_K_TILE)"
    before[-3:] = copy.deepcopy(after[-1:])
    if version == 2:
        assert ast.unparse(new.body[-1]) == "return (x_hc32_flat, inv_rms)"
        assert any(isinstance(n, ast.Assign) and ast.unparse(n) == "x_hc32_flat = x_fp32" for n in new.body)
        old.body[-1] = copy.deepcopy(new.body[-1])
    assert ast.dump(old) == ast.dump(new)


@pytest.mark.parametrize("version", [1, 2])
def test_normalization_preserves_rounding_rms_and_full_row_path(version):
    source = HERE / f"runs/hc_direct_tail_v{version}"
    old = function(BASE / "hc_stages.py", "hc_mix_norm_range")
    new = function(source / "hc_stages.py", "hc_mix_norm_range")
    before, after = tail_branch(old, "valid_rows == T_TILE"), tail_branch(new, "valid_rows == T_TILE")
    assert ast.unparse(before[0].targets[0]) == "x_normed_tail_store[0:T_TILE, d0:d0 + D_TILE]"
    assert ast.unparse(before[1].value.func) == "pl.load"
    assert ast.unparse(after[0]) == "normed_valid = pl.set_validshape(normed_bf16, valid_rows, D_TILE)"
    before[:2] = copy.deepcopy(after[:1])
    if version == 2:
        assert ast.unparse(after[-1]) == "x_normed = pl.assemble(x_normed, normed_valid, [t0, d0])"
        before[-1] = copy.deepcopy(after[-1])
    assert ast.dump(old) == ast.dump(new)


@pytest.mark.parametrize("version", [1, 2])
def test_all_other_source_and_helper_bodies_remain_unchanged(version):
    source = HERE / f"runs/hc_direct_tail_v{version}"
    assert {p.name for p in source.glob("*.py") if p.read_bytes() != (BASE / p.name).read_bytes()} == {"hc_stages.py"}
    before = ast.parse((BASE / "hc_stages.py").read_text())
    after = ast.parse((source / "hc_stages.py").read_text())
    changed = {"hc_widen_rms_range", "hc_mix_norm_range"}
    remaining = lambda tree: [
        ast.dump(n) for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name not in changed
    ]
    assert remaining(before) == remaining(after)
