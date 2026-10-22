# SPDX-License-Identifier: Apache-2.0
"""Preserve the full quantization, valid-row, ownership and zero-tail AST."""

import ast
from pathlib import Path

from prepare_output_quant_reuse import transform
from test_query_resident_weights import functions


def test_only_duplicate_load_changes_and_whole_math_and_tail_remain():
    here = Path(__file__).resolve().parent
    source = (here / "runs/frozen_v122/output_projection_a_stages.py").read_text()
    old, new = functions(source), functions(transform(source))
    name = "o_projection_groups_quant_stage"
    for key in old:
        if key != name:
            assert ast.dump(old[key], include_attributes=False) == ast.dump(new[key], include_attributes=False)
    fn = new[name]
    first = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "oc_amax")
    second = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "oc_q")
    assert ast.unparse(second.value) == "oc_amax"
    second.value = first.value
    assert ast.dump(fn, include_attributes=False) == ast.dump(old[name], include_attributes=False)
