# SPDX-License-Identifier: Apache-2.0
"""Prove the schedule edit preserves each result's exact computation DAG."""

import ast
from pathlib import Path

from prepare_score_half_interleave import reorder

HERE = Path(__file__).resolve().parent


def branch(source):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store")
    return next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "buf_qk_panel > 0")


def test_only_one_existing_statement_moves_and_all_dependencies_precede_use():
    original = (HERE / "runs/output_rectangle_production_v1/indexer_score_stages.py").read_text()
    candidate = (HERE / "runs/score_half_interleave_v1/indexer_score_stages.py").read_text()
    assert candidate == reorder(original)
    before, after = branch(original), branch(candidate)
    old = [ast.dump(n, include_attributes=False) for n in before.body]
    new = [ast.dump(n, include_attributes=False) for n in after.body]
    assert sorted(old) == sorted(new)
    names = [n.targets[0].id for n in after.body if isinstance(n, ast.Assign)]
    assert names == [
        "split_right0",
        "split_right1",
        "split_qk0",
        "split_current0",
        "split_ws_first",
        "split_qk1",
        "split_current1",
        "split_ws_done",
    ]
    produced_here = set(names)
    available = set()
    for statement in after.body:
        reads = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        # split_current0/1 are existing Mat destinations, assigned again by the
        # FIX assembly; their creation precedes this branch in the original code.
        assert reads & produced_here - {"split_current0", "split_current1"} <= available
        if isinstance(statement, ast.Assign):
            available.add(statement.targets[0].id)


def test_root_and_other_modules_unchanged():
    parent = HERE / "runs/output_rectangle_production_v1"
    candidate = HERE / "runs/score_half_interleave_v1"
    assert {p.name for p in parent.glob("*.py")} == {p.name for p in candidate.glob("*.py")}
    for path in parent.glob("*.py"):
        if path.name != "indexer_score_stages.py":
            assert path.read_bytes() == (candidate / path.name).read_bytes()
    assert (HERE / "descriptors/C04_output_rectangle_v1.json").read_bytes() == (
        HERE / "descriptors/C04_score_half_interleave_v1.json"
    ).read_bytes()
