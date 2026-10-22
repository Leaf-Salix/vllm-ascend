# SPDX-License-Identifier: Apache-2.0
"""Check exactly-once KV production/publication and unchanged mathematical helpers."""

import ast
import hashlib
import itertools
import json
from pathlib import Path

import pytest


def functions():
    root = Path(__file__).with_name("runs") / "defer_kv_v1"
    if not root.is_dir():
        pytest.skip("Materialize the frozen scheduling candidate first")
    return {
        fn.name: fn
        for name in ("cube_schedule.py", "projection_overlap_stages.py")
        for fn in ast.parse((root / name).read_text()).body
        if isinstance(fn, ast.FunctionDef)
    }


def events(body, config, functions):
    result = []
    for node in body:
        if isinstance(node, ast.If):
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if names <= config.keys():
                condition = eval(compile(ast.Expression(node.test), "constexpr", "eval"), {}, config)
                result += events(node.body if condition else node.orelse, config, functions)
            else:
                result += events(node.body + node.orelse, config, functions)
        elif isinstance(node, (ast.For, ast.While)):
            result += events(node.body, config, functions)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            name = ast.unparse(call.func)
            if name == "projection_overlap_stage":
                deferred = eval(compile(ast.Expression(call.args[-1]), "constexpr", "eval"), {}, config)
                result += events(functions[name].body, {**config, "DEFER_KV": deferred}, functions)
            elif name == "kv_project_stage":
                result.append("kv")
            elif name == "indexer_score_cube_stage":
                result.append("score")
            elif name == "publish_phase" and ast.unparse(call.args[1]) == "PHASE_KV_PROJECT":
                result.append("publish_kv")
    return result


@pytest.mark.parametrize("cooperative,split,cols", itertools.product([False, True], [False, True], [128, 256]))
def test_kv_once_and_never_published_before_computation(cooperative, split, cols):
    fns = functions()
    config = dict(COOPERATIVE_PREFIX=cooperative, SPLIT_SCORE_ROWS=split, OA_COLS=cols)
    sequence = events(fns["cube_schedule"].body, config, fns)
    expected = ["score", "kv", "publish_kv"] if cooperative and split and cols == 256 else ["kv", "publish_kv", "score"]
    assert sequence == expected


def test_math_consumers_and_workspace_are_unchanged():
    here = Path(__file__).resolve().parent
    expected = json.loads((here / "baseline.json").read_text())["files"]
    root = here / "runs/defer_kv_v1"
    if not root.is_dir():
        pytest.skip("Materialize the frozen scheduling candidate first")
    changed = {
        name for name, digest in expected.items() if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
    }
    assert changed == {"cube_schedule.py", "projection_overlap_stages.py"}
    fns = functions()
    calls = [
        n
        for n in ast.walk(fns["cube_schedule"])
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "kv_project_stage"
    ]
    assert len(calls) == 2
    assert ast.unparse(calls[1]) == "kv_project_stage(normed, wkv, kv_fp32, kv_worker, GROUPS)"
    assignments = [
        n
        for n in ast.walk(fns["cube_schedule"])
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "kv_worker" for t in n.targets)
    ]
    assert len(assignments) == 1
    assert ast.unparse(assignments[0].value) == "(worker + GROUPS - PREFIX_BRANCH_GROUPS) % GROUPS"
