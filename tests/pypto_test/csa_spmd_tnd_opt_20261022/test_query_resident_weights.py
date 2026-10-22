# SPDX-License-Identifier: Apache-2.0
"""Prove the weight-owner invariant and preserve the entire previous math body."""

import ast
from pathlib import Path

from prepare_query_resident_weights import transform

HERE = Path(__file__).resolve().parent


def functions(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def test_only_the_weight_load_location_changes():
    source = (HERE / "runs/query_contiguous_heads_v4/query_projection_stages.py").read_text()
    old, new = functions(source), functions(transform(source))
    name = "_q_dequant_rope_contiguous_heads"
    for key in old:
        if key != name:
            assert ast.dump(old[key], include_attributes=False) == ast.dump(new[key], include_attributes=False)
    fn = new[name]
    owner_init = next(
        n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "resident_head_scale"
    )
    owner_guard = next(n for n in fn.body if isinstance(n, ast.If))
    fn.body.remove(owner_init)
    fn.body.remove(owner_guard)
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    guard = next(n for n in loop.body if isinstance(n, ast.If))
    assert ast.dump(guard.test) == ast.dump(owner_guard.test)
    assert ast.unparse(guard.body[0]) == "q_head_scale = resident_head_scale"
    assert len(guard.orelse) == 1
    loop.body[loop.body.index(guard)] = guard.orelse[0]
    assert ast.dump(fn, include_attributes=False) == ast.dump(old[name], include_attributes=False)


def test_actual_worker_stride_preserves_resident_head_and_other_strides_fall_back():
    source = (HERE / "runs/query_contiguous_heads_v4/query_projection_stages.py").read_text()
    fn = functions(transform(source))["_q_dequant_rope_contiguous_heads"]
    guard = next(n for n in fn.body if isinstance(n, ast.If))
    owner = next(
        n.value for n in guard.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "resident_head_begin"
    )
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    hg = next(n.value for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "hg")

    def evaluate(expr, values):
        return eval(compile(ast.Expression(expr), "<head-owner>", "eval"), {}, values)

    for workers in (1, 2, 7, 8, 16, 20, 24, 28, 32, 40, 48, 64, 96):
        ns = dict(H=64, Q_ROPE_T_TILE=8, WORKERS=workers)
        enabled = evaluate(guard.test, ns)
        assert enabled == (workers % 8 == 0)
        for tokens in (8, 24, 48, 56, 64, 72, 80, 96, 144, 168, 192, 384):
            for worker in range(workers):
                ns["worker_id"] = worker
                resident = evaluate(owner, ns)
                rows = []
                for work in range(worker, tokens * 8, workers):
                    ns["dq_work"] = work
                    rows.append(evaluate(hg, ns))
                if enabled:
                    assert all(row == resident for row in rows)
                elif len(rows) > 1:
                    assert len(set(rows)) > 1  # Hoisting here would be wrong.
