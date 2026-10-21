# SPDX-License-Identifier: Apache-2.0
"""Check the actual head-group selector against complete row/head ownership."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from prepare_front_partition import build, function
from test_mixed_candidates import plain_function

REFERENCE = Path(__file__).resolve().parent / "runs/frozen_v122"


def selector(fn):
    selected = copy.deepcopy(fn)
    loop = next(i for i, n in enumerate(selected.body) if isinstance(n, ast.For) and ast.unparse(n.target) == "dq_work")
    selected.body = selected.body[:loop] + [ast.Return(value=ast.Name(id="dq_head_tile", ctx=ast.Load()))]
    selected.args.args = [ast.arg(arg="q_flat"), ast.arg(arg="WORKERS")]
    selected.args.defaults = []
    source = ast.unparse(ast.fix_missing_locations(selected))
    ns = dict(
        pl=SimpleNamespace(tensor=SimpleNamespace(dim=lambda tensor, axis: tensor)),
        Q_ROPE_T_TILE=8,
        Q_ROPE_H_TILE=4,
        Q_DEQUANT_WORKERS=48,
        H=64,
    )
    return plain_function(source, fn.name, ns)


@pytest.fixture(scope="module")
def functions():
    old = function((REFERENCE / "query_projection_stages.py").read_text(), "q_dequant_rope_stage")
    new = function(build(REFERENCE, 8, True, True)["query_projection_stages.py"], "front_q_dequant_rope_stage")
    return old, new


@pytest.mark.parametrize("workers", (24, 32, 40, 48))
def test_each_real_head_row_has_one_owner_and_tail_math_keeps_four_heads(functions, workers):
    choose = selector(functions[1])
    for tokens in (0, 1, 6, 8, 24, 32, 41, 48, 56, 64, 65, 80, 96, 128, 144, 168, 192, 384, 385):
        tile = choose(tokens, workers)
        assert tile in (4, 8, 16, 32, 64)
        if tokens % 8:
            assert tile == 4
        covered = set()
        for worker in range(workers):
            for item in range(worker, ((tokens + 7) // 8) * (64 // tile), workers):
                first_t, first_h = item // (64 // tile) * 8, item % (64 // tile) * tile
                for t in range(first_t, min(first_t + 8, tokens)):
                    for h in range(first_h, first_h + tile):
                        assert (t, h) not in covered
                        covered.add((t, h))
        assert len(covered) == tokens * 64


def test_math_body_exact_and_t96_worker_load_model(functions):
    old, new = functions
    old_loop = next(n for n in old.body if isinstance(n, ast.For))
    new_loop = next(n for n in new.body if isinstance(n, ast.For))
    assert ast.dump(old_loop, include_attributes=False) == ast.dump(new_loop, include_attributes=False)
    old_select, new_select = selector(old), selector(new)
    old_tile, new_tile = old_select(96, 32), new_select(96, 32)
    assert (old_tile, new_tile) == (16, 8)
    old_counts = [len(range(w, 12 * (64 // old_tile), 32)) * old_tile for w in range(32)]
    new_counts = [len(range(w, 12 * (64 // new_tile), 32)) * new_tile for w in range(32)]
    assert sorted(old_counts) == [16] * 16 + [32] * 16
    assert new_counts == [24] * 32
    assert sum(old_counts) == sum(new_counts) == 12 * 64
    assert new_select(56, 32) == old_select(56, 32) == 4
    assert new_select(65, 32) == old_select(65, 32) == 4
