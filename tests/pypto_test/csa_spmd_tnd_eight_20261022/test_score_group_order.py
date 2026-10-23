# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual scalar planner and root waits with a simulated GM array."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not Path(__file__).with_name("runs").joinpath("score_group_order_v2/attention_dependencies.py").exists(),
    reason="Materialize the frozen baseline and prepare score_group_order_v2 first",
)


def helpers():
    here = Path(__file__).resolve().parent
    nodes = []
    for path, names in (
        (here / "score_group_order_source.py", None),
        (here / "runs/baseline/indexer_group_stages.py", {"indexer_work_item"}),
        (here / "runs/score_group_order_v2/attention_dependencies.py", {"wait_indexer_query_roots"}),
    ):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and (names is None or node.name in names):
                node.decorator_list, node.returns = [], None
                for argument in node.args.args:
                    argument.annotation = None
                nodes.append(node)
    pl = SimpleNamespace(
        INDEX=int,
        INT32=int,
        cast=lambda value, dtype: dtype(value),
        range=range,
        read=lambda array, indices: array[tuple(indices)],
        write=lambda array, indices, value: array.__setitem__(tuple(indices), value),
        min=min,
        max=max,
        system=SimpleNamespace(fence=lambda: None, cacheinvalid=lambda: None),
    )
    namespace = dict(
        pl=pl,
        SCORE_WORKERS=24,
        GROUPS=24,
        CORES=72,
        PHASE_SCORE_VECTOR=35,
        SIGNAL_SLOT_WORDS=128,
        csa_signal_wait=lambda signal, offset, gen: signal + [(offset, gen)],
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "score_order_helpers", "exec"), namespace)
    return namespace


def check(lives, visible, *, tokens, requests=None, leaves=6, storage=None):
    funcs = helpers()
    groups = len(lives)
    requests = groups if requests is None else requests
    start = ((requests + 16) // 16) * 16
    if storage is None:
        storage = np.full(start + 4 * tokens + 128, 37, dtype=np.int32)
    for group, (live, count) in enumerate(zip(lives, visible)):
        storage[start + 4 * group : start + 4 * group + 4] = [group, 0, live, count]
    previous = storage.copy()
    funcs["plan_score_group_order"](storage, groups, requests, tokens, leaves)
    base = funcs["active_score_order_base"](storage, groups, requests, tokens)
    changed = np.flatnonzero(storage != previous)
    assert all(start + 4 * groups <= pos < start + 4 * tokens for pos in changed)
    order = list(range(groups)) if base < 0 else storage[base + 1 : base + 1 + groups].tolist()
    assert sorted(order) == list(range(groups))
    if base >= 0:
        inverse = storage[base + 1 + groups : base + 1 + 2 * groups]
        assert all(inverse[group] == slot for slot, group in enumerate(order))
    writes = {}
    for item in range(groups * leaves):
        slot, leaf = funcs["indexer_work_item"](item, groups, leaves)
        group = order[slot]
        assert (group, leaf) not in writes
        writes[group, leaf] = item % 24
    assert len(writes) == groups * leaves
    plan = np.array([groups, 32769, leaves, 5, 3, 6, 1, 1], dtype=np.int32)
    for group in range(groups):
        for query in range(lives[group]):
            waits = funcs["wait_indexer_query_roots"](
                [], plan, storage, requests, tokens, group, query, leaves, 9, 6, True
            )
            producers = [(offset // 128 - 35 * 72, gen) for offset, gen in waits]
            assert producers == [(24 + 2 * writes[group, leaf] + query // 3, 9) for leaf in range(leaves)]
    return base, storage


def test_work_item_bijection_and_inverse_waits():
    rng = np.random.default_rng(1024)
    for groups in (1, 4, 8, 15, 16, 17, 24, 28):
        for leaves in (3, 6, 9, 30, 32):
            lives = rng.integers(1, 7, size=groups).tolist()
            visible = [32768 + int(q > 3) for q in lives]
            check(lives, visible, tokens=sum(lives), leaves=leaves)


def test_mixed_reorders_and_same_storage_resets_flag():
    lives = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 3, 4, 6]
    base, storage = check(lives, [32768 + int(q > 3) for q in lives], tokens=56)
    assert base >= 0
    base, _ = check([2] * 16, [32768] * 16, tokens=56, storage=storage)
    assert base < 0  # Stale nonidentity tables are never consumed.


def test_unequal_histories_and_nonmultiple_leaves_keep_identity():
    lives = [6, 6] + [2] * 6 + [6, 6] + [2] * 6
    for counts, leaves in (([1] + [32768] * 15, 6), ([32768] * 16, 32)):
        base, _ = check(lives, counts, tokens=56, leaves=leaves)
        assert base < 0
