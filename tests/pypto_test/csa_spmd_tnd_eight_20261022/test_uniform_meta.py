# SPDX-License-Identifier: Apache-2.0
"""Verify the actual META scan, reserved storage and per-query fallback semantics."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_uniform_geometry import oracle

HERE = Path(__file__).resolve().parent


def functions(path, namespace, wanted):
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list, node.returns = [], None
            for arg in node.args.args:
                arg.annotation = None
            nodes.append(node)
    assert len(nodes) == len(wanted)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture(scope="module")
def helpers():
    read = lambda data, index: int(data[tuple(index)])  # noqa: E731
    pl = SimpleNamespace(
        INDEX=int, INT32=int, cast=lambda value, dtype: dtype(value), range=range, min=min, max=max,
        tensor=SimpleNamespace(dim=lambda data, axis: data.shape[axis], read=read), read=read,
        write=lambda data, index, value: data.__setitem__(tuple(index), value),
    )
    common = dict(pl=pl, COMPRESS_RATIO=4, GROUP_DESCRIPTOR_ALIGNMENT=16, GROUP_DESCRIPTOR_WORDS=4,
                  MAX_CANDIDATES=262144, CANDIDATES_PER_LEAF=8192, SCORE_TILE=1024,
                  MAX_LEAVES=32, SCORE_WORKERS=24)
    functions(HERE / "runs/uniform_meta_v1/request_geometry.py", common, {"request_bounds", "query_request"})
    previous, current = dict(common), dict(common)
    wanted = {"indexer_long_leaf_plan", "indexer_group_plan_stage"}
    functions(HERE / "runs/baseline/indexer_group_stages.py", previous, wanted)
    functions(HERE / "runs/uniform_meta_v1/indexer_group_stages.py", current, wanted)
    functions(HERE / "runs/uniform_meta_v1/uniform_query_stages.py", current,
              {"read_uniform_six", "query_request_uniform_six"})
    return previous, current


def prove_and_compare(helpers, bounds, lengths, tokens, positions=None, existing=None, enabled=True):
    previous, current = helpers
    bounds = np.asarray(bounds, np.int32)
    lengths = np.asarray(lengths, np.int32)
    if positions is None:
        positions = np.arange(tokens, dtype=np.int64) % 6 + 131072
    requests = len(lengths)
    base = ((requests + 1 + 15) // 16) * 16
    if existing is None:
        old = np.full(base + 4 * tokens + 64, 37, dtype=np.int32)
        new = old.copy()
    else:
        old, new = existing
    old_plan, new_plan = np.full(8, 37, np.int32), np.full(8, 37, np.int32)
    previous["indexer_group_plan_stage"](bounds, lengths, positions, old, old_plan, tokens, 0, 6, True, 1)
    current["indexer_group_plan_stage"](bounds, lengths, positions, new, new_plan, tokens, 0, 6, True, 1, enabled)
    assert np.array_equal(old_plan, new_plan)
    unchanged = np.ones(len(old), dtype=bool)
    if enabled:
        if base > requests + 1:
            unchanged[base - 1] = False
        width = current["read_uniform_six"](new, requests)
    else:
        width = 0
    assert np.array_equal(old[unchanged], new[unchanged])
    expected = bool(
        enabled and requests > 1 and base > requests + 1 and tokens == 6 * requests
        and len(bounds) >= requests + 1
        and np.array_equal(bounds[:requests + 1], np.arange(requests + 1) * 6)
        and np.all(lengths == lengths[0]) and lengths[0] >= 6
    )
    assert width == int(expected)
    for query in range(-1, tokens + 1):
        assert current["query_request_uniform_six"](bounds, lengths, query, tokens, width) == oracle(
            bounds, lengths, query, tokens
        )
    return old, new


@pytest.mark.parametrize("batch", [1, 2, 8, 15, 16, 24, 28, 31, 32])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6])
def test_uniform_and_padded_queries(helpers, batch, width):
    prove_and_compare(helpers, np.arange(batch + 1) * width, [131072 + width] * batch, batch * 6)


@pytest.mark.parametrize("batch", [2, 15, 16, 24, 31])
def test_middle_boundary_and_length_corruption_cannot_enable_fast_path(helpers, batch):
    bounds, lengths = np.arange(batch + 1) * 6, np.full(batch, 131078)
    for index in range(batch + 1):
        changed = bounds.copy()
        changed[index] += 1
        prove_and_compare(helpers, changed, lengths, batch * 6)
    for index in range(batch):
        for value in [0, -1, 5, 131077]:
            changed = lengths.copy()
            changed[index] = value
            prove_and_compare(helpers, bounds, changed, batch * 6)


@pytest.mark.parametrize("batch", [16, 24])
def test_same_storage_overwrites_proof_on_every_transition(helpers, batch):
    bounds, lengths = np.arange(batch + 1) * 6, np.full(batch, 131078)
    state = prove_and_compare(helpers, bounds, lengths, batch * 6)
    altered = lengths.copy()
    altered[batch // 2] -= 1
    state = prove_and_compare(helpers, bounds, altered, batch * 6, existing=state)
    state = prove_and_compare(helpers, bounds, lengths, batch * 6, existing=state)
    altered_bounds = bounds.copy()
    altered_bounds[batch // 2] -= 1
    state = prove_and_compare(helpers, altered_bounds, lengths, batch * 6, existing=state)
    prove_and_compare(helpers, bounds, lengths, batch * 6, existing=state)


def test_disabled_proof_preserves_all_metadata_storage(helpers):
    prove_and_compare(helpers, np.arange(17) * 6, [131078] * 16, 96, enabled=False)


def test_both_prefix_orders_acquire_meta_before_attention():
    path = HERE / "runs/uniform_meta_v1/vector_prefix_stages.py"
    function = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "indexer_compressor_vector_stage")
    waits = [n for n in function.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Name) and n.value.func.id == "wait_producer"]
    assert any(isinstance(n.value.args[1], ast.Name) and n.value.args[1].id == "PHASE_META" for n in waits)
