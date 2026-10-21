# SPDX-License-Identifier: Apache-2.0
"""Check actual generated producer mapping against the SPMD consumer's writer selection."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from prepare_mixed_candidates import live_half_source, rotate_sources


def plain_function(source, name, namespace):
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), name, "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def generated():
    here = Path(__file__).resolve().parent / "runs/frozen_v122"
    if not here.exists():
        pytest.skip("Materialize frozen reference first")
    score = live_half_source((here / "indexer_score_stages.py").read_text())
    return rotate_sources(
        (here / "indexer_group_stages.py").read_text(), score, (here / "attention_dependencies.py").read_text()
    )


@pytest.mark.parametrize("query_group", [2, 6])
def test_producer_inverse_wait_mapping_including_empty_and_large_groups(generated, query_group):
    group, _, dependency = generated
    observed = []

    def wait(signals, offset, generation):
        observed.append(offset // 128 - 72)
        return signals

    namespace = {
        "pl": SimpleNamespace(
            cast=lambda x, _: x,
            INDEX=None,
            range=range,
            read=lambda t, i: t[i[0]],
            system=SimpleNamespace(fence=lambda: None, cacheinvalid=lambda: None),
        ),
        "SCORE_WORKERS": 24,
        "GROUPS": 24,
        "CORES": 72,
        "PHASE_SCORE_VECTOR": 1,
        "SIGNAL_SLOT_WORDS": 128,
        "csa_signal_wait": wait,
    }
    forward = plain_function(group, "indexer_work_item", namespace)
    consumer = plain_function(dependency, "wait_indexer_query_roots", namespace)
    for groups in (0, 1, 2, 3, 4, 5, 8, 11, 16, 23, 24, 25, 28, 32, 48, 64):
        for leaves in (1, 2, 3, 5, 6, 9, 16, 32):
            owners = {}
            for item in range(groups * leaves):
                logical = forward(item, groups, leaves, query_group)
                assert logical not in owners
                owners[logical] = item % 24
                if query_group == 2:
                    expected = (item % groups, item // groups) if groups < 24 else (item // leaves, item % leaves)
                    assert logical == expected
            assert len(owners) == groups * leaves
            assert forward(-1, groups, leaves, query_group) == (-1, -1)
            assert forward(groups * leaves, groups, leaves, query_group) == (-1, -1)
            plan = [groups, 0, leaves]
            for logical_group in range(groups):
                for lane in (0, 1):
                    observed.clear()
                    consumer([], plan, logical_group, lane * (query_group // 2), leaves, 1, query_group, True)
                    assert observed == [24 + 2 * owners[(logical_group, leaf)] + lane for leaf in range(leaves)]


def test_fixed_mixed_work_model_is_not_a_latency_prediction():
    queries = (1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 3, 4, 6)
    maximums = []
    for rotate in (False, True):
        cube, vector = [0] * 24, [[0] * 24, [0] * 24]
        for item in range(16 * 6):
            leaf, ordinal = divmod(item, 16)
            group = (ordinal + leaf) % 16 if rotate else ordinal
            begin = leaf * 5 + min(leaf, 3)
            span = 5 + (leaf < 3)
            steps = max(0, min(span, (((131072 + queries[group]) // 4 + 1023) // 1024) - begin))
            cube[item % 24] += steps * (1 if queries[group] <= 3 else 2)
            vector[0][item % 24] += steps * min(queries[group], 3)
            vector[1][item % 24] += steps * max(queries[group] - 3, 0)
        maximums.append((max(cube), max(vector[0]), max(vector[1])))
    assert maximums == [(44, 66, 55), (39, 66, 36)]
