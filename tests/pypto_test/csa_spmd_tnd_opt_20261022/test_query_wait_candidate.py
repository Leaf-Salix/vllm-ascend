# SPDX-License-Identifier: Apache-2.0
"""Match real consumer head accesses to the actual physical Q_B column striping."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from prepare_query_wait_candidate import HELPER, factor_head_selector


def function(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)


def executable(fn, namespace):
    fn = copy.deepcopy(fn)
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), fn.name, "exec"), namespace)
    return namespace[fn.name]


@pytest.fixture(scope="module")
def factored():
    root = Path(__file__).resolve().parent / "runs/frozen_v122"
    if not root.exists():
        pytest.skip("Materialize frozen v122 first")
    original = (root / "query_projection_stages.py").read_text()
    return original, factor_head_selector(original)


def test_factoring_preserves_complete_dequant_math_ast(factored):
    original, changed = factored
    old = function(original, "q_dequant_rope_stage")
    new = function(changed, "q_dequant_rope_stage")
    helper = function(changed, "select_dequant_head_tile")
    assignment = next(
        n
        for n in new.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "select_dequant_head_tile"
    )
    pos = new.body.index(assignment)
    new.body[pos : pos + 1] = copy.deepcopy(helper.body[:-1])
    assert ast.dump(new) == ast.dump(old)


@pytest.mark.parametrize(
    "cooperative,wide,branch",
    [(False, False, 4)] + [(True, wide, branch) for wide in (False, True) for branch in (2, 4, 8, 12, 20, 23)],
)
def test_exact_waits_cover_every_consumed_column_and_no_unused_writer(factored, cooperative, wide, branch):
    _, changed = factored
    observed, events = [], []

    def wait(signal, offset, generation):
        assert generation == 7
        observed.append(offset // 128 - 10 * 72)
        events.append("wait")
        return signal

    namespace = {
        "pl": SimpleNamespace(
            min=min,
            range=range,
            system=SimpleNamespace(fence=lambda: events.append("fence"), cacheinvalid=lambda: events.append("acquire")),
        ),
        "Q_ROPE_T_TILE": 8,
        "Q_ROPE_H_TILE": 4,
        "Q_DEQUANT_WORKERS": 48,
        "H": 64,
        "CORES": 72,
        "GROUPS": 24,
        "PHASE_QB": 10,
        "VECTORS": 48,
        "SIGNAL_SLOT_WORDS": 128,
        "csa_signal_wait": wait,
    }
    choose = executable(function(changed, "select_dequant_head_tile"), namespace)
    consume = executable(function(HELPER, "wait_query_projection_columns"), namespace)
    count = 24 - branch if cooperative else 24
    width = 512 if cooperative and wide else 256
    # Build ownership in the producer's forward direction, not by the inverse
    # formula used by the new consumer helper.
    writers = {}
    for physical in range(branch if cooperative else 0, 24):
        for column in range(physical % count, 32768 // width, count):
            assert column not in writers
            writers[column] = physical
    assert len(writers) == 32768 // width
    for tokens in (
        0,
        1,
        6,
        7,
        8,
        9,
        23,
        24,
        25,
        41,
        48,
        56,
        63,
        64,
        65,
        96,
        127,
        128,
        129,
        144,
        168,
        192,
        256,
        384,
        385,
    ):
        tile = choose(tokens)
        for worker in range(48):
            expected, heads = set(), set()
            for task in range(worker, ((tokens + 7) // 8) * (64 // tile), 48):
                first = task % (64 // tile) * tile
                heads.add(first)
                for head in range(first, first + tile):
                    expected.update(writers[col] for col in range(head * 512 // width, (head + 1) * 512 // width))
            assert len(heads) <= 1
            observed.clear()
            events.clear()
            consume([], tokens, worker, 7, cooperative, branch, wide)
            assert set(observed) == expected
            assert len(observed) == len(set(observed))
            if expected:
                assert events[-3:] == ["fence", "acquire", "fence"]
            else:
                assert not events
