# SPDX-License-Identifier: Apache-2.0
"""Execute real branch ownership and prove Q_B publication follows all writes."""

import ast
import collections
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/qb_tail_helpers_v1"
PARENT = HERE / "runs/output_rectangle_production_v1"


def function(root, filename, name):
    return next(
        n for n in ast.parse((root / filename).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def runnable(fn, scope):
    fn.decorator_list, fn.returns = [], None
    for arg in fn.args.args:
        arg.annotation = None
    tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(tree, "actual_source_control", "exec"), scope)
    return scope[fn.name]


def range_geometry(records):
    fn = function(SOURCE, "query_projection_stages.py", "q_b_nz_column_range")
    outer = next(n for n in fn.body if isinstance(n, ast.For))
    inner = next(n for n in outer.body if isinstance(n, ast.For))
    valid_rows = next(n for n in inner.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "m_rows")
    inner.body = [valid_rows] + ast.parse("record(qproj_n_idx, t0, m_rows)").body
    return runnable(
        fn,
        dict(
            pl=SimpleNamespace(max=max, min=min, range=range),
            QPROJ_PIPE_M_TILE=128,
            QPROJ_N_BLOCKS=128,
            QPROJ_MM_N_TILE=256,
            record=lambda *args: records.append(args),
        ),
    )


def execute_projection(root, tokens, worker, *, wide=False, branch_groups=4, nz=True, joint=False):
    events, writes = [], []
    geometry = range_geometry(writes)

    def publish(signal, phase, core, count, generation):
        assert (core, count, generation) == (worker, 72, 1)
        events.append(("publish", phase))
        return signal

    def qb_range(*args):
        events.append(("range", args[4:]))
        return geometry(*args)

    def passthrough(name, outputs):
        def invoke(*_args):
            events.append(("math", name))
            return (None,) * outputs if outputs > 1 else None

        return invoke

    tags = (
        "MAIN_PROJECT",
        "WEIGHTS_PROJECT",
        "QB",
        "META",
        "IDX_NORM",
        "IDX_HADAMARD",
        "IDX_QUERY_ROPE",
        "IDX_QUERY_HAD",
        "KV_PROJECT",
    )
    scope = dict(
        pl=SimpleNamespace(
            tensor=SimpleNamespace(dim=lambda v, axis: v),
            min=min,
            cast=lambda v, dtype: v,
            read=lambda v, ix: 1,
            INDEX=int,
        ),
        GROUPS=24,
        VECTORS=48,
        CORES=72,
        QUANT_WEIGHT_NZ=nz,
        QPROJ_N_BLOCKS=128,
        publish_phase=publish,
        wait_producer=lambda signal, *_: signal,
        wait_row_owners=lambda signal, *_: signal,
        compressor_project_stage=passthrough("main", 2),
        indexer_weights_project_stage=passthrough("weights", 1),
        indexer_compressor_hadamard_stage=passthrough("compact_hadamard", 1),
        _query_hadamard_four_tokens=passthrough("query_hadamard", 1),
        kv_project_stage=passthrough("kv", 1),
        q_b_stage=passthrough("original_qb", 1),
        q_b_nz_column_range=qb_range,
        **{"PHASE_" + tag: tag for tag in tags},
    )
    fn = function(root, "projection_overlap_stages.py", "projection_overlap_stage")
    names = [arg.arg for arg in fn.args.args]
    invoke = runnable(fn, scope)
    args = dict.fromkeys(names)
    args.update(
        normed=tokens,
        idx_lengths=max(1, math.ceil(tokens / 6)),
        worker=worker,
        generation=1,
        BRANCH_GROUPS=branch_groups,
        QB_WIDE_N=wide,
        JOINT_BF16_PREFIX=joint,
    )
    invoke(**args)
    return events, writes


@pytest.mark.parametrize("tokens", [0, 1, 6, 16, 32, 64, 96, 127, 128, 129, 144, 168, 192, 256])
def test_all_output_columns_once_and_real_completion(tokens):
    writes = []
    for core in range(24):
        events, emitted = execute_projection(SOURCE, tokens, core)
        writes += [(core, *entry) for entry in emitted]
        assert events.count(("publish", "QB")) == 1
        ranged = next(i for i, row in enumerate(events) if row[0] == "range")
        assert ranged < events.index(("publish", "QB"))
        if core < 4:
            assert events.index(("publish", "IDX_QUERY_HAD")) < ranged
        assert events.index(("publish", "QB")) < events.index(("math", "kv"))
    actual = collections.Counter(
        (column, row) for _, column, first, count in writes for row in range(first, first + count)
    )
    assert actual == collections.Counter({(column, row): 1 for column in range(128) for row in range(tokens)})
    assert collections.Counter(core for core, *_ in writes) == collections.Counter(
        {core: (2 if core < 4 else 6) * math.ceil(tokens / 128) for core in range(24)}
    )


@pytest.mark.parametrize("mode", [dict(wide=True), dict(branch_groups=2), dict(nz=False)])
def test_closed_modes_preserve_exact_call_and_publication_order(mode):
    for core in range(24):
        for joint in (False, True):
            assert execute_projection(SOURCE, 96, core, joint=joint, **mode) == execute_projection(
                PARENT, 96, core, joint=joint, **mode
            )


def test_original_qb_matrix_body_and_all_other_modules_unchanged():
    original = function(PARENT, "query_projection_stages.py", "_q_b_nz_stage")
    helper = function(SOURCE, "query_projection_stages.py", "q_b_nz_column_range")
    inner = lambda fn: next(n for n in ast.walk(fn) if isinstance(n, ast.For) and ast.unparse(n.target) == "t0")
    assert ast.dump(inner(original)) == ast.dump(inner(helper))
    for path in PARENT.glob("*.py"):
        if path.name not in ("query_projection_stages.py", "projection_overlap_stages.py"):
            assert path.read_bytes() == (SOURCE / path.name).read_bytes()
    # The unchanged consumer includes all new physical writers, not the old
    # twenty-rank modulo domain used inside the Q_B math helper.
    tree = ast.parse((SOURCE / "vector_schedule.py").read_text())
    waits = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "wait_row_owners"
        and len(n.args) > 1
        and ast.unparse(n.args[1]) == "PHASE_QB"
    ]
    assert len(waits) == 1
    args = [eval(compile(ast.Expression(n), "wait_owner", "eval"), {"GROUPS": 24}) for n in waits[0].args[4:]]
    first, count, tile, workers, base, stride = args
    owners = {base + ((first // tile + i) % workers) * stride for i in range(min(math.ceil(count / tile), workers))}
    assert owners == set(range(24))
