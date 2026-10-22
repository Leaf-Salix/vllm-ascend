# SPDX-License-Identifier: Apache-2.0
"""Check the actual producer's uniform rows, empty safety and unchanged Query math."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from prepare_query_resident_swap import transform
from test_mixed_candidates import plain_function
from test_query_resident_weights import functions

HERE = Path(__file__).resolve().parent


def test_only_index_preparation_moves_and_empty_load_is_guarded():
    source = (HERE / "runs/query_resident_weights/query_projection_stages.py").read_text()
    old, new = functions(source), functions(transform(source))
    name = "_q_dequant_rope_contiguous_heads"
    for key in old:
        if key != name:
            assert ast.dump(old[key], include_attributes=False) == ast.dump(new[key], include_attributes=False)
    fn = new[name]
    guard = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "tokens > 0")
    assert ast.unparse(guard.body[0].value.args[1]) == "[0, 0]"
    assert all(not eval(compile(ast.Expression(guard.test), "<nonempty>", "eval"), {}, {"tokens": t}) for t in (0,))
    owner = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "q_abs_swap")
    fn.body.remove(owner)
    fn.body.remove(guard)
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    scratch = next(
        n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "q_flat_gather_tmp"
    )
    guard.body[0].value.args[1].elts[0] = ast.Name(id="qt", ctx=ast.Load())
    index = loop.body.index(scratch)
    loop.body[index:index] = guard.body
    assert ast.dump(fn, include_attributes=False) == ast.dump(old[name], include_attributes=False)


def test_actual_producer_uniformity_full_tail_idle_and_changing_input():
    source = (HERE / "runs/query_resident_weights/query_projection_stages.py").read_text()

    def load(tensor, start, shape, valid_shape, **_):
        r, c = start
        rows, cols = valid_shape
        return tensor[r : r + rows, c : c + cols].copy()

    def store(tile, start, tensor):
        r, c = start
        tensor[r : r + tile.shape[0], c : c + tile.shape[1]] = tile
        return tensor

    tile = SimpleNamespace(
        full=lambda shape, dtype, value: np.full(shape, value, dtype),
        arange=lambda start, shape, dtype: np.arange(start, start + np.prod(shape), dtype=dtype).reshape(shape),
    )
    pl = SimpleNamespace(
        FP32=np.float32,
        INT32=np.int32,
        MemorySpace=SimpleNamespace(Vec=None),
        full=tile.full,
        arange=tile.arange,
        tile=tile,
        tensor=SimpleNamespace(dim=lambda t, axis: t.shape[axis]),
        cast=lambda x, target_type, **_: x.astype(target_type),
        min=min,
        range=range,
        mul=np.multiply,
        add=np.add,
        sub=np.subtract,
        col_expand_mul=np.multiply,
        load=load,
        store=store,
        set_validshape=lambda x, rows, cols: x[:rows, :cols],
    )
    producer = plain_function(source, "q_rope_prepare_stage", dict(pl=pl, Q_ROPE_T_TILE=8, ROPE_DIM=64))
    rng = np.random.default_rng(20261022)
    for tokens in (0, 1, 6, 7, 8, 9, 24, 48, 56, 65, 72, 96, 144, 168, 192, 193):
        for workers in (1, 7, 16, 48):
            sin = rng.standard_normal((tokens, 64), dtype=np.float32)
            signed = np.full_like(sin, np.nan)
            swap = np.full((tokens, 64), -1, dtype=np.int32)
            for worker in range(workers):
                producer(sin, signed, swap, worker, workers)
            columns = np.arange(64, dtype=np.int32)
            np.testing.assert_array_equal(swap, np.broadcast_to(columns ^ 1, (tokens, 64)))
            np.testing.assert_array_equal(signed, sin * (2 * (columns % 2) - 1))
            if tokens:
                absolute = np.broadcast_to(swap[0], (8, 64)) + np.arange(8)[:, None] * 64
                np.testing.assert_array_equal(absolute.ravel(), np.arange(512) ^ 1)


def test_existing_rope_acquisition_precedes_query_consumer():
    source = (HERE / "runs/query_resident_weights/vector_schedule.py").read_text()
    fn = functions(source)["vector_schedule"]
    operations = [
        (n.lineno, ast.unparse(n.func), [ast.unparse(a) for a in n.args])
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
    ]
    prepare = next(line for line, name, _ in operations if name == "q_rope_prepare_stage")
    publish = next(line for line, name, args in operations if name == "publish_phase" and args[1] == "PHASE_ROPE")
    wait = next(
        (line, args) for line, name, args in operations if name == "wait_row_owners" and args[1] == "PHASE_ROPE"
    )
    consume = next(line for line, name, _ in operations if name == "q_dequant_rope_stage")
    assert prepare < publish < wait[0] < consume
    assert wait[1][4:] == ["0", "tokens", "8", "VECTORS", "GROUPS", "1"]
