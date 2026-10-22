# SPDX-License-Identifier: Apache-2.0
"""Exercise the copied leaf proof against an independent complete-page oracle."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def leaf_proof():
    source = Path(__file__).with_name("runs") / "leaf_scale_v1/leaf_scale_layout_stages.py"
    if not source.is_file():
        pytest.skip("Materialize the frozen leaf candidate first")
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef))
    node.decorator_list, node.returns = [], None
    for arg in node.args.args:
        arg.annotation = None
    tile = SimpleNamespace(
        ci=lambda begin, shape, dtype: np.arange(begin, begin + np.prod(shape), dtype=dtype).reshape(shape),
        full=lambda shape, dtype, value: np.full(shape, value, dtype=dtype),
        read=lambda data, index: data[tuple(index)],
    )
    pl = SimpleNamespace(
        INDEX=int,
        INT32=np.int32,
        FP32=np.float32,
        tile=tile,
        tensor=SimpleNamespace(dim=lambda data, axis: data.shape[axis]),
        cast=lambda data, dtype: dtype(data),
        min=min,
        max=max,
        read=lambda data, index: data[tuple(index)],
        load=lambda data, index, shape, valid_shape: data[
            index[0] : index[0] + valid_shape[0], index[1] : index[1] + valid_shape[1]
        ].copy(),
        add=np.add,
        sub=np.subtract,
        mul=np.multiply,
        abs=np.abs,
        set_validshape=lambda data, rows, cols: data[:rows, :cols],
        col_expand_mul=np.multiply,
        create_tile=lambda shape, dtype: np.empty(shape, dtype=dtype),
        row_max=lambda data, scratch: data.max(axis=1, keepdims=True),
    )
    namespace = dict(pl=pl, LAYOUT_PROOF_PAGES=256, LAYOUT_PROOF_ROWS=8, LAYOUT_EXACT_INTEGER_LIMIT=1 << 24)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "leaf_scale_proof", "exec"), namespace)
    return namespace[node.name]


def oracle(table, request, begin, count, physical):
    if not (0 <= request < table.shape[0] and begin >= 0 and 0 < count <= 256 and begin + count <= table.shape[1]):
        return (0, 0)
    if not 0 < physical <= 1 << 24:
        return (0, 0)
    values = table[request, begin : begin + count].astype(np.int64)
    if values.min() < 0 or values.max() >= physical:
        return (0, 0)
    direction = -1 if values[-1] < values[0] else 1
    if not np.array_equal(values, values[0] + direction * np.arange(count)):
        return (0, 0)
    return (int(values[0]), direction)


@pytest.mark.parametrize("count", [1, 31, 32, 33, 127, 255, 256])
@pytest.mark.parametrize("direction", [1, -1])
def test_every_middle_page_and_reverse_direction(count, direction):
    proof = leaf_proof()
    table = (1000 + direction * np.arange(320)).astype(np.int32).reshape(1, -1)
    args = (0, 17, count, 2048)
    assert proof(table, *args) == oracle(table, *args)
    for lane in range(17, 17 + count):
        for value in (-1, 2048, np.iinfo(np.int32).max, int(table[0, lane]) + 1):
            altered = table.copy()
            altered[0, lane] = value
            assert proof(altered, *args) == oracle(altered, *args)


def test_bounds_and_exact_integer_limit():
    proof = leaf_proof()
    table = ((1 << 24) - 320 + np.arange(320)).astype(np.int32).reshape(1, -1)
    for request in (-1, 0, 1):
        for begin in (-1, 0, 64, 319, 320):
            for count in (0, 1, 32, 256, 257):
                for physical in (0, 1 << 24, (1 << 24) + 1):
                    args = (request, begin, count, physical)
                    assert proof(table, *args) == oracle(table, *args)
