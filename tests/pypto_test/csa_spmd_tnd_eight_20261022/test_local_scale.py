# SPDX-License-Identifier: Apache-2.0
"""Check local-contiguity proof and exact fallback bytes against the old reader."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_scale_gather import legacy_bytes


def helpers():
    nodes = []
    for node in ast.parse(Path(__file__).with_name("local_scale_source.py").read_text()).body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list, node.returns = [], None
            for arg in node.args.args:
                arg.annotation = None
            nodes.append(node)

    def gather(destination, source, dst, src, shape):
        result = destination.copy()
        result[dst[0] : dst[0] + shape[0], dst[1] : dst[1] + shape[1]] = source[
            src[0] : src[0] + shape[0], src[1] : src[1] + shape[1]
        ]
        return result

    tile = SimpleNamespace(
        ci=lambda begin, shape, dtype: np.arange(begin, begin + np.prod(shape), dtype=dtype).reshape(shape),
        full=lambda shape, dtype, value: np.full(shape, value, dtype=dtype),
        reinterpret_view=lambda value, dtype: value.view(dtype),
        reshape=lambda value, shape: value.reshape(shape),
        read=lambda value, index: value[tuple(index)],
    )
    pl = SimpleNamespace(
        tile=tile,
        INDEX=int,
        INT32=np.int32,
        INT8=np.int8,
        FP32=np.float32,
        FP16=np.float16,
        PadValue=SimpleNamespace(zero=0),
        tensor=SimpleNamespace(dim=lambda value, axis: value.shape[axis]),
        cast=lambda value, dtype: dtype(value),
        range=range,
        min=min,
        max=max,
        read=lambda value, index: value[tuple(index)],
        load=lambda value, index, shape: value[index[0] : index[0] + shape[0], index[1] : index[1] + shape[1]].copy(),
        add=np.add,
        sub=np.subtract,
        abs=np.abs,
        col_expand_mul=np.multiply,
        set_validshape=lambda value, rows, cols: value[:rows, :cols],
        create_tile=lambda shape, dtype: np.empty(shape, dtype=dtype),
        row_max=lambda value, scratch: value.max(axis=1, keepdims=True),
        fillpad=lambda value, pad_value: value,
        gather_row=gather,
    )
    namespace = dict(pl=pl)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "local_scale_helpers", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("stride", [4160, 4224, 4352])
@pytest.mark.parametrize("columns", [0, 1, 7, 31, 32, 33, 65])
def test_local_read_and_original_fallback(stride, columns):
    rng = np.random.default_rng(1024)
    cache = rng.integers(0, 256, size=(41, stride), dtype=np.uint8)
    table = np.stack([np.arange(columns) % 41, np.arange(columns)[::-1] % 41, np.arange(columns) % 41]).astype(np.int32)
    if columns:
        table[2, ::3] = -1
        table[2, 1::7] = np.iinfo(np.int32).max
        table[2, 2::11] = 41
    reader = helpers()["read_local_scales"]
    for batch in range(3):
        for length in (0, 1, 31, 32, 33, 1024, 1025, 2080):
            for begin in (-32, 0, 32, 1024, 2048):
                actual = reader(cache, table, batch, begin, length, 1024).reshape(-1).view(np.uint8)
                np.testing.assert_array_equal(actual, legacy_bytes(cache, table, batch, begin, length))


@pytest.mark.parametrize("first", [0, 41, (1 << 24) - 33, (1 << 24) - 32])
def test_proof_checks_every_page_with_exact_integer_bounds(first):
    proof = helpers()["contiguous_scale_block"]
    table = np.arange(first, first + 32, dtype=np.int32).reshape(1, 32)
    assert proof(table, 0, 0, 1024, 1 << 24, 1024) == first
    for lane in range(32):
        for wrong in (-1, np.iinfo(np.int32).max, int(table[0, lane]) + 1):
            revised = table.copy()
            revised[0, lane] = wrong
            assert proof(revised, 0, 0, 1024, 1 << 24, 1024) == -1
    assert proof(table, 0, 0, 1024, (1 << 24) + 1, 1024) == -1
