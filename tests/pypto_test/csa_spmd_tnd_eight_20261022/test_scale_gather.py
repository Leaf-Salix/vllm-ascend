# SPDX-License-Identifier: Apache-2.0
"""Execute the actual helper with CPU tile semantics against the legacy byte reader.

This checks addresses, clamped tails and invalid pages. It does not certify ISA
lowering, cache visibility or performance; those require compiled/device checks.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def memory_gather(mem, indices, *, coalesce, gather_oob):
    assert gather_oob == "zero"
    ids = indices.reshape(-1)
    if coalesce == "row":
        result = np.zeros((len(ids), mem.shape[1]), dtype=mem.dtype)
        valid = (ids >= 0) & (ids < mem.shape[0])
        result[valid] = mem[ids[valid]]
    else:
        assert coalesce == "elem"
        result = np.zeros(indices.shape, dtype=mem.dtype)
        valid = (indices >= 0) & (indices < mem.size)
        result[valid] = mem.reshape(-1)[indices[valid]]
    return result


def execute_helper():
    source = Path(__file__).with_name("scale_gather_source.py").read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    function.decorator_list = []
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None

    def sel(mask, lhs, rhs, tmp):
        assert tmp.dtype == np.uint32 and tmp.size >= 16
        return np.where(mask, lhs, rhs)

    tile = SimpleNamespace(
        ci=lambda start, shape, dtype: np.arange(start, start + np.prod(shape), dtype=dtype).reshape(shape),
        create=lambda shape, dtype: np.empty(shape, dtype=dtype),
        full=lambda shape, dtype, value: np.full(shape, value, dtype=dtype),
        sel=sel,
        mgather=memory_gather,
        reshape=lambda value, shape: value.reshape(shape),
        reinterpret_view=lambda value, dtype: value.view(dtype),
    )
    ops = [np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal]
    pl = SimpleNamespace(
        tile=tile,
        tensor=SimpleNamespace(dim=lambda value, axis: value.shape[axis]),
        INT32=np.int32,
        UINT32=np.uint32,
        FP16=np.float16,
        FP32=np.float32,
        cast=lambda value, dtype: dtype(value),
        add=np.add,
        mul=np.multiply,
        minimum=np.minimum,
        max=max,
        and_=np.bitwise_and,
        cmp=lambda a, b, cmp_type: ops[cmp_type](a, b),
    )
    namespace = {"pl": pl}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "scale_gather_helper", "exec"), namespace)
    return namespace[function.name]


def legacy_bytes(cache, table, batch, score_begin, cache_len):
    result = np.zeros(2048, dtype=np.uint8)
    for lane in range(32):
        logical = min(score_begin // 32 + lane, max((cache_len - 1) // 32, 0))
        if 0 <= logical < table.shape[1]:
            physical = int(table[batch, logical])
            if 0 <= physical < cache.shape[0]:
                result[lane * 64 : (lane + 1) * 64] = cache[physical, 4096:4160]
    return result


@pytest.mark.parametrize("page_bytes", [4160, 4224, 4352])
@pytest.mark.parametrize("columns", [0, 1, 7, 31, 32, 33, 65])
def test_scale_page_bytes_and_boundaries(page_bytes, columns):
    rng = np.random.default_rng(1024 + columns)
    cache = rng.integers(0, 256, size=(41, page_bytes), dtype=np.uint8)
    scale_rows = cache.reshape(-1, 64)
    table = rng.integers(0, 41, size=(3, columns), dtype=np.int32)
    if columns:
        table[0] = np.arange(columns, dtype=np.int32) % 41
        table[1] = table[0, ::-1]
        table[2, ::3] = -1
        table[2, 1::7] = np.iinfo(np.int32).max
        table[2, 2::11] = 41
    helper = execute_helper()
    for batch in range(3):
        for cache_len in (0, 1, 31, 32, 33, 1024, 1025, 2080):
            for score_begin in (-32, 0, 32, 1024, 2048):
                actual = helper(scale_rows, table, batch, score_begin, cache_len, len(cache), page_bytes, 1024)
                expected = legacy_bytes(cache, table, batch, score_begin, cache_len)
                np.testing.assert_array_equal(actual.reshape(-1).view(np.uint8), expected)


@pytest.mark.parametrize("capacity", [1, 2, 31, 32, 65535, (1 << 24) - 1])
def test_float_predicates_preserve_bounded_integer_membership(capacity):
    values = np.array(
        [-(1 << 31), -1, 0, 1, capacity - 1, capacity, capacity + 1, (1 << 24), (1 << 31) - 1],
        dtype=np.int32,
    )
    expected = (values >= 0) & (values < capacity)
    converted = values.astype(np.float32)
    actual = (converted >= 0.0) & (converted < np.float32(capacity))
    np.testing.assert_array_equal(actual, expected)
