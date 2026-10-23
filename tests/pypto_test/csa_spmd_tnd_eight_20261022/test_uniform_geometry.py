# SPDX-License-Identifier: Apache-2.0
"""Check the executed helper source against an independent owner-set oracle."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def geometry():
    path = Path(__file__).with_name("runs") / "uniform_geometry_v1/request_geometry.py"
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list, node.returns = [], None
            for arg in node.args.args:
                arg.annotation = None
            nodes.append(node)
    read = lambda data, index: int(data[tuple(index)])  # noqa: E731
    namespace = dict(
        COMPRESS_RATIO=4,
        pl=SimpleNamespace(
            INDEX=int, INT32=int, cast=lambda value, dtype: dtype(value), range=range,
            tensor=SimpleNamespace(dim=lambda data, axis: data.shape[axis], read=read), read=read,
        ),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def oracle(bounds, lengths, query, capacity):
    owners = []
    if 0 <= query < capacity:
        for request, length in enumerate(lengths):
            if request + 1 >= len(bounds):
                continue
            begin, end = map(int, bounds[request:request + 2])
            if length > 0 and 0 <= begin <= query < end <= capacity and end - begin <= length:
                owners.append((request, begin, end, 1))
    return owners[0] if len(owners) == 1 else (-1, 0, 0, 0)


def assert_equivalent(geometry, bounds, lengths, capacity):
    bounds, lengths = np.array(bounds, np.int32), np.array(lengths, np.int32)
    width = geometry["uniform_query_width"](bounds, lengths, capacity)
    for query in range(-1, capacity + 2):
        assert geometry["query_request_with_uniform"](bounds, lengths, query, capacity, width) == oracle(
            bounds, lengths, query, capacity
        )
    return width


@pytest.mark.parametrize("batch", [1, 2, 8, 16, 24, 28, 32])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6, 7, 16])
def test_dense_equal_query_and_equal_kv(geometry, batch, width):
    bounds = np.arange(batch + 1) * width
    lengths = np.full(batch, 131072 + width)
    assert assert_equivalent(geometry, bounds, lengths, batch * width) == (width if batch > 1 else 0)


@pytest.mark.parametrize("batch", [2, 16, 24])
def test_every_boundary_and_length_is_proved(geometry, batch):
    original_bounds = np.arange(batch + 1) * 6
    original_lengths = np.full(batch, 131078)
    for index in range(batch + 1):
        for value in [-1, 0, int(original_bounds[index]) + 1, 2**31 - 1]:
            bounds = original_bounds.copy()
            bounds[index] = value
            width = assert_equivalent(geometry, bounds, original_lengths, batch * 6)
            assert width == (6 if np.array_equal(bounds, original_bounds) else 0)
    for index in range(batch):
        for value in [-1, 0, 1, 131077, 2**31 - 1]:
            lengths = original_lengths.copy()
            lengths[index] = value
            assert assert_equivalent(geometry, original_bounds, lengths, batch * 6) == 0


@pytest.mark.parametrize("bounds,lengths,capacity", [
    ([], [], 0), ([0], [], 0), ([0], [131078], 6), ([0, 6], [131078, 131078], 12),
    ([0, 6, 12], [131078, 131078], 16), ([0, 6, 12], [131078, 131078], 11),
    ([0, 6, 8, 14], [131078] * 3, 14), ([0, 6, 6, 12], [131078, 0, 131078], 12),
    ([0, 6, 3, 12], [131078] * 3, 12), ([0, 6, 12], [5, 5], 12),
])
def test_padding_mixed_overlap_and_short_storage_fallback(geometry, bounds, lengths, capacity):
    assert assert_equivalent(geometry, bounds, lengths, capacity) == 0


def test_same_storage_rechecks_width_and_kv_each_invocation(geometry):
    bounds = np.arange(17, dtype=np.int32) * 6
    lengths = np.full(16, 131078, np.int32)
    proof = geometry["uniform_query_width"]
    assert proof(bounds, lengths, 96) == 6
    bounds[8] -= 1
    assert proof(bounds, lengths, 96) == 0
    bounds[8] += 1
    assert proof(bounds, lengths, 96) == 6
    lengths[7] += 1
    assert proof(bounds, lengths, 96) == 0
    lengths[7] -= 1
    assert proof(bounds, lengths, 96) == 6


def test_random_adversarial_query_maps(geometry):
    rng = np.random.default_rng(1024)
    for _ in range(100):
        batch = int(rng.integers(1, 25))
        capacity = batch * 6
        bounds = rng.integers(-2, capacity + 3, batch + 1)
        lengths = rng.integers(-2, capacity * 2 + 1, batch)
        assert_equivalent(geometry, bounds, lengths, capacity)
