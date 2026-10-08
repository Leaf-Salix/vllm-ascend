# SPDX-License-Identifier: Apache-2.0
"""Execute scalar metadata producers with checked CPU buffers, without an NPU."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

KERNELS = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa"


def _producer(filename, name):
    tree = ast.parse((KERNELS / filename).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))

    def read(buffer, indices):
        assert all(0 <= index < size for index, size in zip(indices, buffer.shape)), indices
        return int(buffer[tuple(indices)])

    def write(buffer, indices, value):
        assert all(0 <= index < size for index, size in zip(indices, buffer.shape)), indices
        buffer[tuple(indices)] = value

    language = SimpleNamespace(
        range=range,
        cast=lambda value, dtype: int(value),
        INT32=int,
        INDEX=int,
        min=min,
        read=read,
        write=write,
        tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]),
    )
    namespace = {"pl": language}
    exec(compile(module, str(KERNELS / filename), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "lengths,active,capacity",
    [
        ([6] * 4, [True] * 4, 24),
        ([1, 3, 6, 2], [True] * 4, 17),
        ([3, 0, 1], [True, False, True], 12),
        ([1] * 63 + [321], [True] * 63 + [False], 384),
        ([24], [False], 24),
    ],
)
def test_request_mapping_and_groups_respect_active_boundaries(lengths, active, capacity):
    bounds = np.array([0, *np.cumsum(lengths)], dtype=np.int32)
    seq_lens = np.array([8192 if value else 0 for value in active], dtype=np.int32)
    mapping = np.full(capacity, 999, dtype=np.int32)
    _producer("compact_metadata.py", "build_token_request")(bounds, seq_lens, mapping)
    expected = np.full(capacity, -1, dtype=np.int32)
    for request, valid in enumerate(active):
        if valid:
            expected[bounds[request] : bounds[request + 1]] = request
    np.testing.assert_array_equal(mapping, expected)
    for size in (2, 6):
        rows, valid_rows = np.full(432, -99, dtype=np.int32), np.full(432, -99, dtype=np.int32)
        count = np.full(1, -1, dtype=np.int32)
        _producer("decode_indexer.py", "indexer_build_query_groups")(
            bounds,
            seq_lens,
            rows,
            valid_rows,
            count,
            size,
        )
        pairs = [
            (start, min(size, bounds[request + 1] - start))
            for request, valid in enumerate(active)
            if valid
            for start in range(bounds[request], bounds[request + 1], size)
        ]
        assert count[0] == len(pairs)
        assert list(zip(rows[: count[0]], valid_rows[: count[0]])) == pairs
        assert _producer("decode_indexer.py", "indexer_query_group_count")(bounds, seq_lens, size) == len(pairs)
        assert count[0] <= 432 // size


def test_replay_does_not_retain_old_token_requests():
    producer = _producer("compact_metadata.py", "build_token_request")
    mapping = np.empty(24, dtype=np.int32)
    producer(np.array([0, 6, 12, 18, 24]), np.full(4, 8192), mapping)
    producer(np.array([0, 1, 3, 24]), np.array([8192, 8192, 0]), mapping)
    np.testing.assert_array_equal(mapping, [0, 1, 1] + [-1] * 21)


@pytest.mark.parametrize(
    "tokens,requests,capacity,max_query,expected",
    [
        (18, 4, 24, 6, True),
        (10, 3, 16, 6, True),
        (1, 1, 1, 1, True),
        (0, 0, 24, 0, True),
        (10, 1, 16, 10, False),
        (25, 4, 24, 6, False),
    ],
)
def test_tnd_graph_gate_uses_real_query_length(tokens, requests, capacity, max_query, expected):
    tree = ast.parse((KERNELS / "service_config.py").read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "can_replay_csa_graph"
    )
    function.returns = None
    namespace = {"MAX_BATCH_SIZE": 64, "QUERY_TOKENS": 6}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "service_config.py", "exec"),
        namespace,
    )
    gate = namespace["can_replay_csa_graph"]
    assert (
        gate(
            num_tokens=tokens,
            num_reqs=requests,
            uniform_decode=False,
            padded_tokens=capacity,
            variable_queries=True,
            max_query_tokens=max_query,
        )
        is expected
    )
    if expected and tokens != requests * 6 and capacity % 6 == 0:
        assert gate(num_tokens=tokens, num_reqs=requests, uniform_decode=False, padded_tokens=capacity) is False
