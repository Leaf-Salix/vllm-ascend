# SPDX-License-Identifier: Apache-2.0
"""Execute the actual retiled helper against independent per-token/head math.

NumPy checks addresses, operation order, ownership and the old tail branch.
It cannot establish Ascend instruction rounding or device event correctness.
"""

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/query_local_v5"


def bf16(value):
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def actual_helper(branches, writes):
    tree = ast.parse((SOURCE / "query_local_stages.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    node.decorator_list, node.returns = [], None
    for arg in node.args.args:
        arg.annotation = None

    def load(tensor, offsets, shape, **kwargs):
        assert len(offsets) == tensor.ndim == len(shape)
        assert all(off >= 0 and off + size <= dim for off, size, dim in zip(offsets, shape, tensor.shape))
        return tensor[tuple(slice(o, o + s) for o, s in zip(offsets, shape))].copy()

    def store(tile, offsets, tensor):
        writes.append((tuple(offsets), tile.shape))
        assert all(off >= 0 and off + size <= dim for off, size, dim in zip(offsets, tile.shape, tensor.shape))
        tensor[tuple(slice(o, o + s) for o, s in zip(offsets, tile.shape))] = tile

    def cast(value, dtype, **kwargs):
        return bf16(value) if dtype == "bf16" else np.asarray(value).astype(dtype)

    def rsqrt(value, tmp):
        branches.append("full")
        return np.reciprocal(np.sqrt(value))

    def reciprocal(value):
        branches.append("tail")
        return np.reciprocal(value)

    pl = SimpleNamespace(
        range=range,
        tensor=SimpleNamespace(dim=lambda a, axis: a.shape[axis]),
        FP32=np.float32,
        INT32=np.int32,
        BF16="bf16",
        MemorySpace=SimpleNamespace(Vec=None),
        read=lambda a, ix: a[tuple(ix)],
        load=load,
        store=store,
        cast=cast,
        reshape=np.reshape,
        mul=np.multiply,
        add=np.add,
        sqrt=np.sqrt,
        recip=reciprocal,
        col_expand=lambda dst, row: np.broadcast_to(row, dst.shape).copy(),
        col_expand_mul=np.multiply,
        row_expand_mul=np.multiply,
        row_expand_add=np.add,
        row_sum=lambda a, tmp_tile: np.sum(a, axis=1, keepdims=True, dtype=np.float32),
        tile=SimpleNamespace(
            create=lambda shape, dtype: np.empty(shape, dtype=np.float32 if dtype == "bf16" else dtype),
            arange=lambda start, shape, dtype: np.arange(start, start + np.prod(shape), dtype=dtype).reshape(shape),
            gather=lambda a, ix, tmp: a.ravel()[ix],
            rsqrt=rsqrt,
        ),
    )
    namespace = dict(
        pl=pl,
        H=64,
        HEAD_DIM=512,
        NOPE_DIM=448,
        ROPE_DIM=64,
        ROPE_DIM_SCALE=64.0,
        EPS=1e-6,
        Q_ROPE_T_TILE=8,
        QUERY_HEAD_TILE=8,
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "query_local_actual", "exec"),
        namespace,
    )
    return namespace[node.name]


@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 23, 24, 25, 63, 64, 65, 95, 96, 97, 144, 168])
def test_actual_math_and_unique_coverage(tokens):
    rng = np.random.default_rng(1024 + tokens)
    acc = rng.integers(-1048576, 1048576, size=(tokens, 64, 512), dtype=np.int32)
    act_scale = rng.uniform(1e-5, 0.1, size=(tokens, 1)).astype(np.float32)
    weight_scale = rng.uniform(1e-4, 0.1, size=(64, 512)).astype(np.float32)
    cos = rng.uniform(-1, 1, size=(tokens, 64)).astype(np.float32)
    sin = rng.uniform(-1, 1, size=(tokens, 64)).astype(np.float32)
    swap = np.broadcast_to(np.arange(64, dtype=np.int32) ^ 1, (tokens, 64)).copy()
    backing = np.full((tokens * 64 + 2, 512), -4321.25, dtype=np.float32)
    actual = backing[1:-1]
    branches, writes = [], []
    helper = actual_helper(branches, writes)
    # Real Attention ownership, with nontrivial order and partial final groups.
    for core in range(24):
        for token in range(core, tokens, 24):
            for lane in (1, 0):
                before = len(branches)
                helper(
                    weight_scale.ravel(),
                    cos,
                    sin,
                    swap,
                    actual,
                    act_scale,
                    acc.reshape(tokens, -1),
                    token,
                    lane,
                )
                assert branches[before:] == ["full" if token < tokens // 8 * 8 else "tail"] * 4
    dq = (acc.astype(np.float32) * act_scale[:, None, :]) * weight_scale[None, :, :]
    inv = np.reciprocal(np.sqrt(np.sum(dq * dq, axis=2, keepdims=True) * np.float32(1 / 512) + np.float32(1e-6)))
    normed = dq * inv
    expected = np.empty_like(normed)
    expected[:, :, :448] = bf16(normed[:, :, :448])
    rope = normed[:, :, 448:]
    expected[:, :, 448:] = bf16(rope * cos[:, None, :] + rope[:, :, np.arange(64) ^ 1] * sin[:, None, :])
    np.testing.assert_array_equal(actual.view(np.uint32), expected.reshape(-1, 512).view(np.uint32))
    coverage = np.zeros(actual.shape, dtype=np.uint8)
    for (row, col), (nr, nc) in writes:
        coverage[row : row + nr, col : col + nc] += 1
    assert np.all(coverage == 1)
    assert np.all(backing[[0, -1]] == -4321.25)


@pytest.mark.parametrize("tokens", [6, 24, 48, 56, 63, 64, 96, 128, 144, 168, 192, 193])
def test_policy_preserves_all_retained_features(tokens):
    original = runpy.run_path(str(HERE / "runs/merged_schedule_v1/policy.py"))["select_policy"]
    candidate = runpy.run_path(str(SOURCE / "policy.py"))["select_policy"]
    args = dict(tokens=tokens, requests=(tokens + 5) // 6, max_query_tokens=6, max_seq_len=131078)
    actual = candidate(**args).constexprs()
    enabled = actual.pop("LOCAL_QUERY_PREP")
    assert enabled == (64 <= tokens <= 192)
    assert actual == original(**args).constexprs()
