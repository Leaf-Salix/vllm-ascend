# SPDX-License-Identifier: Apache-2.0
"""Execute generated QK slices/inserts against the unsplit integer/FP16 oracle.

This checks row placement and the mathematical intermediate. Physical fractal
layout, asynchronous reuse and full-chain performance still require the device.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).with_name("runs") / "score_n128_two_acc_v1/indexer_score_stages.py"
pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason="Prepare the isolated N128 two-Acc candidate first")


@pytest.mark.parametrize("live_half", [False, True])
@pytest.mark.parametrize("pattern", ["random", "positive_limit", "negative_limit", "alternating"])
def test_qk_intermediate_row_placement(live_half, pattern):
    tree = ast.parse(SOURCE.read_text())
    statements = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)]
    prefix = "live_half_split_" if live_half else "split_"
    halves = 1 if live_half else 2
    setup = [
        n
        for n in statements
        if n.targets[0].id
        in {prefix + "qk_acc_" + s for s in ("large", "small")}
        | {prefix + "query" + str(h) + "_" + s for h in range(halves) for s in ("large", "small")}
    ]
    assert len(setup) == 2 + 2 * halves
    lines = SOURCE.read_text().splitlines()
    blocks = []
    for half in range(halves):
        mm = next(n for n in statements if n.targets[0].id == prefix + "qk" + str(half) + "_large")
        blocks.extend(ast.parse("\n".join(line.strip() for line in lines[mm.lineno - 1 : mm.lineno + 3])).body)
    rng = np.random.default_rng(1024)
    query = rng.integers(-128, 128, size=(192 * halves, 128), dtype=np.int8)
    key = rng.integers(-128, 128, size=(128, 128), dtype=np.int8)
    if pattern == "positive_limit":
        query.fill(-128)
        key.fill(-128)
    elif pattern == "negative_limit":
        query.fill(-128)
        key.fill(127)
    elif pattern == "alternating":
        query[::2] = -128
        query[1::2] = 127
        key[:, ::2] = -128
        key[:, 1::2] = 127

    def assemble(dst, src, offset, *, pre_quant, pre_relu):
        assert pre_relu
        result = dst.copy()
        rows, cols = src.shape
        converted = (np.maximum(src, 0).astype(np.float32) * pre_quant).astype(np.float16)
        result[offset[0] : offset[0] + rows, offset[1] : offset[1] + cols] = converted
        return result

    tile = SimpleNamespace(
        create=lambda shape, dtype, target_memory: np.zeros(shape, dtype=dtype),
        extract=lambda src, row, col, shape, target_memory: src[row : row + shape[0], col : col + shape[1]].copy(),
        matmul_acc=lambda acc, lhs, rhs, init_cond: lhs.astype(np.int32) @ rhs.astype(np.int32),
        assemble=assemble,
    )
    pl = SimpleNamespace(tile=tile, INT32=np.int32, MemorySpace=SimpleNamespace(Acc="acc", Left="left"))
    namespace = dict(
        pl=pl,
        SPLIT_QK_ROWS=192,
        IDX_HEAD_DIM=128,
        qk_panel_cols=128,
        BUFFERED_SCORE_SCALE=1024,
        buf_query_vector=query,
        live_half_buf_query_vector=query,
        buf_key_panel=key,
        live_half_buf_key_panel=key,
    )
    for half in range(halves):
        namespace[prefix + "current" + str(half)] = np.zeros((192, 128), dtype=np.float16)
    exec(compile(ast.Module(body=setup + blocks, type_ignores=[]), "two_acc_blocks", "exec"), namespace)
    expected = (np.maximum(query.astype(np.int32) @ key.astype(np.int32), 0).astype(np.float32) / 1024).astype(
        np.float16
    )
    actual = np.concatenate([namespace[prefix + "current" + str(h)] for h in range(halves)])
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


def test_ws_reductions_and_fifo_are_identical_to_parent():
    parent = SOURCE.parent.parent / "score_n128_single_acc_v3/indexer_score_stages.py"

    def calls(path):
        tree = ast.parse(path.read_text())
        ws = [
            ast.unparse(n)
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id.startswith(("split_ws", "live_half_split_ws"))
        ]
        fifo = [
            ast.unparse(n)
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and ast.unparse(n.func) in ("pl.system.sync_wait", "pl.system.sync_set")
        ]
        return ws, fifo

    assert calls(SOURCE) == calls(parent)
