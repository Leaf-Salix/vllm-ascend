# SPDX-License-Identifier: Apache-2.0
"""Validate guarded pure-DMA control and preserve the checked proof producer."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_attention_overlap import (
    BASE,
    ExecuteCarriedLoops,
    assert_event_dag,
    assert_fast_math_and_fallback,
    compiled_projection,
    event_trace,
    function,
)

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/attention_valid_overlap_v1/attention_stream_stages.py"
FUNCTION = "attention_qk_pv_vector_valid_overlap_stage"


@pytest.mark.parametrize("tokens", [96, 144])
@pytest.mark.parametrize("proof", [0, 1])
def test_guarded_event_sequences_for_all_two_query_masks(tokens, proof):
    cube = compiled_projection(BASE, "attention_qk_pv_cube_stage")
    old = compiled_projection(BASE, "attention_qk_pv_vector_lane_stage")
    new = compiled_projection(SOURCE, FUNCTION)
    for pattern in range(1 << 10):
        mask = [[0] * 16 for _ in range(tokens)]
        for work in range(10):
            query, block = (work // 5) * 24, work % 5
            mask[query][block] = (pattern >> work) & 1
            if block:
                mask[query][7 + block - 1] = proof
        traces = [event_trace(cube, tokens, mask)]
        for lane in (0, 1):
            before, after = event_trace(old, tokens, mask, lane), event_trace(new, tokens, mask, lane)
            assert before == after
            traces.append(after)
        assert_event_dag(traces)


def test_exact_proof_producer_and_unchanged_math():
    proof = HERE / "runs/plan_gather_valid_v1/attention_plan_stages.py"
    assert (SOURCE.parent / proof.name).read_bytes() == proof.read_bytes()
    assert_fast_math_and_fallback(SOURCE, FUNCTION)
    for name in ("attention_qk_pv_cube_stage", "attention_qk_pv_vector_lane_stage"):
        assert ast.dump(function(BASE, name)) == ast.dump(function(SOURCE, name))


@pytest.mark.parametrize("proof", [0, 1])
@pytest.mark.parametrize("physical", [0, 1])
def test_actual_entry_predicate_requires_current_proof(proof, physical):
    fn = function(SOURCE, FUNCTION)
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    selected = next(
        i for i, n in enumerate(loop.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "overlap_ready != 0"
    )
    predicate = ast.Module(body=copy.deepcopy(loop.body[:selected]), type_ignores=[])
    code = compile(ast.fix_missing_locations(predicate), "actual_valid_overlap_predicate", "exec")
    pl = SimpleNamespace(
        cast=lambda value, dtype: int(value), INT32=int, read=lambda array, index: array[index[0]][index[1]]
    )
    for tick in range(33):
        for current_valid, previous_valid in ((0, 0), (0, 1), (1, 0), (1, 1)):
            mask = [[0] * 16 for _ in range(168)]
            if tick < 30:
                mask[(tick // 5) * 24][tick % 5] = current_valid
                if tick % 5:
                    mask[(tick // 5) * 24][7 + tick % 5 - 1] = proof
            if 0 < tick <= 30:
                mask[((tick - 1) // 5) * 24][(tick - 1) % 5] = previous_valid
            namespace = dict(
                pl=pl,
                qk_tick=tick,
                qk_core=0,
                qk_block_count=30,
                CORE_GROUPS=24,
                SPARSE_BLOCKS=5,
                PLAN_PHYSICAL_ROWS=True,
                physical_mode=physical,
                valid_block_mask=mask,
                COMPRESSED_ALL_VALID_COL0=7,
            )
            exec(code, namespace)
            expected = 0 < tick < 30 and tick % 5 > 0 and physical and proof and current_valid and previous_valid
            assert bool(namespace["overlap_ready"]) == bool(expected)


def test_actual_unchecked_reader_only_with_valid_rows():
    fn = function(SOURCE, FUNCTION)
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    branch = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == "overlap_ready != 0")
    reader = copy.deepcopy(next(n for n in branch.body if isinstance(n, ast.For)))
    module = ExecuteCarriedLoops().visit(ast.Module(body=[reader], type_ignores=[]))
    code = compile(ast.fix_missing_locations(module), "actual_valid_reader", "exec")
    rng = np.random.default_rng(20261023)
    cache = rng.integers(0, 65536, (256, 512), dtype=np.uint16)

    def gather(dst, src, dst_offset, src_offset, shape):
        dr, dc = dst_offset
        sr, sc = src_offset
        nr, nc = shape
        assert 0 <= sr < len(src)
        dst[dr : dr + nr, dc : dc + nc] = src[sr : sr + nr, sc : sc + nc]
        return dst

    pl = SimpleNamespace(
        cast=lambda value, dtype: int(value),
        INDEX=int,
        read=lambda array, index: int(array[tuple(index)]),
        gather_row=gather,
        store=lambda tile, offset, dst: gather(dst, tile, offset, [0, 0], list(tile.shape)),
    )
    for lane in (0, 1):
        state = rng.integers(0, 65536, (32, 512), dtype=np.uint16)
        for _ in range(8):
            indices = rng.integers(0, len(cache), 64)
            output = np.full((160, 512), 0xA55A, np.uint16)
            expected = output.copy()
            expected[8 + lane * 64 : 8 + (lane + 1) * 64] = cache[indices]
            namespace = dict(
                pl=pl,
                dma_kv_iter=state,
                KV_DMA_ROWS=32,
                HEAD_DIM=512,
                physical_rows=indices[None, :],
                cmp_kv_flat=cache,
                qk_kv_row=8,
                qk_lane_kv=lane * 64,
                kv_transfer=output,
            )
            exec(code, namespace)
            state = namespace["dma_finished"]
            np.testing.assert_array_equal(output, expected)
