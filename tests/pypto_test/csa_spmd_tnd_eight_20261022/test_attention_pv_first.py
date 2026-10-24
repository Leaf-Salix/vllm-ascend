# SPDX-License-Identifier: Apache-2.0
"""Prove source event pairing, ring ownership and unchanged branch computations."""

import ast
import collections
from pathlib import Path

import pytest
from test_attention_overlap import assert_event_dag, compiled_projection, function
from test_head_partition import trace

HERE = Path(__file__).resolve().parent
BASE = HERE / "runs/output_rectangle_production_v1"
SOURCE = HERE / "runs/attention_pv_first_v1"
STREAM = SOURCE / "attention_stream_stages.py"


def assert_l1_ring_ownership(cube_trace):
    slots, finished = {}, set()
    for operation, event, tick in cube_trace:
        if (operation, event) == ("wait", 2):
            # The L1 write follows KV_READY and precedes SCORE_READY. Its slot
            # must have completed the previous PV before being overwritten.
            prior = slots.get(tick % 3)
            assert prior is None or prior in finished
            slots[tick % 3] = tick
        elif (operation, event) == ("wait", 4):
            work = tick - 2
            assert slots[work % 3] == work
        elif (operation, event) == ("set", 5):
            work = tick - 2
            assert slots[work % 3] == work
            finished.add(work)
    assert finished == {tick for operation, event, tick in cube_trace if (operation, event) == ("wait", 2)}


@pytest.mark.parametrize("tokens", [0, 1, 23, 24, 25, 48, 49, 96, 144, 168, 192])
def test_every_two_query_mask_and_multiquery_drain(tokens):
    old_cube = compiled_projection(BASE / "attention_stream_stages.py", "attention_qk_pv_cube_stage")
    cube = compiled_projection(STREAM, "attention_qk_pv_cube_stage")
    vector = compiled_projection(STREAM, "attention_qk_pv_vector_lane_stage")
    for core in (0, 23):
        work_count = max(0, (tokens + 23 - core) // 24) * 5
        varied = min(10, work_count)
        for pattern in range(1 << varied):
            mask = [[0] * 16 for _ in range(tokens)]
            for work in range(varied):
                mask[core + (work // 5) * 24][work % 5] = (pattern >> work) & 1
            actual = trace(cube, tokens, mask, core)
            before = trace(old_cube, tokens, mask, core)
            assert collections.Counter(actual) == collections.Counter(before)
            traces = [actual] + [trace(vector, tokens, mask, core, lane) for lane in (0, 1)]
            assert_event_dag(traces)
            assert_l1_ring_ownership(actual)
        for pattern in ("all", "odd", "empty"):
            mask = [[0] * 16 for _ in range(tokens)]
            for work in range(work_count):
                mask[core + (work // 5) * 24][work % 5] = int(pattern == "all" or (pattern == "odd" and work % 2))
            actual = trace(cube, tokens, mask, core)
            assert_event_dag([actual] + [trace(vector, tokens, mask, core, lane) for lane in (0, 1)])
            assert_l1_ring_ownership(actual)


def test_each_original_math_branch_and_vector_body_is_unchanged():
    old = function(BASE / "attention_stream_stages.py", "attention_qk_pv_cube_stage")
    new = function(STREAM, "attention_qk_pv_cube_stage")
    loop = next(n for n in new.body if isinstance(n, ast.For))
    assert [ast.unparse(n.test) for n in loop.body] == [
        "qk_tick >= QK_PRE_LAUNCH",
        "qk_tick < qk_block_count",
    ]
    loop.body.reverse()
    assert ast.dump(new) == ast.dump(old)
    name = "attention_qk_pv_vector_lane_stage"
    assert ast.dump(function(STREAM, name)) == ast.dump(function(BASE / "attention_stream_stages.py", name))
    for path in BASE.glob("*.py"):
        if path.name != "attention_stream_stages.py":
            assert path.read_bytes() == (SOURCE / path.name).read_bytes()


def test_ring_verifier_rejects_reuse_before_pv():
    with pytest.raises(AssertionError):
        assert_l1_ring_ownership([("wait", 2, 0), ("set", 3, 0), ("wait", 2, 3)])
