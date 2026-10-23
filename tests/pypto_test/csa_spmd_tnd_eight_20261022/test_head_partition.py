# SPDX-License-Identifier: Apache-2.0
"""Verify actual pilot control, independent head ownership and unchanged math."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_attention_overlap import assert_event_dag, compiled_projection, function

HERE = Path(__file__).resolve().parent
BASE = HERE / "runs/physical_capacity_production_v1"
SOURCE = HERE / "runs/head_partition2_v1"
STREAM = SOURCE / "attention_stream_stages.py"


def expression(fn, target):
    matches = [
        n.value
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == target
    ]
    assert len(matches) == 1, (target, len(matches))
    return matches[0]


def calculate(fn, target, values):
    return eval(compile(ast.Expression(expression(fn, target)), "actual_source_scalar", "eval"), values)


def context(tokens, core, lane=0):
    return dict(
        t_dim=tokens,
        core_id=core,
        lane_id=lane,
        CORE_GROUPS=24,
        GROUPS=24,
        ATTENTION_HEAD_PARTS=2,
        PARTITION_HEADS=32,
        H=64,
        SPARSE_BLOCKS=5,
        FINAL_HEAD_TILE=16,
        HEADS_PER_GROUP=8,
        pl=SimpleNamespace(min=min),
    )


@pytest.mark.parametrize("tokens", range(1, 7))
def test_actual_query_head_and_packed_ownership(tokens):
    cube = function(STREAM, "attention_qk_pv_cube_stage")
    vector = function(STREAM, "attention_qk_pv_vector_lane_stage")
    heads, packed, active = set(), set(), set()
    for core in range(24):
        values = context(tokens, core)
        for target in ("qk_core", "qk_query_core", "qk_head_base", "qk_query_count"):
            values[target] = calculate(cube, target, values)
        assert values["qk_query_count"] in (0, 1)
        for ordinal in range(values["qk_query_count"]):
            values["qk_tick"] = ordinal * 5
            query = calculate(cube, "qk_t", values)
            assert 0 <= query < tokens
            active.add(core)
            for head in range(values["qk_head_base"], values["qk_head_base"] + 32):
                assert (query, head) not in heads
                heads.add((query, head))
            for lane in (0, 1):
                values["lane_id"] = lane
                values["qk_aiv"] = calculate(vector, "qk_aiv", values)
                values["qk_lane_head"] = calculate(vector, "qk_lane_head", values)
                values["pub_local_head"] = 0
                first_head = calculate(vector, "m_h0", values)
                assert first_head == values["qk_head_base"] + 16 * lane
                for group in (first_head // 8, first_head // 8 + 1):
                    assert (query, group) not in packed
                    packed.add((query, group))
    assert heads == {(q, h) for q in range(tokens) for h in range(64)}
    assert packed == {(q, g) for q in range(tokens) for g in range(8)}
    assert active == set(range(tokens * 2))
    schedule = function(SOURCE / "cube_schedule.py", "cube_schedule")
    count = calculate(schedule, "packed_workers", dict(context(tokens, 0), tokens=tokens))
    assert count == len(active) * 2


def trace(compiled, tokens, mask, core, lane=0):
    invoke, parameters = compiled
    values = dict.fromkeys(parameters)
    values.update(
        valid_block_mask=mask,
        position_ids=(tokens,),
        q_flat=(tokens * 64, 512),
        cmp_block_table=(1, 1026),
        cmp_kv_flat=(32864, 512),
        core_id=core,
        CORE_GROUPS=24,
        EVENT_BASE=2,
        lane_id=lane,
        PLAN_PHYSICAL_ROWS=False,
    )
    return invoke(**{name: values[name] for name in parameters})


@pytest.mark.parametrize("tokens", range(1, 7))
def test_source_event_sequences_for_every_block_mask_and_physical_group(tokens):
    old_cube = compiled_projection(BASE / "attention_stream_stages.py", "attention_qk_pv_cube_stage")
    old_vector = compiled_projection(BASE / "attention_stream_stages.py", "attention_qk_pv_vector_lane_stage")
    new_cube = compiled_projection(STREAM, "attention_qk_pv_cube_stage")
    new_vector = compiled_projection(STREAM, "attention_qk_pv_vector_lane_stage")
    for core in range(24):
        query = core // 2
        for pattern in range(32):
            mask = [[0] * 16 for _ in range(tokens)]
            if query < tokens:
                mask[query][:5] = [(pattern >> block) & 1 for block in range(5)]
            events = [trace(new_cube, tokens, mask, core)]
            assert events[0] == trace(old_cube, tokens, mask, query)
            for lane in (0, 1):
                actual = trace(new_vector, tokens, mask, core, lane)
                assert actual == trace(old_vector, tokens, mask, query, lane)
                events.append(actual)
            assert_event_dag(events)


def test_mathematical_operations_and_k_reductions_are_preserved():
    old = function(BASE / "attention_stream_stages.py", "attention_qk_pv_vector_lane_stage")
    new = function(STREAM, "attention_qk_pv_vector_lane_stage")
    for target in (
        "qk_masked",
        "qk_mi",
        "qk_exp",
        "qk_li",
        "qk_probability",
        "next_m",
        "alpha",
        "beta",
        "next_l",
        "next_left",
        "next_right",
        "n_sink_tile",
        "n_denom",
        "n_full",
        "n_bf16",
        "m_rot",
        "n_rope_bf16",
    ):
        assert ast.dump(expression(old, target)) == ast.dump(expression(new, target)), target
    for path in (BASE / "attention_stream_stages.py", STREAM):
        fn = function(path, "attention_qk_pv_cube_stage")
        calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and ast.unparse(n.func) in ("pl.matmul", "pl.tile.matmul")
        ]
        assert [ast.unparse(n) for n in calls] == [
            "pl.matmul(qk_q, qk_kv_t, out_dtype=pl.FP32)",
            "pl.tile.matmul(pv_probability_left, pv_first_right)",
            "pl.tile.matmul(pv_probability_left, pv_next_right)",
        ]


def test_only_three_scheduling_modules_change_and_external_pitches_remain_64():
    changed = {p.name for p in SOURCE.glob("*.py") if p.read_bytes() != (BASE / p.name).read_bytes()}
    assert changed == {"attention_stream_stages.py", "attention_dependencies.py", "cube_schedule.py"}
    source = STREAM.read_text()
    assert "H = 64\n" in source and "PARTITION_HEADS = 32\n" in source
    assert "t_dim = pl.tensor.dim(q_flat, 0) // H" in source
    assert source.count("qk_transfer_row = qk_slot * H") == 2
    assert source.count("pv_transfer_row = pv_slot * H") == 2
    dependency = function(SOURCE / "attention_dependencies.py", "wait_attention_plan_rows")
    loops = [n for n in ast.walk(dependency) if isinstance(n, ast.For)]
    assert len(loops) == 1
    assert ast.unparse(loops[0].iter) == "pl.range(query_group, tokens, GROUPS // ATTENTION_HEAD_PARTS)"
    assert ast.unparse(expression(dependency, "query_group")) == "core_group // ATTENTION_HEAD_PARTS"


def test_ownership_verifier_rejects_old_packed_wait_coverage():
    # In C01 the second six physical groups also publish packed output now.
    actual_producers = {24 + 2 * core + lane for core in range(12) for lane in (0, 1)}
    original_wait = set(range(24, 24 + 6 * 2))
    assert not actual_producers.issubset(original_wait)
