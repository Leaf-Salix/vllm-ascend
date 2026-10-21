# SPDX-License-Identifier: Apache-2.0
"""Scope, logical reduction coverage, and unchanged FIFO/page contracts."""

import ast
import json
import runpy
from pathlib import Path

import pytest
from prepare_score_query_pipeline import function, panel_math, transform_policy, transform_score

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "runs/frozen_v122"


def test_policy_changes_only_score_panel_and_key_prefetch_in_existing_split_domain(tmp_path):
    original = (REFERENCE / "policy.py").read_text()
    changed = tmp_path / "policy.py"
    changed.write_text(transform_policy(original))
    select_old = runpy.run_path(str(REFERENCE / "policy.py"))["select_policy"]
    select_new = runpy.run_path(str(changed))["select_policy"]
    for tokens in (1, 6, 24, 31, 32, 48, 56, 96, 144, 168, 192, 385):
        for requests in (1, 4, 16, 28, 32):
            for query in (1, 5, 6):
                for history in (8192, 131072):
                    args = dict(tokens=tokens, requests=requests, max_query_tokens=query, max_seq_len=history + query)
                    if tokens > requests * query:
                        for select in (select_old, select_new):
                            with pytest.raises(ValueError, match="token容量超过"):
                                select(**args)
                        continue
                    old = select_old(**args).constexprs()
                    new = select_new(**args).constexprs()
                    expected = dict(old)
                    if old["SPLIT_SCORE_ROWS"]:
                        expected.update(QK_PANEL=256, TO_L0=False)
                    assert new == expected


def test_ws_keeps_global_k_order_and_complete_single_query_head_ranges():
    block = ast.parse("def panel():\n" + panel_math()).body[0]
    ws = [n for n in block.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]).startswith("query_ws_")]
    assert len(ws) == 6
    consumed = []
    for q, statement in enumerate(ws):
        call = statement.value
        assert ast.unparse(call.func) == "pl.tile.matmul_acc"
        assert ast.unparse(call.args[0]) == ("query_ws_acc" if q == 0 else f"query_ws_{q - 1}")
        assert ast.unparse(call.args[1]) == f"coefficient_left_{q}"
        assert ast.unparse(call.args[2]) == f"query_right_{q}"
        assert bool(call.keywords) == (q == 0)
        if q == 0:
            assert call.keywords[0].arg == "init_cond" and ast.literal_eval(call.keywords[0].value) is True
        consumed.extend(range(q * 64, (q + 1) * 64))
    assert consumed == list(range(384))
    # The four N256 panel stores still cover exactly one unchanged 1024-column slot.
    assert [n for panel in range(4) for n in range(panel * 256, (panel + 1) * 256)] == list(range(1024))


def test_native_pages_fifo_and_vector_math_are_unchanged():
    source = (REFERENCE / "indexer_score_stages.py").read_text()
    candidate = transform_score(source)
    for name in ("_indexer_score_cube_original", "indexer_score_vector_lane_stage", "indexer_score_local_half_leaf"):
        assert function(source, name) == function(candidate, name)

    def contracts(text):
        fn = ast.parse(function(text, "_indexer_score_cube_split_store"))
        return [
            ast.dump(n, include_attributes=False)
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and ast.unparse(n.func)
            in (
                "indexer_work_item",
                "indexer_decode_group",
                "indexer_leaf_range",
                "pl.gather_row",
                "pl.system.sync_set",
                "pl.system.sync_wait",
            )
        ]

    assert contracts(source) == contracts(candidate)


def test_active_cpu_descriptor_matches_candidate_policy_and_keeps_all_buffers():
    base = json.loads(
        (HERE.parent / "csa_spmd_tnd_20261020/compile_descriptors/shared_combined_v122_h131072_b16.json").read_text()
    )
    actual = json.loads((HERE / "descriptors/score_query_pipeline_b16.json").read_text())
    select = runpy.run_path(str(HERE / "runs/score_query_pipeline/policy.py"))["select_policy"]
    assert actual["buffers"] == base["buffers"]
    assert actual["policy"] == select(tokens=96, requests=16, max_query_tokens=6, max_seq_len=131078).constexprs()
    assert (actual["policy"]["QK_PANEL"], actual["policy"]["TO_L0"]) == (256, False)
