# SPDX-License-Identifier: Apache-2.0
"""Check math ordering, page/FIFO contracts and complete pooled write coverage."""

import ast
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

from prepare_score_query_pipeline import function
from prepare_score_ws_pool import transform

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "runs/frozen_v122"


def helper(text):
    return ast.parse(function(text, "_indexer_score_cube_split_store")).body[0]


def test_original_math_pages_and_fifo_are_retained():
    source = (REFERENCE / "indexer_score_stages.py").read_text()
    candidate = transform(source)
    for name in (
        "_indexer_score_cube_original",
        "indexer_score_cube_stage",
        "indexer_score_vector_lane_stage",
        "indexer_score_local_half_leaf",
    ):
        assert function(source, name) == function(candidate, name)

    class NormalizeSlot(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in ("split_ws_slot", "split_ws_last_slot"):
                node.id = "split_ws_acc"
            return node

    def calls(text):
        fn = NormalizeSlot().visit(helper(text))
        # ast.walk is breadth-first; sort by source position to compare execution
        # order despite the additional pool store conditional.
        names = (
            "pl.tile.matmul_acc",
            "pl.gather_row",
            "pl.load",
            "indexer_work_item",
            "indexer_decode_group",
            "indexer_leaf_range",
            "pl.system.sync_set",
            "pl.system.sync_wait",
        )
        return [
            ast.dump(n, include_attributes=False)
            for n in sorted(ast.walk(fn), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
            if isinstance(n, ast.Call) and ast.unparse(n.func) in names
        ]

    assert calls(source) == calls(candidate)

    # Every QK FIX keeps its original FP16 conversion, scale and ReLU arguments.
    def fixes(text):
        return [
            ast.dump(n, include_attributes=False)
            for n in ast.walk(helper(text))
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.tile.assemble" and n.keywords
        ]

    assert fixes(source) == fixes(candidate)


def test_real_source_offsets_initialize_all_windows_before_each_store():
    fn = helper(transform((REFERENCE / "indexer_score_stages.py").read_text()))
    assignments = {ast.unparse(n.targets[0]): n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)}
    guard = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "buf_qk_panel % SPLIT_WS_POOL_PANELS == 0"
    )
    store = next(n for n in ast.walk(guard) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.store")

    def value(node, context):
        return eval(compile(ast.Expression(node), "<pool-source>", "eval"), {}, context)

    ns = dict(pl=SimpleNamespace(max=max), qk_panel_cols=64, SPLIT_WS_POOL_PANELS=8)
    pool = [None] * 512
    output = [None] * 1024
    writes = []
    for panel in range(1, 16):
        ns.update(buf_qk_panel=panel, buf_panel_col=panel * 64)
        offset = value(assignments["split_ws_col"], ns)
        assert offset % 64 == 0 and 0 <= offset <= 448
        pool[offset : offset + 64] = range((panel - 1) * 64, panel * 64)
        if value(guard.test, ns):
            column = value(store.args[1].elts[1], ns)
            assert None not in pool
            output[column : column + 512] = pool
            writes.append(column)
            pool = [None] * 512
    offset = value(assignments["split_ws_last_col"], ns)
    pool[offset : offset + 64] = range(960, 1024)
    assert None not in pool
    output[512:1024] = pool
    writes.append(512)
    assert writes == [0, 512]
    assert output == list(range(1024))


def test_unchanged_policy_enables_the_actual_cpu_descriptor():
    actual = json.loads(
        (HERE.parent / "csa_spmd_tnd_20261020/compile_descriptors/shared_combined_v122_h131072_b16.json").read_text()
    )
    select = runpy.run_path(str(HERE / "runs/score_ws_pool/policy.py"))["select_policy"]
    assert actual["policy"] == select(tokens=96, requests=16, max_query_tokens=6, max_seq_len=131078).constexprs()
    assert actual["policy"]["SPLIT_SCORE_ROWS"]
    assert actual["policy"]["QK_PANEL"] == 64
    assert actual["policy"]["TO_L0"]
