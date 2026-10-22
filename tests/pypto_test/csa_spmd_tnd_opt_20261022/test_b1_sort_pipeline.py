# SPDX-License-Identifier: Apache-2.0
"""Execute actual sorting expressions and compare partition/order and fallback contracts.

The CPU model checks operation ordering, not device Sort32/merge instruction accuracy;
device checks with many equal scores remain necessary.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from prepare_b1_sort_pipeline import STREAM_INIT, STREAM_STEP, skip_unused_fills, stream_source
from test_mixed_candidates import plain_function


def fake_pl(output):
    def sort32(scores, indices):
        values = np.stack((scores.ravel(), indices.ravel()), axis=1).reshape(-1, 32, 2)
        return np.concatenate([x[np.argsort(-x[:, 0], kind="stable")] for x in values]).reshape(1, -1)

    def merge(*args, block_len=None, tmp=None):
        if block_len is not None:
            groups = args[0].reshape(-1, 2 * block_len, 2)
            return np.concatenate([x[np.argsort(-x[:, 0], kind="stable")] for x in groups]).reshape(1, -1)
        pairs = np.concatenate([a.reshape(-1, 2) for a in args])
        return pairs[np.argsort(-pairs[:, 0], kind="stable")].reshape(1, -1)

    tile = SimpleNamespace(
        create=lambda shape, **kw: np.full(shape, np.nan),
        extract=lambda a, row, col, shape, **kw: a[row : row + shape[0], col : col + shape[1]].copy(),
        reshape=np.reshape,
        arange=lambda start, shape, **kw: np.arange(start, start + np.prod(shape)).reshape(shape),
        fillpad=lambda a, **kw: a.copy(),
        slice=lambda a, shape, offset: a[offset[0] : offset[0] + shape[0], offset[1] : offset[1] + shape[1]],
        mrgsort=merge,
    )
    return SimpleNamespace(
        tile=tile,
        cast=lambda a, dtype: a,
        add=np.add,
        maximum=np.maximum,
        sort32=sort32,
        mrgsort=merge,
        reinterpret_view=lambda a, dtype: a,
        set_validshape=lambda a, rows, cols: a[:rows, :cols],
        store=lambda a, offset, target: output.append((offset, a.copy())),
        INT32=None,
        FP32=None,
        UINT32=None,
        PadValue=SimpleNamespace(min=None),
        MemorySpace=SimpleNamespace(Vec=None),
    )


@pytest.mark.parametrize("pattern", ("ties", "random_ties", "ascending", "descending", "negative", "infinity"))
@pytest.mark.parametrize("begin", (0, 28672))
def test_actual_full_leaf_expressions_match_reference_including_ties(pattern, begin):
    output = []
    namespace = {"pl": fake_pl(output), "TOPK_PAIR_WIDTH": 1024, "FP32_NEG_INF": -np.inf}
    reference = Path(__file__).resolve().parent / "runs/frozen_v122/indexer_topk_stages.py"
    original = plain_function(reference.read_text(), "indexer_topk_half_leaf_resident", namespace)
    rng = np.random.default_rng(1122)
    scores = {
        "ties": np.zeros(4096),
        "random_ties": rng.integers(-4, 5, 4096).astype(float),
        "ascending": np.arange(4096, dtype=float),
        "descending": -np.arange(4096, dtype=float),
        "negative": -rng.random(4096),
        "infinity": np.resize([np.inf, -np.inf, 0.0], 4096),
    }[pattern]
    original(scores.reshape(1, 4096), None, begin, 4096, 3)
    expected = output[-1][1]
    namespace.update(
        query_group_size=2,
        legacy_stream_full=1,
        single_begin=begin,
        legacy_half_scores=scores.reshape(4, 1024),
        BUFFERED_LONG_SCORE_TILE=1024,
    )
    block = compile(STREAM_STEP, "actual_stream_step", "exec")
    for step in range(4):
        namespace["single_step"] = step
        exec(block, namespace)
    np.testing.assert_array_equal(namespace["legacy_stream_root"], expected)


def test_full_predicate_excludes_partial_empty_and_other_groups():
    namespace = {
        "pl": fake_pl([]),
        "TOPK_PAIR_WIDTH": 1024,
        "bounds": None,
        "kv_seq_lens": None,
        "position_ids": None,
        "single_batch": 0,
        "single_query": 0,
        "single_begin": 4096,
    }
    block = compile(STREAM_INIT, "actual_stream_predicate", "exec")
    for group in (2, 6):
        for live in (0, 1, 2):
            for lane in (0, 1):
                for count in (0, 512, 1024, 2048, 2560, 3072, 4095, 4096, 4097):
                    namespace.update(
                        query_group_size=group,
                        buf_live=live,
                        buf_score_lane=lane,
                        single_steps=4,
                        single_capacity=4096,
                        indexer_query_visibility=lambda *args, count=count: (4096 + count, 1),
                    )
                    exec(block, namespace)
                    assert bool(namespace["legacy_stream_full"]) == (group == 2 and lane < live and count >= 4096)


def test_fifo_calls_and_non_target_functions_are_unchanged():
    source = (Path(__file__).resolve().parent / "runs/frozen_v122/indexer_score_stages.py").read_text()
    for transform in (stream_source, skip_unused_fills):
        revised = transform(source)

        def events(text):
            return [
                ast.dump(n, include_attributes=False)
                for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.Call) and ast.unparse(n.func).startswith("pl.system.sync_")
            ]

        assert events(revised) == events(source)
        original_functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        changed = []
        for fn in (n for n in ast.parse(revised).body if isinstance(n, ast.FunctionDef)):
            if ast.dump(fn, include_attributes=False) != ast.dump(
                original_functions[fn.name], include_attributes=False
            ):
                changed.append(fn.name)
        assert changed == ["indexer_score_vector_lane_stage"]
