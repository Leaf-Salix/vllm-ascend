# SPDX-License-Identifier: Apache-2.0
"""Execute real proof helpers against contiguous, fragmented, invalid and tail pages."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from prepare_page_proof_single_row import transform
from test_mixed_candidates import plain_function
from test_query_resident_weights import functions

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "runs/frozen_v122"


def numpy_pl():
    def read(t, index):
        return t[tuple(index)]

    def write(t, index, value):
        t[tuple(index)] = value
        return t

    def load(t, start, shape, valid_shape=None, **_):
        shape = valid_shape or shape
        return t[tuple(slice(int(i), int(i + n)) for i, n in zip(start, shape))].copy()

    tile = SimpleNamespace(
        full=lambda shape, dtype, value: np.full(shape, value, dtype),
        ci=lambda start, shape, dtype: np.arange(start, start + np.prod(shape), dtype=dtype).reshape(shape),
        read=read,
    )
    return SimpleNamespace(
        FP32=np.float32,
        INT32=np.int32,
        INDEX=np.int64,
        tile=tile,
        tensor=SimpleNamespace(dim=lambda t, axis: t.shape[axis], read=read),
        cast=lambda x, target_type, **_: np.asarray(x).astype(target_type),
        min=min,
        max=max,
        range=range,
        load=load,
        read=read,
        write=write,
        add=np.add,
        sub=np.subtract,
        mul=np.multiply,
        neg=np.negative,
        abs=np.abs,
        minimum=np.minimum,
        maximum=np.maximum,
        col_expand_mul=np.multiply,
        create_tile=lambda shape, dtype: np.empty(shape, dtype=dtype),
        set_validshape=lambda t, rows, cols: t[: int(rows), : int(cols)],
        row_min=lambda t, _: np.min(t, axis=1, keepdims=True),
        row_max=lambda t, _: np.max(t, axis=1, keepdims=True),
    )


@pytest.mark.parametrize(
    "file,name,rows",
    [
        ("attention_plan_stages.py", "compressed_pages_all_valid", "ROWS"),
        ("indexer_scale_layout_stages.py", "indexer_scale_layout_plan_stage", "LAYOUT_PROOF_ROWS"),
    ],
)
def test_only_valid_row_count_changes(file, name, rows):
    source = (REFERENCE / file).read_text()
    old, new = functions(source), functions(transform(source, name, rows))
    count = 0
    for node in ast.walk(new[name]):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in ("ones", "repeated"):
            assert ast.literal_eval(node.value.args[1]) == 1
            node.value.args[1] = ast.Name(id=rows, ctx=ast.Load())
            count += 1
    assert count == 2
    assert {k: ast.dump(v, include_attributes=False) for k, v in new.items()} == {
        k: ast.dump(v, include_attributes=False) for k, v in old.items()
    }


def test_attention_proof_matches_independent_range_oracle():
    source = (REFERENCE / "attention_plan_stages.py").read_text()
    ns = dict(pl=numpy_pl(), BLOCK_SIZE=32, EXACT_FP32_INT=16777216, ROWS=8, PAGE_CHECK_TILE=1024)
    old = plain_function(source, "compressed_pages_all_valid", ns.copy())
    new = plain_function(
        transform(source, "compressed_pages_all_valid", "ROWS"), "compressed_pages_all_valid", ns.copy()
    )
    for needed in (0, 1, 7, 32, 33, 1023, 1024, 1025, 2049):
        for cols in (max(0, needed - 1), needed, needed + 3):
            table = np.arange(cols, dtype=np.int32)[None, :]
            length = needed * 128
            for physical in (0, 4096, 16777216, 16777217):
                variants = [table, table[:, ::-1].copy()]
                for invalid in (-1, physical, 2147483647):
                    bad = table.copy()
                    if cols:
                        bad[0, cols // 2] = invalid
                    variants.append(bad)
                for data in variants:
                    expected = int(
                        0 < physical <= 16777216
                        and needed <= cols
                        and np.all((data[0, :needed] >= 0) & (data[0, :needed] < physical))
                    )
                    assert int(old(data, 0, length, physical)) == expected
                    assert int(new(data, 0, length, physical)) == expected
    assert int(new(np.zeros((1, 200000), dtype=np.int32), 0, 16777217, 4096)) == 0


def test_layout_proof_preserves_direction_fallback_and_workspace_guards():
    source = (REFERENCE / "indexer_scale_layout_stages.py").read_text()
    pl = numpy_pl()
    ns = dict(
        pl=pl,
        SCALE_JOB_WORDS=4,
        LAYOUT_WORDS_PER_REQUEST=2,
        LAYOUT_PROOF_PAGES=1024,
        LAYOUT_PROOF_ROWS=8,
        LAYOUT_EXACT_INTEGER_LIMIT=16777216,
        LAYOUT_COMPRESS_RATIO=4,
        LAYOUT_ROWS_PER_PAGE=32,
        SCALE_PLAN_ALIGNMENT=16,
        GROUP_WORDS_PER_TOKEN=4,
    )
    plain_function((REFERENCE / "request_geometry.py").read_text(), "request_bounds", ns)
    plain_function((REFERENCE / "indexer_scale_plan_stages.py").read_text(), "indexer_scale_plan_base", ns)
    base = plain_function(source, "indexer_scale_layout_base", ns)
    old = plain_function(source, "indexer_scale_layout_plan_stage", ns.copy())
    new = plain_function(
        transform(source, "indexer_scale_layout_plan_stage", "LAYOUT_PROOF_ROWS"),
        "indexer_scale_layout_plan_stage",
        ns.copy(),
    )
    bounds = np.array([0, 6, 7, 7, 13], dtype=np.int32)
    for pages in (1, 2, 31, 32, 33, 1023, 1024, 1025, 2049):
        lengths = np.array([pages * 128] * 4, dtype=np.int32)
        for physical in (0, 4096, 16777217):
            for pattern in ("ascending", "descending", "fragmented", "negative", "out_of_range"):
                table = np.tile(np.arange(pages + 3, dtype=np.int32), (4, 1))
                if pattern == "descending":
                    table[:, :pages] = table[:, :pages][:, ::-1]
                if pattern == "fragmented" and pages > 2:
                    table[:, pages // 2] = 0
                if pattern == "negative":
                    table[:, 0] = -1
                if pattern == "out_of_range":
                    table[:, pages - 1] = physical
                a = np.full(int(base(4, 13, 48)) + 8 + 19, -777, dtype=np.int32)
                b = a.copy()
                old(bounds, lengths, table, a, 13, physical, 48)
                new(bounds, lengths, table, b, 13, physical, 48)
                np.testing.assert_array_equal(a, b)
                offset = int(base(4, 13, 48))
                assert np.all(b[:offset] == -777) and np.all(b[offset + 8 :] == -777)
                assert np.all(b[offset + 4 : offset + 6] == 0)  # Empty request overwritten, not stale.
