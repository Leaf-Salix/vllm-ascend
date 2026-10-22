# SPDX-License-Identifier: Apache-2.0
"""Ensure the diagnostic has real work and removes only the consumer protocol."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generate_score_sim_case import write_inputs
from prepare_score_producer_diagnostic import BUFFERS, isolate
from test_mixed_candidates import plain_function

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize("parent", ("frozen_v122", "score_query_pipeline"))
def test_only_fifo_and_empty_drain_are_removed(parent):
    source = (HERE / "runs" / parent / "indexer_score_stages.py").read_text()

    def calls(text, filter_fifo=False):
        fn = next(
            n
            for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store"
        )
        selected = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            value = ast.unparse(node)
            if filter_fifo and (
                ast.unparse(node.func).startswith("pl.system.sync_")
                or value
                in (
                    "pl.range(pl.min(buf_score_iters, 2))",
                    "pl.min(buf_score_iters, 2)",
                )
            ):
                continue
            selected.append(ast.dump(node, include_attributes=False))
        return selected

    assert calls(source, True) == calls(isolate(source))


def test_actual_group_decoder_produces_one_active_six_step_leaf():
    source = (HERE / "runs/frozen_v122/indexer_group_stages.py").read_text()
    ns = dict(
        pl=SimpleNamespace(
            cast=lambda x, _: x,
            INDEX=None,
            min=min,
            max=max,
            read=lambda a, i: a[i[0]],
            tensor=SimpleNamespace(dim=lambda a, d: len(a)),
        ),
        SCORE_WORKERS=24,
        SCORE_TILE=1024,
        CANDIDATES_PER_LEAF=8192,
        GROUP_DESCRIPTOR_ALIGNMENT=16,
        GROUP_DESCRIPTOR_WORDS=4,
    )
    decode = plain_function(source, "indexer_decode_group", ns)
    work = plain_function(source, "indexer_work_item", ns)
    leaf = plain_function(source, "indexer_leaf_range", ns)
    offsets = [0] * 32
    offsets[1] = 1
    offsets[16:20] = [0, 0, 6, 6144]
    assert work(0, 1, 1) == (0, 0)
    assert work(1, 1, 1) == (-1, -1)  # Default block_idx=1 would be an empty trace.
    assert decode([0, 6], [24576], list(range(24570, 24576)), offsets, 0, 6) == (0, 0, 6, 6144)
    assert leaf(0, 6, 0, 6144, True, 6) == (0, 6144, 6144)
    for logical_page in range(192):
        physical = 192 - logical_page
        byte = physical * 4160
        shift = byte % 128
        assert shift in (0, 64)
        first_row = byte // 128
        assert first_row * 128 + shift == byte
        assert first_row + 32 <= 6272


@pytest.mark.parametrize("label", ("v122", "n256"))
def test_exact_sizes_scalar_bindings_and_inputs(label):
    case = HERE / "runs" / f"score_sim_{label}" / "drycase/ptoas/dry_score"
    binding = json.loads((case / "input_bindings.json").read_text())
    assert [row["value"] for row in binding["scalars"]] == [
        384,
        16,
        384,
        6272,
        6272,
        193,
        4160,
        1,
        192,
        6,
        2,
        32,
        12,
        1024,
        0,
        1,
    ]
    write_inputs(case)
    values = {}
    for row in binding["buffers"]:
        name = row["logical_name"]
        dtype = getattr(np, BUFFERS[name]["dtype"].removeprefix("torch."))
        value = np.fromfile(case / f"{row['cpp_name']}.bin", dtype=dtype)
        assert value.size == row["elements"] == np.prod(BUFFERS[name]["shape"])
        values[name] = value.reshape(BUFFERS[name]["shape"])
    assert values["plan"].tolist() == [1, 6144, 1, 6, 0, 0, 0, 0]
    assert values["offsets"][16:20].tolist() == [0, 0, 6, 6144]
    assert values["coefficients"].sum() == 384
    assert values["table"][0].tolist() == list(range(192, 0, -1))
    assert np.all(values["transfer"] == -777)
    # Integer dot 128, original FIX /1024 => exactly FP16 .125; 64 heads sum to 8.
    assert np.float16(128 / 1024) * 64 == 8
    cpp = (case / "main.cpp").read_text()
    assert "SCORE_DIAGNOSTIC_MISMATCH" in cpp and "score_output_check.txt" in cpp
