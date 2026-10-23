# SPDX-License-Identifier: Apache-2.0
"""Verify the actual rank expression preserves the original complete KV work cover."""

import ast
import hashlib
import json
from pathlib import Path

import pytest


def mapping(enabled, branch=4):
    root = Path(__file__).with_name("runs") / "rotate_kv_v1"
    if not root.is_dir():
        pytest.skip("Materialize the frozen KV rotation candidate first")
    tree = ast.parse((root / "projection_overlap_stages.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "projection_overlap_stage")
    initial = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "kv_worker")
    conditional = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "KV_TAIL_OWNERS")
    program = compile(ast.Module(body=[initial, conditional], type_ignores=[]), "actual_kv_mapping", "exec")
    ranks = []
    for worker in range(24):
        values = dict(worker=worker, GROUPS=24, BRANCH_GROUPS=branch, KV_TAIL_OWNERS=enabled)
        exec(program, values)
        ranks.append(values["kv_worker"])
    assert sorted(ranks) == list(range(24))
    return ranks


def work(tokens, ranks):
    result = []
    for physical, logical_rank in enumerate(ranks):
        for base in range(0, tokens, 512):
            rows = min(tokens - base, 512)
            dense = 32 if rows < 128 else 64
            groups = min(3, max(1, rows // dense))
            complete = rows // dense * dense
            padded = (rows + 15) // 16 * 16
            for logical in range(logical_rank, 4 * groups, 24):
                column, group = logical // groups * 128, logical % groups
                for first in range(group * dense, complete, groups * dense):
                    result.append((base + first, column, dense, dense, physical))
                for first in range(complete + group * 16, padded, groups * 16):
                    result.append((base + first, column, 16, min(16, rows - first), physical))
    return result


@pytest.mark.parametrize(
    "tokens", [1, 15, 16, 31, 32, 63, 64, 95, 96, 127, 128, 129, 144, 168, 192, 256, 511, 512, 513, 1025]
)
def test_same_tiles_and_valid_rows(tokens):
    before, after = work(tokens, mapping(False)), work(tokens, mapping(True))
    assert sorted(x[:-1] for x in before) == sorted(x[:-1] for x in after)
    assert len(after) == len({x[:2] for x in after})
    if tokens == 144:
        assert {x[-1] for x in before} == set(range(4, 12))
        assert {x[-1] for x in after} == set(range(16, 24))


def test_original_mapping_and_full_consumer_wait_unchanged():
    for branch in range(1, 24):
        assert mapping(False, branch) == [(w + 24 - branch) % 24 for w in range(24)]
    here = Path(__file__).resolve().parent
    expected = json.loads((here / "baseline.json").read_text())["files"]
    root = here / "runs/rotate_kv_v1"
    changed = {n for n, digest in expected.items() if hashlib.sha256((root / n).read_bytes()).hexdigest() != digest}
    assert changed == {"projection_overlap_stages.py", "cube_schedule.py"}
    source = (root / "vector_prefix_stages.py").read_text()
    assert "wait_row_owners(signal, PHASE_KV_PROJECT, CORES, generation, 0, GROUPS, 1, GROUPS, 0, 1)" in source
