# SPDX-License-Identifier: Apache-2.0
"""Reuse the same eight-head weight scale in UB across a worker's Query loop.

When WORKERS is divisible by heads-per-query-work, the head owner is invariant:
(worker + iteration*WORKERS) % (H/8) == worker % (H/8). Nondivisible worker counts
retain the exact original per-work load. The physical eight-row math is unchanged.
"""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_query_head_guard import edit_functions


def transform(source):
    def edit(fn):
        loop = next(n for n in fn.body if isinstance(n, ast.For))
        index = fn.body.index(loop)
        before = ast.parse("""
resident_head_scale = pl.tile.create([Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
if WORKERS % (H // Q_ROPE_T_TILE) == 0:
    resident_head_begin = worker_id % (H // Q_ROPE_T_TILE) * Q_ROPE_T_TILE
    resident_head_scale = pl.load(
        q_head_weights, [resident_head_begin, 0], [Q_ROPE_T_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec
    )
""").body
        fn.body[index:index] = before
        scale = next(n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "q_head_scale")
        assert ast.unparse(scale.value.func) == "pl.load" and ast.unparse(scale.value.args[0]) == "q_head_weights"
        replacement = ast.parse("""
if WORKERS % (H // Q_ROPE_T_TILE) == 0:
    q_head_scale = resident_head_scale
""").body[0]
        replacement.orelse = [scale]
        loop.body[loop.body.index(scale)] = replacement

    return edit_functions(source, ["_q_dequant_rope_contiguous_heads"], edit)


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/query_contiguous_heads_v4"
    expected = json.loads((here / "query_contiguous_heads_v4.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob("*.py")} == expected
    target = here / "runs/query_resident_weights"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    name = "query_projection_stages.py"
    (target / name).write_text(transform((parent / name).read_text()))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")}
    assert [n for n in files if files[n] != expected[n]] == [name]
    report = dict(
        source=str(target),
        parent="query_contiguous_heads_v4",
        reference_kind="frozen_tnd_v122",
        files=files,
        modified_files=[name],
        adopted=False,
        device_execution=False,
        enabled_domain="original v4 full token tiles; readonly hoist when WORKERS divisible by H/8",
        retained="Per-work load for nondivisible worker counts; whole original helper for token tails",
        resources="One 16KiB readonly Vec tile; actual allocator and last-consumer proof pending",
        logical_weight_read_bytes_b16=dict(original_v122=1572864, v4=12582912, resident=786432),
        limitations="Logical reads are not HBM traffic or measured latency; no performance claim",
    )
    with (here / "query_resident_weights.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
