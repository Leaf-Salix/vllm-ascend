# SPDX-License-Identifier: Apache-2.0
"""Reuse the internal, row-invariant RoPE permutation in each Query worker.

The internal producer writes the same adjacent-column permutation to every row.
The existing PHASE_ROPE acquisition precedes Query consumption. Only the index
is hoisted; position-dependent cos/sin and writable gather scratch stay per work.
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
        names = ("q_swap_one", "q_swap_repeated", "q_abs_swap")
        moved = [
            next(n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name)
            for name in names
        ]
        assert ast.unparse(moved[0].value) == (
            "pl.load(rope_swap_idx, [qt, 0], [1, ROPE_DIM], target_memory=pl.MemorySpace.Vec)"
        )
        moved[0].value.args[1].elts[0] = ast.Constant(value=0)
        for node in moved:
            loop.body.remove(node)
        before = ast.parse("""
q_abs_swap = pl.tile.create([1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32)
if tokens > 0:
    pass
""").body
        before[1].body = moved
        fn.body[fn.body.index(loop) : fn.body.index(loop)] = before

    return edit_functions(source, ["_q_dequant_rope_contiguous_heads"], edit)


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/query_resident_weights"
    expected = json.loads((here / "query_resident_weights.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob("*.py")} == expected
    target = here / "runs/query_resident_swap"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    name = "query_projection_stages.py"
    (target / name).write_text(transform((parent / name).read_text()))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")}
    assert [n for n in files if files[n] != expected[n]] == [name]
    report = dict(
        source=str(target),
        parent="query_resident_weights",
        reference_kind="frozen_tnd_v122",
        files=files,
        modified_files=[name],
        adopted=False,
        device_execution=False,
        invariant="Internal producer writes (1,0,3,2,...) to every valid row; no external RoPE value is assumed equal",
        acquisition="Existing PHASE_ROPE wait_row_owners covers row 0 before q_dequant_rope_stage",
        zero_tokens="Guard the new row-0 load; empty invocation does not read that row",
        retained="Per-query cos/sin; per-iteration writable gather scratch; original token-tail helper",
        limitations="Requires generated physical lifetime audit and complete paired L1 measurement",
    )
    with (here / "query_resident_swap.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
