# SPDX-License-Identifier: Apache-2.0
"""Keep QK panel order but represent it as runtime loops instead of 16 copies."""

import ast
import hashlib
import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/score_group_order_v4"
    expected = json.loads((here / "score_group_order_v4.json").read_text())["files"]
    source = {p.name: p.read_text() for p in parent.glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == expected
    filename = "indexer_score_stages.py"
    text = source[filename]
    fn = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store"
    )
    loops = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id in ("buf_qk_panel", "live_half_buf_qk_panel")
    ]
    assert len(loops) == 2
    lines = text.splitlines(keepends=True)
    edits = []
    for node in loops:
        assert ast.unparse(node.iter.func) == "pl.unroll"
        assert ast.unparse(node.iter.args[0]) == "score_tile // qk_panel_cols"
        offset = sum(len(line) for line in lines[: node.iter.func.lineno - 1])
        edits.append((offset + node.iter.func.col_offset, offset + node.iter.func.end_col_offset))
    for start, stop in sorted(edits, reverse=True):
        text = text[:start] + "pl.range" + text[stop:]
    source[filename] = text
    target = here / "runs/compact_score_loops_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    report = dict(
        source=str(target),
        baseline="baseline.json",
        parent="score_group_order_v4",
        adopted=False,
        device_execution=False,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        changes=[filename],
        changed_panel_loops=[n.target.id for n in loops],
        mechanism="Reduce static panel copies while preserving ordered QK/WS operations and existing tile geometry",
        hypothesis="Lower code footprint may help; dynamic index work and conservative sync may instead regress",
        limitations=("No instruction-cache bottleneck claim without device evidence; "
                     "compiler must preserve carried tiles"),
    )
    with (here / "compact_score_loops_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
