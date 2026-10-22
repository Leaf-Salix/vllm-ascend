# SPDX-License-Identifier: Apache-2.0
"""Peel fixed first/last QK panels to avoid conditional Mat carries in a compact loop."""

import ast
import copy
import hashlib
import json
from pathlib import Path


def specialized_body(loop, stage):
    class KnownPanelConditions(ast.NodeTransformer):
        def visit_If(self, node):
            condition = ast.unparse(node.test)
            known = {
                "buf_qk_panel > 0": stage != "first",
                "buf_qk_panel + key_prefetch_panels < score_tile // qk_panel_cols": stage != "last",
                "buf_qk_panel + 1 < score_tile // qk_panel_cols": stage != "last",
            }
            if condition in known:
                chosen = node.body if known[condition] else node.orelse
                module = ast.Module(body=chosen, type_ignores=[])
                return self.generic_visit(module).body
            updated = self.generic_visit(node)
            if not updated.body and not updated.orelse:
                return []
            if not updated.body:
                updated.body = [ast.Pass()]
            return updated

    module = ast.Module(body=copy.deepcopy(loop.body), type_ignores=[])
    return KnownPanelConditions().visit(module).body


def transform(source):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store")
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and ast.unparse(n.target) == "buf_qk_panel")
    assert ast.unparse(loop.iter) == "pl.unroll(score_tile // qk_panel_cols)"
    middle = copy.deepcopy(loop)
    middle.iter = ast.parse("pl.range(1, score_tile // qk_panel_cols - 1)", mode="eval").body
    middle.body = specialized_body(loop, "middle")
    statements = [*ast.parse("buf_qk_panel = 0").body, *specialized_body(loop, "first"), middle]
    statements += [*ast.parse("buf_qk_panel = score_tile // qk_panel_cols - 1").body, *specialized_body(loop, "last")]
    guarded = ast.If(
        test=ast.parse("key_prefetch_panels == 1 and key_prefetch_to_l0", mode="eval").body,
        body=statements,
        orelse=[copy.deepcopy(loop)],
    )
    lines = source.splitlines(keepends=True)
    lines[loop.lineno - 1 : loop.end_lineno] = [
        " " * loop.col_offset + s + "\n" for s in ast.unparse(guarded).splitlines()
    ]
    result = "".join(lines)
    ast.parse(result)
    return result


def main():
    here = Path(__file__).resolve().parent
    baseline = json.loads((here / "baseline.json").read_text())
    source = {p.name: p.read_text() for p in (here / "runs/baseline").glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == baseline["files"]
    source["indexer_score_stages.py"] = transform(source["indexer_score_stages.py"])
    target = here / "runs/compact_score_peeled_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        (target / name).write_text(content)
    original = json.loads(Path(baseline["cases"]["C04"]["source_report"]).read_text())
    descriptor = dict(buffers=original["descriptors"]["spmd"], policy=original["policies"]["spmd"])
    (here / "descriptors/C04_compact_score_peeled_v1.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    report = dict(
        source=str(target),
        baseline="baseline.json",
        adopted=False,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        changes=["indexer_score_stages.py"],
        mechanism="Full six-query branch: peel first/last, compact middle fourteen panels; retain panel math order",
        unchanged="Live-half branch, original fallback, tile dimensions, ABI and FIFO protocol",
        limitations="CPU feasibility experiment after conditional Mat self-move; no device benefit established",
    )
    with (here / "compact_score_peeled_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
