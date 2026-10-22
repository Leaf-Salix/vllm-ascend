# SPDX-License-Identifier: Apache-2.0
"""N128 with alternating M128/M64 QK accumulators and unchanged K192 WS halves."""

import ast
import copy
import hashlib
import json
from pathlib import Path

from prepare_score_n128 import assignment


def split_assignment(source, name, *, left):
    original = assignment(source, name)
    replacement = []
    for suffix, rows in (("large", 128), ("small", 64)):
        node = copy.deepcopy(original)
        node.targets[0].id = name.removesuffix("_acc0") + "_acc_" + suffix if not left else name + "_" + suffix
        shape = node.value.args[3] if left else node.value.args[0]
        shape.elts[0] = ast.Constant(rows)
        if left and suffix == "small":
            node.value.args[1] = ast.BinOp(node.value.args[1], ast.Add(), ast.Constant(128))
        replacement.append(" " * original.col_offset + ast.unparse(ast.fix_missing_locations(node)) + "\n")
    lines = source.splitlines(keepends=True)
    lines[original.lineno - 1 : original.end_lineno] = replacement
    return "".join(lines)


def transform(source):
    for name in ("split_query0", "split_query1", "live_half_split_query0"):
        source = split_assignment(source, name, left=True)
    for name in ("split_qk_acc0", "live_half_split_qk_acc0"):
        source = split_assignment(source, name, left=False)
    for prefix, half in (("split_", 0), ("split_", 1), ("live_half_split_", 0)):
        qk, current, query = (prefix + stem + str(half) for stem in ("qk", "current", "query"))
        candidates = [
            n
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == qk
        ]
        assert len(candidates) == 2
        lines = source.splitlines(keepends=True)
        for node in sorted(candidates, key=lambda n: n.lineno, reverse=True):
            # The existing next statement is the FP16/ReLU insertion into the
            # corresponding 192-row half; retain its attributes exactly.
            insertion = ast.parse(lines[node.end_lineno].strip()).body[0]
            assert insertion.targets[0].id == current
            replacement = []
            for suffix, offset in (("large", 0), ("small", 128)):
                mm = copy.deepcopy(node)
                mm.targets[0].id = qk + "_" + suffix
                mm.value.args[0] = ast.Name(prefix + "qk_acc_" + suffix, ast.Load())
                mm.value.args[1] = ast.Name(query + "_" + suffix, ast.Load())
                fix = copy.deepcopy(insertion)
                fix.value.args[1] = ast.Name(qk + "_" + suffix, ast.Load())
                fix.value.args[2] = ast.List([ast.Constant(offset), ast.Constant(0)], ast.Load())
                replacement.extend(
                    " " * node.col_offset + ast.unparse(ast.fix_missing_locations(n)) + "\n" for n in (mm, fix)
                )
            lines[node.lineno - 1 : node.end_lineno + 1] = replacement
        source = "".join(lines)
    assert "split_qk_acc0" not in source
    ast.parse(source)
    return source


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/score_n128_single_acc_v3"
    expected = json.loads((here / "score_n128_single_acc_v3.json").read_text())["files"]
    source = {p.name: p.read_text() for p in parent.glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == expected
    source["indexer_score_stages.py"] = transform(source["indexer_score_stages.py"])
    target = here / "runs/score_n128_two_acc_v1"
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        ast.parse(text)
        (target / name).write_text(text)
    report = dict(
        source=str(target),
        baseline="baseline.json",
        adopted=False,
        device_execution=False,
        parent="score_n128_single_acc_v3",
        changes_from_parent=["indexer_score_stages.py"],
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        mechanism="Two Acc tiles of M128/M64 alternate across each 192-row QK half; WS still consumes full K192 halves",
        preserved=[
            "all QK K128 sums",
            "per-element FP16/ReLU conversion",
            "K192 + K192 WS order",
            "FIFO and physical leaves",
        ],
        risk="More QK instructions than single-Acc N128; FIX overlap and resource fit do not prove a full-chain gain",
    )
    with (here / "score_n128_two_acc_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
