# SPDX-License-Identifier: Apache-2.0
"""Publish Indexer inputs before main-compressor AIV work, retaining all lifetimes."""

import ast
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def transform(source):
    nodes = list(ast.walk(ast.parse(source)))
    first = next(
        n
        for n in nodes
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "COOPERATIVE_PREFIX"
        and any(isinstance(v, ast.Call) and ast.unparse(v.func) == "main_compressor_vector_stage" for v in ast.walk(n))
    )
    assert len(first.body) == len(first.orelse) == 1
    assert ast.unparse(first.orelse[0].value.func) == "indexer_compressor_vector_stage"
    publish = next(
        n
        for n in nodes
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "publish_phase"
        and ast.unparse(n.value.args[1]) == "PHASE_IDX_SCALE"
    )
    lines = source.splitlines(keepends=True)
    moved = "".join(lines[first.lineno - 1 : first.body[-1].end_lineno])
    kept = "    if not COOPERATIVE_PREFIX:\n" + "".join(lines[first.orelse[0].lineno - 1 : first.end_lineno])
    lines[publish.end_lineno : publish.end_lineno] = [
        "    # Indexer inputs are now published; AIC may start Score during independent main AIV work.\n" + moved
    ]
    lines[first.lineno - 1 : first.end_lineno] = [kept]
    result = "".join(lines)

    def calls(text):
        return Counter(
            ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call)
        )

    assert calls(result) == calls(source)
    assert moved in result
    # The moved helper's original pool -> all readers done -> state commit -> cache
    # protocol is copied verbatim. No consumer in the intervening Indexer chain reads
    # main cache/state; Attention still waits PHASE_MAIN_OUTPUT below Score/Top-K.
    return result


def main():
    here = Path(__file__).resolve().parent
    reference = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*.py")} == expected
    revised = transform((reference / "vector_schedule.py").read_text())
    target = here / "runs/main_vector_late"
    shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "vector_schedule.py").write_text(revised)
    report = {
        "source": str(target),
        "reference_kind": "frozen_tnd_v122",
        "adopted": False,
        "device_execution": False,
        "change": "cooperative main compressor vector chain moves after IDX_SCALE publication, before main Query",
        "preserved": [
            "all helper bodies and arguments",
            "all 72 generations and phase notifications",
            "pool-read-before-state-commit",
            "main cache visibility before Attention",
            "Score FIFO and drain",
        ],
        "hypothesis": (
            "earlier Indexer query quant/coeff/cache publication lets AIC start Score while AIV processes main"
        ),
        "risk": "AIV delayed Score consumption may lose overlap and backpressure the existing two slots",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    }
    with (here / "main_vector_late.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
