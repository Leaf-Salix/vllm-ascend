# SPDX-License-Identifier: Apache-2.0
"""Keep the physical eight-row reduction tile but prove only one valid row."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/leaf_scale_v1"
    expected = json.loads((here / "leaf_scale_v1.json").read_text())["files"]
    source = {p.name: p.read_text() for p in parent.glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == expected
    name = "leaf_scale_layout_stages.py"
    source[name] = replace_once(
        source[name], "                                LAYOUT_PROOF_ROWS,\n", "                                1,\n"
    )
    source[name] = replace_once(
        source[name],
        "pl.col_expand_mul(ones, difference), LAYOUT_PROOF_ROWS, pages",
        "pl.col_expand_mul(ones, difference), 1, pages",
    )
    target = here / "runs/leaf_scale_one_row_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    report = dict(
        source=str(target),
        parent="leaf_scale_v1",
        baseline="baseline.json",
        adopted=False,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        changes=["leaf_scale_layout_stages.py"],
        mechanism="Proof compares the same complete page list once; seven repeated identical rows are invalid",
        unchanged="Physical 8x256 tile, page/tail/FP32 integer bounds, forward/reverse predicates, all other code",
        reason="Leaf proof with eight valid rows regressed both priority cases; isolate its redundant Vector work",
    )
    with (here / "leaf_scale_one_row_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
