# SPDX-License-Identifier: Apache-2.0
"""Preserve the measured 128K gain and disable the proof beyond six base Score leaves."""

import ast
import hashlib
import json
import runpy
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/leaf_scale_guard_v1"
    expected = json.loads((here / "leaf_scale_guard_v1.json").read_text())["files"]
    source = {p.name: p.read_text() for p in parent.glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == expected
    text = source["policy.py"]
    # Six base 8192-candidate leaves also bound the existing 8/16-group
    # three-leaf balancing interval. Runtime page checks remain unchanged.
    node = next(
        n
        for n in ast.walk(ast.parse(text))
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "leaf_scale_layout" for t in n.targets)
    )
    lines = text.splitlines(keepends=True)
    lines[node.lineno - 1 : node.end_lineno] = [
        "    leaf_scale_layout = (" + ast.unparse(node.value) + " and cache_rows <= LEAF_LAYOUT_MAX_CACHE_ROWS)\n"
    ]
    text = replace_once(
        "".join(lines),
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass\n\n"
        "# At most six original 8192-candidate leaves before balancing.\n"
        "LEAF_LAYOUT_MAX_CACHE_ROWS = 6 * 8192\n",
    )
    source["policy.py"] = text
    target = here / "runs/leaf_scale_guard_v2"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        (target / name).write_text(content)
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    base = json.loads((here / "baseline.json").read_text())
    prior = json.loads((here / "leaf_scale_guard_v1.json").read_text())["eight_case_policies"]
    decisions = {}
    for cid, record in base["cases"].items():
        case = record["case"]
        policy = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        ).constexprs()
        assert policy["LEAF_SCALE_LAYOUT"] == (cid == "C04")
        if cid != "C02":
            assert policy == prior[cid]
        else:
            original = json.loads(Path(record["source_report"]).read_text())
            assert {k: v for k, v in policy.items() if k != "LEAF_SCALE_LAYOUT"} == original["policies"]["spmd"]
            (here / "descriptors/C02_leaf_scale_guard_v2.json").write_text(
                json.dumps(dict(buffers=original["descriptors"]["spmd"], policy=policy), indent=2) + "\n"
            )
        decisions[cid] = policy
    report = dict(
        source=str(target),
        parent="leaf_scale_guard_v1",
        baseline="baseline.json",
        adopted=False,
        changes=["policy.py"],
        guard="long G6, 64<=token_capacity<=96, compressed history upper bound <=6*8192 rows",
        reason="C04 gains 1.89%; C02 has no gain. Bound the existing balanced-leaf resource interval",
        unchanged_kernel_sources=True,
        preserved_C04_performance_state_and_service=True,
        preserved_other_six_off_classes=True,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        eight_case_policies=decisions,
    )
    with (here / "leaf_scale_guard_v2.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
