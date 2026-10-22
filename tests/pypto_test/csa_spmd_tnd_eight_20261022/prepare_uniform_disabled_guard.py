# SPDX-License-Identifier: Apache-2.0
"""Prune uniform lookup before inlining when its resource tier is disabled."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    expected = json.loads((here / "uniform_geometry_v1.json").read_text())["files"]
    source = {p.name: p.read_text() for p in (here / "runs/uniform_geometry_v1").glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == expected
    name = "attention_plan_stages.py"
    source[name] = replace_once(
        source[name],
        "            request, begin, end, active = query_request_with_uniform(\n"
        "                bounds, lengths, token, tokens, uniform_width\n"
        "            )\n",
        "            if PLAN_PHYSICAL_ROWS:\n"
        "                request, begin, end, active = query_request_with_uniform(\n"
        "                    bounds, lengths, token, tokens, uniform_width\n"
        "                )\n"
        "            else:\n"
        "                request, begin, end, active = query_request(bounds, lengths, token, tokens)\n",
    )
    source[name] = replace_once(
        source[name], "    uniform_width: pl.Scalar[pl.INDEX],\n",
        "    uniform_width: pl.Scalar[pl.INDEX],\n    UNIFORM_LOOKUP: pl.constexpr,\n",
    )
    source[name] = replace_once(
        source[name],
        "        owner, begin, end, owner_active = query_request_with_uniform(\n"
        "            bounds, lengths, query, pl.tensor.dim(positions, 0), uniform_width\n"
        "        )\n",
        "        if UNIFORM_LOOKUP:\n"
        "            owner, begin, end, owner_active = query_request_with_uniform(\n"
        "                bounds, lengths, query, pl.tensor.dim(positions, 0), uniform_width\n"
        "            )\n"
        "        else:\n"
        "            owner, begin, end, owner_active = query_request(\n"
        "                bounds, lengths, query, pl.tensor.dim(positions, 0)\n"
        "            )\n",
    )
    name = "attention_stream_stages.py"
    source[name] = replace_once(source[name], "                        uniform_width,\n",
                                "                        uniform_width, PLAN_PHYSICAL_ROWS,\n")
    target = here / "runs/uniform_geometry_v2"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    report = dict(
        source=str(target), parent="uniform_geometry_v1", baseline="baseline.json", adopted=False,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        changes=["attention_plan_stages.py", "attention_stream_stages.py"],
        reason="When tier is off, width=0 causes early constant folding of unreachable integer division",
        fix="Use existing constexpr resource predicate to prune helper calls before inlining",
        unchanged="Every on-path source operation; must prove C04/C05 compiled equality before reusing timing",
    )
    with (here / "uniform_geometry_v2.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
