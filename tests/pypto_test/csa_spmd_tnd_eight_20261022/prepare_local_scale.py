# SPDX-License-Identifier: Apache-2.0
"""Freeze local-contiguity proof independently of previous failed scale candidates."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    source = {p.name: p.read_text() for p in (here / "runs/baseline").glob("*.py")}
    expected = json.loads((here / "baseline.json").read_text())["files"]
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == expected
    file = "indexer_score_stages.py"
    tree = ast.parse(source[file])
    loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "single_scale_page"
    ]
    assert len(loops) == 1
    loop = loops[0]
    lines = source[file].splitlines(keepends=True)
    lines[loop.lineno - 1 : loop.end_lineno] = [
        " " * loop.col_offset + line + "\n"
        for line in (
            "single_scale_bytes = read_local_scales(",
            "    idx_native_kv_cache, idx_block_table, single_batch, single_score_begin, single_cache_len, score_tile",
            ")",
        )
    ]
    source[file] = replace_once(
        "".join(lines),
        "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .local_scale_stages import read_local_scales\n",
    )
    source["local_scale_stages.py"] = (here / "local_scale_source.py").read_text()
    target = here / "runs/local_scale_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    report = dict(
        source=str(target),
        baseline="baseline.json",
        adopted=False,
        device_execution=False,
        changes=[file, "local_scale_stages.py"],
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        mechanism="Prove a complete 32-page run locally, then use one 2-D strided load; preserve old fallback exactly",
        limitations="Local proof overhead may offset fewer DMA instructions; device benefit remains unproven",
    )
    with (here / "local_scale_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
