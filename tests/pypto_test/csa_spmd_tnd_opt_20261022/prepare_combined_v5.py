# SPDX-License-Identifier: Apache-2.0
"""Preserve B1 quantization reuse and the existing G6 Score improvements.

The v4 interaction regressed. This package does not move main work or remove
idle publications. Complete generated-program comparisons determine whether
the B1 and G6 domains reproduce their separately measured implementations.
"""

import ast
import hashlib
import json
import shutil
from pathlib import Path


def hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.py")}


def main():
    here = Path(__file__).resolve().parent
    parents = {}
    for name in ("output_quant_guard", "combined_v1", "frozen_v122"):
        path = here / "runs" / name
        assert hashes(path) == json.loads((here / (name + ".json")).read_text())["files"]
        parents[name] = path
    target = here / "runs/combined_v5"
    shutil.copytree(parents["output_quant_guard"], target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    file = "indexer_score_stages.py"
    shutil.copyfile(parents["combined_v1"] / file, target / file)
    assert [n for n, sha in hashes(target).items() if sha != hashes(parents["output_quant_guard"])[n]] == [file]
    for file in target.glob("*.py"):
        ast.parse(file.read_text())
    report = dict(
        source=str(target),
        reference_kind="frozen_tnd_v122",
        adopted=False,
        device_execution=False,
        components=["output_quant_guard", "live_half", "batched_sort3_v2"],
        excluded=["main_vector_late", "idle publications", "all Query head rewrites", "combined_v3", "combined_v4"],
        sources={k: str(v) for k, v in parents.items()},
        files=hashes(target),
        required="Full generated B1 vs quant-only and B16 vs combined_v1 C++/packed ABI/host comparisons",
        limitations="Removing interactions does not establish their individual cause or additive gains",
    )
    with (here / "combined_v5.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
