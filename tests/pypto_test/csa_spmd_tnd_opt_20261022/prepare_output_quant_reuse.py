# SPDX-License-Identifier: Apache-2.0
"""Reuse the original O_A tile after amax instead of loading it twice.

This candidate is independent of the Query experiments and starts at v122.
The generated baseline contains both TLOADs, including in each unrolled lane.
Keeping the original tile live requires a separate absolute-value destination.
"""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_query_head_guard import edit_functions


def transform(source):
    def edit(fn):
        first = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "oc_amax")
        second = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "oc_q")
        assert ast.dump(first.value) == ast.dump(second.value)
        assert ast.unparse(first.value.func) == "pl.load"
        second.value = ast.Name(id="oc_amax", ctx=ast.Load())

    return edit_functions(source, ["o_projection_groups_quant_stage"], edit)


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob("*.py")} == expected
    target = here / "runs/output_quant_reuse"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    name = "output_projection_a_stages.py"
    (target / name).write_text(transform((parent / name).read_text()))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")}
    assert [n for n in files if files[n] != expected[n]] == [name]
    report = dict(
        source=str(target),
        parent="frozen_tnd_v122",
        reference_kind="frozen_tnd_v122",
        files=files,
        modified_files=[name],
        adopted=False,
        device_execution=False,
        change="Replace the second identical O_A input load by the original tile alias",
        retained="amax, divide, dequant scale product, rint/FP16/INT8 conversions, valid rows, tail zero and owners",
        logical_input_read_bytes_b16=dict(reference=6291456, candidate=3145728),
        limitations=(
            "Logical bytes are not HBM traffic or latency; physical alias lifetime and complete L1 must be checked"
        ),
    )
    with (here / "output_quant_reuse.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
