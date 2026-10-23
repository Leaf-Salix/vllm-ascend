# SPDX-License-Identifier: Apache-2.0
"""Interleave existing N64 QK halves with the preceding panel's WS halves."""

import ast
import hashlib
import json
import shutil
from pathlib import Path


def reorder(source):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store")
    branch = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "buf_qk_panel > 0")

    def assignment(name):
        return next(
            n
            for n in branch.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
        )

    ws = assignment("split_ws_first")
    qk1 = assignment("split_qk1")
    assert qk1.end_lineno < ws.lineno
    lines = source.splitlines(keepends=True)
    moved = lines[ws.lineno - 1 : ws.end_lineno]
    del lines[ws.lineno - 1 : ws.end_lineno]
    lines[qk1.lineno - 1 : qk1.lineno - 1] = moved
    return "".join(lines)


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    name = "indexer_score_stages.py"
    source[name] = reorder(source[name])
    ast.parse(source[name])
    target = here / "runs/score_half_interleave_v1"
    target.mkdir(exist_ok=False)
    for name, value in source.items():
        (target / name).write_text(value)
    shutil.copy2(
        here / "descriptors/C04_output_rectangle_v1.json", here / "descriptors/C04_score_half_interleave_v1.json"
    )
    report = dict(
        source=str(target),
        parent="output_rectangle_production_v1.json",
        adopted=False,
        device_execution=False,
        mechanism="QK0/FIX0 -> WS(previous) K0 -> QK1/FIX1 -> WS(previous) K1 at the original N64",
        unchanged=(
            "Two QK Acc buffers, independent WS Acc, all operands, per-output K order, key prefetch, events and stores"
        ),
        scope="Existing full G6 split branch; <=3 live-query path and all Vector code unchanged; B16 pilot only",
        historical_exclusion="The measured N128/single-QK-Acc variant used this order to cover destructive Acc reuse; "
        "this prototype keeps N64 and both original QK Acc allocations",
        reference="ops-transformer@28f4035406a1aea65a8d65c6eb527a85ab84e624: "
        "attention/quant_lightning_indexer/op_kernel/arch22/quant_lightning_indexer_service_cube_arch22.h",
        files={name: hashlib.sha256(value.encode()).hexdigest() for name, value in source.items()},
    )
    with (here / "score_half_interleave_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
