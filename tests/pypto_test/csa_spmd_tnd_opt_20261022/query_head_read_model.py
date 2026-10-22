# SPDX-License-Identifier: Apache-2.0
"""Logical reads of the two Query orientations; never an HBM/latency estimate."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace


def head_selector(source):
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "q_dequant_rope_stage")
    prefix = []
    for statement in fn.body:
        if isinstance(statement, ast.For):
            break
        prefix.append(statement)
    stub = ast.parse("def select(q_flat):\n    pass").body[0]
    stub.body = prefix + [ast.Return(value=ast.Name(id="dq_head_tile", ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[]))
    ns = dict(
        pl=SimpleNamespace(tensor=SimpleNamespace(dim=lambda value, axis: value)),
        Q_ROPE_T_TILE=8,
        Q_ROPE_H_TILE=4,
        Q_DEQUANT_WORKERS=48,
        H=64,
    )
    exec(compile(module, "<original-head-selector>", "exec"), ns)
    return ns["select"]


def main():
    here = Path(__file__).resolve().parent
    selector = head_selector((here / "runs/frozen_v122/query_projection_stages.py").read_text())
    cases = []
    for tokens in (6, 24, 48, 56, 96, 144, 168, 192):
        if tokens % 8:
            cases.append(dict(tokens=tokens, path="original whole helper (token tail)"))
            continue
        head_tile = selector(tokens)
        old_groups = tokens // 8 * (64 // head_tile)
        new_groups = tokens * (64 // 8)
        common = dict(
            q_input_bytes=tokens * 64 * 512 * 4, q_output_bytes=tokens * 64 * 512 * 2, rms_8x512_tiles=tokens * 64 // 8
        )
        cases.append(
            dict(
                tokens=tokens,
                path="contiguous heads",
                common=common,
                original_head_group=head_tile,
                original=dict(
                    preparation_groups=old_groups,
                    weight_scale_bytes=tokens // 8 * 64 * 512 * 4,
                    scale_rope_metadata_bytes=old_groups * (8 * 4 + 3 * 8 * 64 * 4),
                ),
                candidate=dict(
                    preparation_groups=new_groups,
                    weight_scale_bytes=tokens * 64 * 512 * 4,
                    scale_rope_metadata_bytes=new_groups * (4 + 3 * 64 * 4),
                ),
            )
        )
    report = dict(
        scope="Logical element read/write counts from original device selector and candidate tiling",
        limitations=[
            "Not physical HBM traffic: caching, alignment, repeats and outstanding transfers are not modeled",
            "Instruction cost and complete critical path are not modeled",
            "No performance prediction",
        ],
        cases=cases,
    )
    with (here / "query_head_read_model.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    for row in cases:
        if "common" in row:
            print(
                row["tokens"],
                "old head group",
                row["original_head_group"],
                "weight scale logical bytes",
                row["original"]["weight_scale_bytes"],
                "->",
                row["candidate"]["weight_scale_bytes"],
                "metadata logical bytes",
                row["original"]["scale_rope_metadata_bytes"],
                "->",
                row["candidate"]["scale_rope_metadata_bytes"],
            )


if __name__ == "__main__":
    main()
