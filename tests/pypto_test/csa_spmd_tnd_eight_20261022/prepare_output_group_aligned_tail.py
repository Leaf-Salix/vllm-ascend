# SPDX-License-Identifier: Apache-2.0
"""Replace the separate N80 L0 allocation with static N64/N96 output tails."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_group_owners_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    name = "output_projection_a_stages.py"
    text = replace_once(source[name], "        if column_lane < 2:\n", "        if column_lane == 0:\n")
    text = replace_once(text, "output_column, 768 + column_lane * 80, 96, 80,", "output_column, 768, 96, 64,")
    text = replace_once(
        text, "output_column, 768 + column_lane * 80, 96, 96,", "output_column, 736 + column_lane * 96, 96, 96,"
    )
    source[name] = text
    target = here / "runs/output_group_owners_v2"
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        ast.parse(text)
        (target / name).write_text(text)
    report = dict(parent)
    report.update(
        source=str(target.resolve()),
        parent="output_group_owners_v1.json",
        mechanism="Three AICs per O_A group, N256 followed by N64/N96/N96; preserve the candidate O_B rank mapping",
        predecessor_failure="N80 lowering leaves an additional 40960-byte Right allocation beside the shared "
        "65536-byte allocation; full CPU allocation rejects 106496 bytes; no device launch",
        correction="Use 64/96/96 tails at offsets 768/832/928; "
        "test actual allocation rather than infer it from tile area",
        columns_and_OB_per_core=[[320, 3], [352, 3], [352, 2]],
        files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()},
    )
    with (here / "output_group_owners_v2.json").open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
