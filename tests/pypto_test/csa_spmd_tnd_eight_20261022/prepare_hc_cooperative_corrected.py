# SPDX-License-Identifier: Apache-2.0
"""Normalize full/tail tensor types before the unchanged K512 row reduction."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "hc_cooperative_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    content = source["hc_cooperative_stages.py"]
    content = replace_once(
        content,
        """            if rows == T_TILE:
                squared = pl.mul(value, value)
                row_sum = pl.reshape(pl.row_sum(squared), [1, T_TILE])
            else:
                valid = pl.set_validshape(value, rows, RMS_K_TILE)
                clean = pl.fillpad(valid, pad_value=pl.PadValue.zero)
                squared = pl.mul(clean, clean)
                row_sum = pl.reshape(pl.row_sum(squared), [1, T_TILE])
""",
        """            # Full tiles have no padding; this also gives both paths one
            # Tensor type before reduction without changing valid-row values.
            valid = pl.set_validshape(value, rows, RMS_K_TILE)
            clean = pl.fillpad(valid, pad_value=pl.PadValue.zero)
            squared = pl.mul(clean, clean)
            row_sum = pl.reshape(pl.row_sum(squared), [1, T_TILE])
""",
    )
    ast.parse(content)
    source["hc_cooperative_stages.py"] = content
    target = here / "runs/hc_cooperative_v2"
    target.mkdir(exist_ok=False)
    for name, body in source.items():
        (target / name).write_text(body)
    report = dict(
        parent,
        source=str(target.resolve()),
        parent="hc_cooperative_v1.json",
        predecessor_failure="Prototype branch variable had incompatible full/tail Tensor valid-shape types",
        correction="Explicit valid-shape and zero-fill before K512 row sum on both full and partial row tiles",
        files={name: hashlib.sha256(body.encode()).hexdigest() for name, body in source.items()},
    )
    with (here / "hc_cooperative_v2.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
