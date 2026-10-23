# SPDX-License-Identifier: Apache-2.0
"""Supply the explicit Tile reduction and high-precision rsqrt scratch."""

import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "hc_cooperative_v3.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    content = source["hc_cooperative_stages.py"]
    content = replace_once(
        content,
        "            row_sum = pl.reshape(pl.row_sum(squared), [1, T_TILE])\n",
        "            row_sum = pl.reshape(\n"
        "                pl.row_sum(squared, tmp_tile=pl.tile.create([T_TILE, RMS_K_TILE], dtype=pl.FP32)),\n"
        "                [1, T_TILE],\n"
        "            )\n",
    )
    content = replace_once(
        content,
        "        inverse = pl.reshape(pl.rsqrt(mean, high_precision=True), [T_TILE, 1])\n",
        "        inverse = pl.reshape(\n"
        "            pl.tile.rsqrt(mean, tmp=pl.tile.create([1, T_TILE], dtype=pl.FP32)), [T_TILE, 1]\n"
        "        )\n",
    )
    source["hc_cooperative_stages.py"] = content
    target = here / "runs/hc_cooperative_v4"
    target.mkdir(exist_ok=False)
    for name, body in source.items():
        (target / name).write_text(body)
    report = dict(
        parent,
        source=str(target.resolve()),
        parent="hc_cooperative_v3.json",
        predecessor_failure="Tile row_sum requires explicit same-shape scratch; "
        "Tensor lowering had supplied it implicitly",
        correction="Explicit row_sum scratch and Tile rsqrt scratch preserving the original high-precision mode",
        files={name: hashlib.sha256(body.encode()).hexdigest() for name, body in source.items()},
    )
    with (here / "hc_cooperative_v4.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
