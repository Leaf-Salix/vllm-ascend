# SPDX-License-Identifier: Apache-2.0
"""Use Tensor assemble for existing Tensor intermediates, retaining valid rows."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "hc_direct_tail_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    hc = source["hc_stages.py"]
    hc = replace_once(
        hc,
        "                    pl.store(w_valid, [w_t0, w_d0], x_hc32_flat)\n",
        "                    x_hc32_flat = pl.assemble(x_hc32_flat, w_valid, [w_t0, w_d0])\n",
    )
    hc = replace_once(hc, "    return x_fp32, inv_rms\n", "    return x_hc32_flat, inv_rms\n")
    hc = replace_once(
        hc,
        "                    pl.store(normed_valid, [t0, d0], x_normed)\n",
        "                    x_normed = pl.assemble(x_normed, normed_valid, [t0, d0])\n",
    )
    ast.parse(hc)
    source["hc_stages.py"] = hc
    target = here / "runs/hc_direct_tail_v2"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        (target / name).write_text(content)
    report = dict(
        parent,
        source=str(target.resolve()),
        parent="hc_direct_tail_v1.json",
        predecessor_failure="tile.store requires TileType, but these source intermediates are TensorType",
        correction="Tensor assemble consumes the valid-shaped Tensor and returns the updated output alias",
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
    )
    with (here / "hc_direct_tail_v2.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
