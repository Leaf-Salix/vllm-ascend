# SPDX-License-Identifier: Apache-2.0
"""Batch the new RMS handoff while preserving its 32 individual row sums."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "hc_cooperative_v2.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    content = source["hc_cooperative_stages.py"]
    content = replace_once(
        content,
        "        for local_chunk in pl.pipeline(RMS_CHUNKS_PER_SPLIT, stage=4):\n",
        "        partial_rows = pl.tile.create([RMS_CHUNKS_PER_SPLIT, T_TILE], dtype=pl.FP32)\n"
        "        for local_chunk in pl.pipeline(RMS_CHUNKS_PER_SPLIT, stage=4):\n",
    )
    content = replace_once(
        content,
        "            source = pl.slice(x, [T_TILE, RMS_K_TILE], [row, column], valid_shape=[rows, RMS_K_TILE])\n",
        "            source = pl.load(x, [row, column], [T_TILE, RMS_K_TILE],\n"
        "                             valid_shape=[rows, RMS_K_TILE], target_memory=pl.MemorySpace.Vec)\n",
    )
    content = replace_once(
        content,
        "            x_fp32 = pl.assemble(x_fp32, output, [row, column])\n"
        "            rms_parts = pl.assemble(rms_parts, row_sum, "
        "[split, row_tile * RMS_SLOT_WORDS + local_chunk * T_TILE])\n",
        "            pl.store(output, [row, column], x_fp32)\n"
        "            partial_rows = pl.tile.assemble(partial_rows, row_sum, [local_chunk, 0])\n"
        "        partial_packet = pl.tile.reshape(partial_rows, [1, RMS_CHUNKS_PER_SPLIT * T_TILE])\n"
        "        pl.store(partial_packet, [split, row_tile * RMS_SLOT_WORDS], rms_parts)\n",
    )
    content = replace_once(
        content,
        """        total = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for chunk in pl.pipeline(HC_DIM // RMS_K_TILE, stage=2):
            split = chunk // RMS_CHUNKS_PER_SPLIT
            local_chunk = chunk % RMS_CHUNKS_PER_SPLIT
            part = pl.slice(rms_parts, [1, T_TILE], [split, row_tile * RMS_SLOT_WORDS + local_chunk * T_TILE])
            total = pl.add(total, part)
""",
        """        # One strided load acquires all four immutable packets. The reshape
        # only changes the local UB view; K512 chunks remain in original order.
        packets = pl.load(rms_parts, [0, row_tile * RMS_SLOT_WORDS],
                          [LINEAR_OK, RMS_CHUNKS_PER_SPLIT * T_TILE], target_memory=pl.MemorySpace.Vec)
        ordered_parts = pl.tile.reshape(packets, [HC_DIM // RMS_K_TILE, T_TILE])
        total = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for chunk in pl.range(HC_DIM // RMS_K_TILE):
            part = pl.tile.extract(ordered_parts, chunk, 0, [1, T_TILE], target_memory=pl.MemorySpace.Vec)
            total = pl.add(total, part)
""",
    )
    content = replace_once(
        content,
        "        inv_rms = pl.assemble(inv_rms, inverse, [row_tile * T_TILE, 0])\n",
        "        pl.store(inverse, [row_tile * T_TILE, 0], inv_rms)\n",
    )
    ast.parse(content)
    source["hc_cooperative_stages.py"] = content
    target = here / "runs/hc_cooperative_v3"
    target.mkdir(exist_ok=False)
    for name, body in source.items():
        (target / name).write_text(body)
    report = dict(
        parent,
        source=str(target.resolve()),
        parent="hc_cooperative_v2.json",
        mechanism="Same four K4096 input owners, with one eight-chunk RMS packet per writer and one consumer load",
        packed_handoff="Four 1x64 FP32 stores and one 4x64 strided load per M8 tile; no sum within each packet",
        additional_ub="Producer separate row sums: 256B; consumer packets: 1024B; subject to actual allocation",
        queued_v2_unchanged=True,
        gate="Exact CPU source math, emitted local layout/extract/carry and sync audit; NPU selection awaits v2 result",
        files={name: hashlib.sha256(body.encode()).hexdigest() for name, body in source.items()},
    )
    with (here / "hc_cooperative_v3.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
