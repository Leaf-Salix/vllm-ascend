# SPDX-License-Identifier: Apache-2.0
"""Reuse plan validity to avoid per-row bounds branches on proven full blocks."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "merged_schedule_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    plan = source["attention_plan_stages.py"]
    plan = replace_once(plan, "QUERY_POSITION_COL = 6\n", "QUERY_POSITION_COL = 6\nCOMPRESSED_ALL_VALID_COL0 = 7\n")
    marker = "    return indices, bias, valid_blocks, signals\n"
    plan = replace_once(
        plan,
        marker,
        """        if PLAN_PHYSICAL_ROWS and physical_mode != 0:
            # c_mask_tile already includes ownership, causal and physical-page
            # validity. Every actual row is rewritten each generation, including
            # inactive rows. Columns 7..10 are separate from event predicates 0..4
            # and query geometry 5/6; one plan owner writes the complete 64B row.
            for proof_block in pl.range(SPARSE_BLOCKS - 1):
                block_all_valid = pl.row_min(
                    pl.slice(
                        c_mask_tile, [ROWS, ATTN_K_TILE], [0, proof_block * ATTN_K_TILE],
                        valid_shape=[rows, ATTN_K_TILE],
                    ),
                    reduce_tmp,
                )
                for proof_row in pl.range(rows):
                    all_valid_value = pl.cast(pl.read(block_all_valid, [proof_row, 0]), pl.INT32)
                    pl.write(
                        valid_blocks, [first + proof_row, COMPRESSED_ALL_VALID_COL0 + proof_block], all_valid_value
                    )
"""
        + marker,
    )
    source["attention_plan_stages.py"] = plan
    stream = source["attention_stream_stages.py"]
    stream = replace_once(
        stream, "    COMPRESSED_COLUMNS_DYN,\n", "    COMPRESSED_COLUMNS_DYN,\n    COMPRESSED_ALL_VALID_COL0,\n"
    )
    old = """                    qk_kv_half = pl.tile.mgather(
                        cmp_kv_flat, physical_rows, coalesce="row", gather_oob="zero"
                    )
"""
    new = """                    if pl.read(valid_block_mask, [qk_t, COMPRESSED_ALL_VALID_COL0 + qk_sb - 1]) == 1:
                        # The same-generation plan proves every row in this full
                        # 128-row block. Both 64-row lanes may omit OOB branches.
                        # Any invalid row keeps the original zero-fill reader.
                        qk_kv_half = pl.tile.mgather(
                            cmp_kv_flat, physical_rows, coalesce="row", gather_oob="undefined"
                        )
                    else:
                        qk_kv_half = pl.tile.mgather(
                            cmp_kv_flat, physical_rows, coalesce="row", gather_oob="zero"
                        )
"""
    source["attention_stream_stages.py"] = replace_once(stream, old, new)
    for content in source.values():
        ast.parse(content)
    target = here / "runs/plan_gather_valid_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        (target / name).write_text(content)
    report = dict(
        source=str(target),
        parent="merged_schedule_v1.json",
        adopted=False,
        device_execution=False,
        mechanism=(
            "Reduce existing page-valid masks per 128-row block; use undefined-OOB MGATHER only with current plan proof"
        ),
        unchanged=(
            "All math, order, event predicates 0..4, geometry 5/6, FFTS protocol, ABI, allocation and retained policies"
        ),
        new_metadata=(
            "Private attention_mask columns 7..10, rewritten for all real capacity rows by the same plan owner"
        ),
        fallback=(
            "Any invalid row keeps MGATHER Zero for the entire 128-row block; "
            "unsupported physical layout stays original"
        ),
        codegen_gate="Full C04/C05 code and resource/ABI checks, plus disabled C03 C++ equivalence",
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
    )
    with (here / "plan_gather_valid_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
