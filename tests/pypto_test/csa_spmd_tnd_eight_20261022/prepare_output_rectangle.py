# SPDX-License-Identifier: Apache-2.0
"""Freeze a single-M160/N192 O_A grid with an explicit 64-column final tile.

Forty-eight O_A jobs give every AIC two jobs. K256 accumulation, quantization
and O_B ownership stay unchanged; this differs from the M192/N128 + O_B
redistribution trial. The token-capacity domain is 128<T<=160, not a B/S list.
"""

import ast
import hashlib
import json
import runpy
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "physical_capacity_production_v1.json").read_text())
    original = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    source = dict(original)
    policy = replace_once(source["policy.py"], "    ob_rows: int\n", "    ob_rows: int\n    oa_rows: int = 128\n")
    policy = replace_once(
        policy,
        '            "OA_COLS": self.oa_cols,\n',
        '            "OA_COLS": self.oa_cols,\n            "OA_ROWS": self.oa_rows,\n',
    )
    policy = replace_once(
        policy, "    return CSASPMDPolicy(\n", "    output_rectangle = 128 < tokens <= 160\n    return CSASPMDPolicy(\n"
    )
    policy = replace_once(
        policy,
        "        oa_cols=128 if tokens <= 96 else 256,\n",
        "        oa_cols=192 if output_rectangle else (128 if tokens <= 96 else 256),\n"
        "        oa_rows=160 if output_rectangle else 128,\n",
    )
    source["policy.py"] = policy
    for name in ("entry.py", "cube_schedule.py", "vector_schedule.py"):
        old = "    OA_COLS: pl.constexpr,\n"
        assert source[name].count(old) == (2 if name == "entry.py" else 1)
        source[name] = source[name].replace(old, old + "    OA_ROWS: pl.constexpr,\n")
    call = "            OA_COLS,\n"
    assert source["entry.py"].count(call) == 3
    source["entry.py"] = source["entry.py"].replace(call, call + "            OA_ROWS,\n")
    for name in ("cube_schedule.py", "vector_schedule.py"):
        old = "((tokens + 127) // 128) * (1024 // OA_COLS)"
        assert source[name].count(old) == (1 if name == "cube_schedule.py" else 2)
        source[name] = source[name].replace(
            old, "((tokens + OA_ROWS - 1) // OA_ROWS) * ((1024 + OA_COLS - 1) // OA_COLS)"
        )
    source["cube_schedule.py"] = replace_once(
        source["cube_schedule.py"],
        "GROUPS, OA_COLS, oa_group, oa_group + 1",
        "GROUPS, OA_COLS, OA_ROWS, oa_group, oa_group + 1",
    )
    source["cube_schedule.py"] = replace_once(
        source["cube_schedule.py"], "GROUPS, OA_COLS)\n", "GROUPS, OA_COLS, OA_ROWS)\n"
    )
    name = "output_projection_a_stages.py"
    lines = source[name].splitlines(keepends=True)
    names = {"o_projection_a_stage_nd", "o_projection_a_groups_stage_nz", "o_projection_a_stage_nz"}
    functions = [
        node for node in ast.parse(source[name]).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(functions) == 3
    for node in reversed(functions):
        body = "".join(lines[node.lineno - 1 : node.end_lineno]).replace("PROJ_A_ROW_TILE", "A_ROW_TILE")
        body = replace_once(
            body, "    A_COL_TILE: pl.constexpr,\n", "    A_COL_TILE: pl.constexpr,\n    A_ROW_TILE: pl.constexpr,\n"
        )
        if node.name == "o_projection_a_stage_nz":
            body = replace_once(
                body, "WORKERS, A_COL_TILE, 0, O_GROUPS", "WORKERS, A_COL_TILE, A_ROW_TILE, 0, O_GROUPS"
            )
        else:
            body = body.replace("O_LORA // A_COL_TILE", "(O_LORA + A_COL_TILE - 1) // A_COL_TILE")
            body = replace_once(
                body,
                "        wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]\n",
                "        if A_COL_TILE == 192:\n"
                "            columns = pl.min(A_COL_TILE, O_LORA - n0)\n"
                "            wa_first = pl.slice(wo_a, [1, A_K_TILE, A_COL_TILE], [g, 0, n0],\n"
                "                                valid_shape=[1, A_K_TILE, columns])\n"
                "        else:\n"
                "            wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]\n",
            )
            body = replace_once(
                body,
                "            wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]\n",
                "            if A_COL_TILE == 192:\n"
                "                columns_k = pl.min(A_COL_TILE, O_LORA - n0)\n"
                "                wa_k_chunk = pl.slice(wo_a, [1, A_K_TILE, A_COL_TILE], [g, k0, n0],\n"
                "                                      valid_shape=[1, A_K_TILE, columns_k])\n"
                "            else:\n"
                "                wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]\n",
            )
        lines[node.lineno - 1 : node.end_lineno] = [body]
    source[name] = "".join(lines)
    target = here / "runs/output_rectangle_v1"
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        ast.parse(text)
        (target / name).write_text(text)
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    cases = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
    for case in cases:
        if case["id"] not in ("C04", "C05"):
            continue
        desc = json.loads((here / f"descriptors/{case['id']}_physical_capacity_tier_v1.json").read_text())
        desc["policy"] = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
            compressed_table_columns=desc["buffers"]["compressed_table"]["shape"][1],
        ).constexprs()
        with (here / f"descriptors/{case['id']}_output_rectangle_v1.json").open("x") as stream:
            stream.write(json.dumps(desc, indent=2) + "\n")
    report = dict(
        source=str(target.resolve()),
        parent="physical_capacity_production_v1.json",
        adopted=False,
        guard="128 < Host token capacity <= 160; actual T/Q/L remain device data",
        mechanism="M160 x N192, 48 O_A jobs with a valid N64 tail; retain current O_B owners",
        costs="Larger N per job; explicit column tail and changed quant-owner sets; no device benefit assumed",
        logical_work="Same output and K chain; one M pass reads 64MiB unique O_A weights instead of two logical passes",
        changes=[name for name in source if source[name] != original[name]],
        files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()},
        gate="Exact column-tail stores, NZ layout, unchanged K chain/ownership, memory and full L1 before adoption",
    )
    with (here / "output_rectangle_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
