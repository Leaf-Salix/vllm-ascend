# SPDX-License-Identifier: Apache-2.0
"""Give each output group three AIC owners and balance their O_A/O_B work."""

import ast
import hashlib
import json
import runpy
from pathlib import Path

from prepare_scale_gather import replace_once

HELPER = """
@pl.jit.inline(auto_scope=False)
def _three_owner_o_projection_a_groups_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    # The entry uses exactly 24 AICs; Host capacity <=96, actual T remains dynamic.
    group = pl.max(worker_id, 0) // 3
    column_lane = pl.max(worker_id, 0) % 3
    if tokens > 0 and group >= group_begin and group < group_end:
        pack_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS
        rows = pl.min(tokens, 96)
        source_row = group * pack_stride
        output_column = group * O_LORA
        o_r_pad = _rectangle_o_projection_a_tile_nz(
            o_packed, wo_a, o_r_pad, group, source_row, 0, rows,
            output_column, column_lane * 256, 96, 256,
        )
        if column_lane < 2:
            o_r_pad = _rectangle_o_projection_a_tile_nz(
                o_packed, wo_a, o_r_pad, group, source_row, 0, rows,
                output_column, 768 + column_lane * 80, 96, 80,
            )
        else:
            o_r_pad = _rectangle_o_projection_a_tile_nz(
                o_packed, wo_a, o_r_pad, group, source_row, 0, rows,
                output_column, 768 + column_lane * 80, 96, 96,
            )
    return o_r_pad
"""


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    policy = source["policy.py"]
    policy = replace_once(
        policy,
        "    output_rectangle = DEFAULT_OA_ROWS < tokens <= RECTANGLE_OA_ROWS\n",
        "    output_rectangle = DEFAULT_OA_ROWS < tokens <= RECTANGLE_OA_ROWS\n"
        "    output_three_owner = long_history and group == 6 and 64 < tokens <= 96\n",
    )
    policy = replace_once(
        policy,
        "        oa_cols=RECTANGLE_OA_COLS if output_rectangle else (128 if tokens <= 96 else 256),\n"
        "        oa_rows=RECTANGLE_OA_ROWS if output_rectangle else DEFAULT_OA_ROWS,\n",
        "        oa_cols=256 if output_three_owner else (RECTANGLE_OA_COLS if output_rectangle "
        "else (128 if tokens <= 96 else 256)),\n"
        "        oa_rows=96 if output_three_owner else (RECTANGLE_OA_ROWS if output_rectangle else DEFAULT_OA_ROWS),\n",
    )
    source["policy.py"] = policy
    name = "output_projection_a_stages.py"
    text = source[name]
    tree = ast.parse(text)
    wrapper = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "o_projection_a_groups_stage_nz"
    )
    lines = text.splitlines(keepends=True)
    body = "".join(lines[wrapper.lineno - 1 : wrapper.end_lineno])
    body = replace_once(
        body,
        "    if A_COL_TILE == 192:\n",
        "    if A_ROW_TILE == 96:\n"
        "        o_r_pad = _three_owner_o_projection_a_groups_stage_nz(\n"
        "            o_packed, wo_a, o_r_pad, tokens, worker_id, group_begin, group_end)\n"
        "    elif A_COL_TILE == 192:\n",
    )
    lines[wrapper.lineno - 1 : wrapper.end_lineno] = [body]
    source[name] = "".join(lines) + "\n" + HELPER
    for name in ("cube_schedule.py", "vector_schedule.py"):
        old = "        oa_per_group = ((tokens + OA_ROWS - 1) // OA_ROWS) * ((1024 + OA_COLS - 1) // OA_COLS)\n"
        source[name] = replace_once(
            source[name], old, old + "        if OA_ROWS == 96:\n            oa_per_group = 3 * ((tokens + 95) // 96)\n"
        )
    text = source["cube_schedule.py"]
    branch = next(
        n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.If) and ast.unparse(n.test) == "OUTPUT_WORK_BALANCE"
    )
    fallback = branch.orelse
    assert len(fallback) == 1 and isinstance(fallback[0], ast.For)

    class Rank(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "worker":
                return ast.copy_location(ast.Name(id="output_rank", ctx=node.ctx), node)
            return node

    loop = ast.unparse(ast.fix_missing_locations(Rank().visit(fallback[0])))
    replacement = (
        "        if OA_ROWS == 96:\n"
        "            output_rank = (worker // 3) * 2 + worker % 3\n"
        "            if worker % 3 == 2:\n"
        "                output_rank = 16 + worker // 3\n"
        + "\n".join("            " + line for line in loop.splitlines())
        + "\n        elif OUTPUT_WORK_BALANCE:\n"
    )
    source["cube_schedule.py"] = replace_once(text, "        if OUTPUT_WORK_BALANCE:\n", replacement)
    target = here / "runs/output_group_owners_v1"
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        ast.parse(text)
        (target / name).write_text(text)
    selected = runpy.run_path(str(target / "policy.py"))["select_policy"]
    cases = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
    for case in cases:
        if case["id"] not in ("C02", "C04", "C05"):
            continue
        desc = json.loads((here / f"descriptors/{case['id']}_output_rectangle_v1.json").read_text())
        desc["policy"] = selected(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=6,
            max_seq_len=max(case["seq_lens"]),
            compressed_table_columns=desc["buffers"]["compressed_table"]["shape"][1],
        ).constexprs()
        with (here / f"descriptors/{case['id']}_output_group_owners_v1.json").open("x") as out:
            out.write(json.dumps(desc, indent=2) + "\n")
    report = dict(
        source=str(target.resolve()),
        parent="output_rectangle_production_v1.json",
        adopted=False,
        guard="Long G6, Host token capacity in (64,96]; real metadata remains device-side",
        mechanism="Three AICs per O_A group; each computes N256 then N80/N80/N96; "
        "O_B rank permutation gives the N96 owners two jobs and N80 owners three",
        logical_costs_C04=dict(
            OA_matmul_jobs=[64, 48],
            maximum_OA_columns_per_AIC=[384, 352],
            activation_load_MiB=[48, 36],
            weight_load_MiB=[64, 64],
            producer_fanin_per_group=[8, 3],
        ),
        risks=[
            "All groups now wait for two tile calls; earliest group may be ready later",
            "O_B distribution changes together with O_A; full-chain performance required",
            "Static 80/96 column tails and real allocator limits require CPU and device verification",
            "Logical bytes and work balance do not predict HBM traffic or elapsed-time gains",
        ],
        files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()},
    )
    with (here / "output_group_owners_v1.json").open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
