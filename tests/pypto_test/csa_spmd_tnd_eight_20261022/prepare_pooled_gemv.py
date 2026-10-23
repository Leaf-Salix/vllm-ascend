# SPDX-License-Identifier: Apache-2.0
"""Prepare pooled full-G6 GEMV with a matched panel-major GM handoff."""

import argparse
import ast
import hashlib
import json
import shutil
import textwrap
from pathlib import Path

from prepare_score_gemv import transform as independent_gemv


def edits(source, replacements):
    lines = source.splitlines(keepends=True)
    changes = sorted(replacements)
    assert all(left[1] <= right[0] for left, right in zip(changes, changes[1:]))
    for start, end, replacement in reversed(changes):
        lines[start:end] = [replacement] if replacement else []
    result = "".join(lines)
    ast.parse(result)
    return result


def cube_transform(source):
    source = independent_gemv(source)
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store")
    changes = []

    def change(node, replacement=""):
        changes.append((node.lineno - 1, node.end_lineno, textwrap.indent(replacement, " " * node.col_offset)))

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "split_ws_acc":
                if ast.unparse(node.value.func) == "pl.tile.create":
                    change(
                        node,
                        "split_ws_acc = pl.tile.create([16, query_group_size * qk_panel_cols], "
                        "dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)\n"
                        "gemv_packed_transfer = pl.reshape(buf_score_transfer, "
                        "[pl.tensor.dim(buf_score_transfer, 0) * pl.tensor.dim(buf_score_transfer, 1) "
                        "// (query_group_size * qk_panel_cols), query_group_size * qk_panel_cols])\n",
                    )
                else:
                    assert ast.unparse(node.value.func) == "pl.set_validshape"
                    change(node)
            elif name.startswith("gemv_result"):
                query = int(name.removeprefix("gemv_result")[0])
                suffix = "_last" if name.endswith("_last") else ""
                change(
                    node,
                    f"gemv_pool_live{query}{suffix} = pl.tile.set_validshape("
                    "split_ws_acc, 1, query_group_size * qk_panel_cols)\n"
                    f"gemv_slot{query}{suffix} = pl.tile.slice(gemv_pool_live{query}{suffix}, "
                    f"[16, qk_panel_cols], [0, {query} * qk_panel_cols], valid_shape=[1, qk_panel_cols])\n"
                    f"{name} = pl.tile.gemv_acc(gemv_slot{query}{suffix}, gemv_coef{query}, "
                    f"gemv_right{query}{suffix}, init_cond=True)\n"
                    f"split_ws_acc = pl.tile.assemble(split_ws_acc, {name}, [0, {query} * qk_panel_cols])\n",
                )
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if ast.unparse(call.func) != "pl.store" or not isinstance(call.args[0], ast.Name):
                continue
            name = call.args[0].id
            if not name.startswith("gemv_result"):
                continue
            query = int(name.removeprefix("gemv_result")[0])
            if query != 5:
                change(node)
                continue
            column = "score_tile - qk_panel_cols" if name.endswith("_last") else "buf_panel_col - qk_panel_cols"
            change(
                node,
                "pl.store(pl.tile.set_validshape(split_ws_acc, 1, query_group_size * qk_panel_cols), "
                "[buf_transfer_row * pl.tensor.dim(buf_score_transfer, 1) // (query_group_size * qk_panel_cols) "
                f"+ ({column}) // qk_panel_cols, 0], gemv_packed_transfer)\n",
            )
    assert len(changes) == 26
    result = edits(source, changes)
    result = result.replace(
        "One shared physical-M16 Acc stores each row independently. Full CSA",
        "One physical-M16 Acc column pool emits one panel-major GM store. Full CSA",
    )
    return result


def vector_transform(source):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_vector_lane_stage")
    load = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "single_scores"
    )
    step = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and ast.unparse(n.target) == "single_step")
    index = step.body.index(load)
    acknowledge = step.body[index + 1]
    assert isinstance(acknowledge, ast.Expr)
    assert ast.unparse(acknowledge.value.func) == "pl.system.sync_set"
    assert ast.unparse(acknowledge.value.args[0]) == "SCORE_CONSUMED_EVENT"
    original_load = ast.unparse(load.value)
    packed_condition = (
        "SPLIT_SCORE_ROWS and balance_leaves and query_group_size == 6 and qk_panel_cols == 64 and score_tile == 1024"
    )
    # ACK remains directly after the MTE2 load; the producer may reuse GM while
    # this lane permutes its private UB payload. Both runtime paths send once.
    block = f"""\
single_transport = pl.tile.create([query_group_size // 2, score_tile], dtype=pl.FP32)
if {packed_condition}:
    if buf_live > 3:
        gemv_transport = pl.reshape(buf_score_transfer, [
            pl.tensor.dim(buf_score_transfer, 0) * pl.tensor.dim(buf_score_transfer, 1)
            // (query_group_size * qk_panel_cols), query_group_size * qk_panel_cols,
        ])
        gemv_transport_row = (
            (single_transfer_row - buf_score_lane * (query_group_size // 2))
            * pl.tensor.dim(buf_score_transfer, 1) // (query_group_size * qk_panel_cols)
        )
        gemv_panels = pl.load(
            gemv_transport, [gemv_transport_row, buf_score_lane * (query_group_size // 2) * qk_panel_cols],
            [score_tile // qk_panel_cols, (query_group_size // 2) * qk_panel_cols],
        )
        single_transport = pl.tile.reshape(gemv_panels, [query_group_size // 2, score_tile])
    else:
        single_transport = {original_load}
else:
    single_transport = {original_load}
{ast.unparse(acknowledge)}
single_scores = single_transport
if {packed_condition}:
    if buf_live > 3:
        gemv_linear = pl.tile.arange(0, [1, (query_group_size // 2) * score_tile], dtype=pl.INT32)
        gemv_query = pl.tile.shrs(gemv_linear, 10)
        gemv_panel = pl.tile.ands(pl.tile.shrs(gemv_linear, 6), 15)
        gemv_position = pl.tile.ands(gemv_linear, 63)
        gemv_indices = pl.add(
            pl.add(pl.mul(gemv_panel, (query_group_size // 2) * qk_panel_cols),
                   pl.mul(gemv_query, qk_panel_cols)), gemv_position,
        )
        gemv_tmp = pl.tile.create([1, (query_group_size // 2) * score_tile], dtype=pl.INT32)
        gemv_rows = pl.tile.gather(
            pl.tile.reshape(single_transport, [1, (query_group_size // 2) * score_tile]), gemv_indices, gemv_tmp,
        )
        single_scores = pl.tile.reshape(gemv_rows, [query_group_size // 2, score_tile])
"""
    source = edits(source, [(load.lineno - 1, acknowledge.end_lineno, textwrap.indent(block, " " * load.col_offset))])
    # Add the existing producer dispatch choice as a constexpr to this helper.
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn.name)
    lines = source.splitlines(keepends=True)
    signature_end = next(i for i in range(fn.lineno - 1, fn.body[0].lineno - 1) if lines[i].strip() == "):")
    lines.insert(signature_end, "    SPLIT_SCORE_ROWS: pl.constexpr = False,\n")
    result = "".join(lines)
    ast.parse(result)
    return result


def forward_flag(source, callee):
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == callee]
    assert len(calls) == 1
    call = calls[0]
    lines = source.splitlines(keepends=True)
    closing = call.end_lineno - 1
    assert lines[closing].strip() == ")"
    indent = len(lines[closing]) - len(lines[closing].lstrip()) + 4
    lines.insert(closing, " " * indent + "SPLIT_SCORE_ROWS=SPLIT_SCORE_ROWS,\n")
    result = "".join(lines)
    ast.parse(result)
    return result


def layout_check():
    # Every lane's 3072 outputs select exactly its original query/token order.
    indices = [((i >> 6) & 15) * 192 + (i >> 10) * 64 + (i & 63) for i in range(3072)]
    assert sorted(indices) == list(range(3072))
    checked = 0
    for worker in range(24):
        for slot in range(2):
            for lane in range(2):
                rows = [
                    ((worker * 2 + slot) * 6 + lane * 3 + query) * 1024 + panel * 64 + token
                    for panel in range(16)
                    for query in range(3)
                    for token in range(64)
                ]
                expected = list(
                    range(((worker * 2 + slot) * 6 + lane * 3) * 1024, ((worker * 2 + slot) * 6 + lane * 3 + 3) * 1024)
                )
                assert [rows[i] for i in indices] == expected
                checked += len(expected)
    return dict(checked_elements=checked, workers=24, fifo_slots=2, lanes=2, bijective=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    args = parser.parse_args()
    proof = json.loads(args.probe.read_text())
    assert proof["passed"] and not proof["cpu_only"] and len(proof["cases"]) == 24
    assert all(row["bitwise"] and row["guards"] and row["inputs_unchanged"] for row in proof["cases"])
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    source["indexer_score_stages.py"] = vector_transform(cube_transform(source["indexer_score_stages.py"]))
    vector = source["vector_schedule.py"]
    ending = "    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,\n):"
    assert vector.count(ending) == 1
    vector = vector.replace(ending, ending.replace("\n):", "\n    SPLIT_SCORE_ROWS: pl.constexpr = False,\n):"))
    source["vector_schedule.py"] = forward_flag(vector, "indexer_score_vector_lane_stage")
    source["entry.py"] = forward_flag(source["entry.py"], "vector_schedule")
    for text in source.values():
        ast.parse(text)
    name = "score_gemv_pool_v1"
    target = here / "runs" / name
    target.mkdir(exist_ok=False)
    for filename, text in source.items():
        (target / filename).write_text(text)
    shutil.copy2(here / "descriptors/C04_output_rectangle_v1.json", here / f"descriptors/C04_{name}.json")
    report = dict(
        source=str(target),
        parent="output_rectangle_production_v1.json",
        adopted=False,
        device_execution=False,
        probe=str(args.probe.resolve()),
        layout_check=layout_check(),
        mechanism="Six GEMVs share Acc columns and one FIX store; AIV permutes panel-major payload after ACK",
        unchanged="QK/FP16 FIX/ReLU, reduction operands, scale/TopK, roots, allocation sizes and FIFO slots",
        scope="SPLIT_SCORE_ROWS/balanced/G6/N64/1024 and runtime buf_live>3; <=3 and closed paths retained",
        tradeoff="Extra AIV unpack and 24KiB Acc pool; resource, exactness and complete-L1 gates remain",
        files={filename: hashlib.sha256(text.encode()).hexdigest() for filename, text in source.items()},
    )
    with (here / f"{name}.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
