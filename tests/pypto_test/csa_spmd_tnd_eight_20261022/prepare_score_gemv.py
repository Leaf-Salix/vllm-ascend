# SPDX-License-Identifier: Apache-2.0
"""Replace only the full-G6 diagonal WS with independent true M1 GEMVs."""

import ast
import hashlib
import json
import shutil
import textwrap
from pathlib import Path


def transform(source):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store")
    assignments = {
        n.targets[0].id: n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
    }
    changes = []

    def change(node, replacement=""):
        changes.append((node.lineno - 1, node.end_lineno, textwrap.indent(replacement, " " * node.col_offset)))

    def replace(name, replacement=""):
        change(assignments[name], replacement)

    def coefficient_rows(first):
        return "".join(
            f"gemv_coef_mat{q} = pl.load(coefficients, [buf_group * NATIVE_QLI_WEIGHT_ROWS + {q}, {q} * IDX_N_HEADS], "
            "[1, 256], valid_shape=[1, IDX_N_HEADS], target_memory=pl.MemorySpace.Mat)\n"
            f"gemv_coef{q} = pl.tile.move(gemv_coef_mat{q}, target_memory=pl.MemorySpace.Left)\n"
            for q in range(first, first + 3)
        )

    def reduce_rows(suffix, column):
        return "".join(
            f"gemv_source{q}{suffix} = pl.set_validshape(split_previous{q // 3}, "
            f"{q % 3 + 1} * IDX_N_HEADS, qk_panel_cols)\n"
            f"gemv_right{q}{suffix} = pl.tile.extract(gemv_source{q}{suffix}, {q % 3} * IDX_N_HEADS, 0, "
            "[256, qk_panel_cols], target_memory=pl.MemorySpace.Right)\n"
            f"gemv_result{q}{suffix} = pl.tile.gemv_acc(split_ws_acc, gemv_coef{q}, "
            f"gemv_right{q}{suffix}, init_cond=True)\n"
            f"pl.store(gemv_result{q}{suffix}, [buf_transfer_row + {q}, {column}], buf_score_transfer)\n"
            for q in range(6)
        )

    replace("buf_coefficients_l1")
    replace("split_coefficient0", coefficient_rows(0))
    replace("split_coefficient1", coefficient_rows(3))
    replace(
        "split_ws_acc",
        "split_ws_acc = pl.tile.create([16, qk_panel_cols], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)\n"
        "split_ws_acc = pl.set_validshape(split_ws_acc, 1, qk_panel_cols)\n",
    )
    replace("split_right0")
    replace("split_right1")
    replace("split_ws_first", reduce_rows("", "buf_panel_col - qk_panel_cols"))
    replace("split_ws_done")
    replace("split_right0_last")
    replace("split_right1_last")
    replace("split_ws_first_last", reduce_rows("_last", "score_tile - qk_panel_cols"))
    replace("split_ws_done_last")
    # Keep physical K256 throughout GEMV. The L1 parent includes the largest
    # physical window (128 + 256 rows); only its first 192 rows are produced,
    # and every Right load has valid K64 wholly inside those produced rows.
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "pl.tile.create"
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("split_previous0", "split_previous1", "split_current0", "split_current1")
        ):
            change(
                node,
                f"{node.targets[0].id} = pl.tile.create([384, qk_panel_cols], "
                "dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)\n",
            )
    stores = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.store"
        and any(name in ast.unparse(n.value.args[0]) for name in ("split_ws_done", "split_ws_done_last"))
    ]
    assert len(stores) == 2
    for node in stores:
        change(node)
    change(
        fn.body[0],
        '"""Experimental full-G6 M1 WS; original QK, <=3 live path and FIFO protocol.\n'
        "\nSix nonzero K64 reductions replace the diagonal K192+192 products.\n"
        "One shared physical-M16 Acc stores each row independently. Full CSA\n"
        'bitwise and complete-L1 checks remain mandatory.\n"""\n',
    )
    lines = source.splitlines(keepends=True)
    ordered = sorted(changes)
    assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))
    for start, end, replacement in reversed(ordered):
        lines[start:end] = [replacement] if replacement else []
    result = "".join(lines)
    after = ast.parse(result)
    original = ast.parse(source)

    def filtered(t):
        return ast.dump(
            ast.Module(
                body=[n for n in t.body if not isinstance(n, ast.FunctionDef) or n.name != fn.name], type_ignores=[]
            ),
            include_attributes=False,
        )

    assert filtered(original) == filtered(after), "Only the selected Cube helper may change"
    old_half = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("buf_live <="))
    new_fn = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == fn.name)
    new_half = next(
        n for n in ast.walk(new_fn) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("buf_live <=")
    )
    assert ast.dump(ast.Module(body=old_half.body, type_ignores=[])) == ast.dump(
        ast.Module(body=new_half.body, type_ignores=[])
    )
    return result


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    source["indexer_score_stages.py"] = transform(source["indexer_score_stages.py"])
    target = here / "runs/score_gemv_v3"
    target.mkdir(exist_ok=False)
    for name, value in source.items():
        (target / name).write_text(value)
    shutil.copy2(here / "descriptors/C04_output_rectangle_v1.json", here / "descriptors/C04_score_gemv_v3.json")
    report = dict(
        source=str(target),
        parent="output_rectangle_production_v1.json",
        adopted=False,
        device_execution=False,
        mechanism="Six true M1/K64 GEMVs replace full-G6 diagonal M16/K192+192 WS; six independent row stores",
        numerical_probe="score_gemv_probe_results.json",
        physical_layout="Coefficients load as physical K256/valid K64 to satisfy A3 512B; "
        "384-row Mat parents support all physical K256 Right windows; each consumes valid K64 only",
        unchanged="QK/FP16 FIX/ReLU, score scale, page gathers, AIV/TopK, GM allocation/layout, "
        "ABI and ready/consumed protocol",
        scope="Existing full-G6 split helper only; <=3 live path identical. C04 pilot, no production adoption",
        tradeoff="Reduces zero products but increases MMA/MTE1/FIX issue count; full-L1 benefit unproven",
        files={name: hashlib.sha256(value.encode()).hexdigest() for name, value in source.items()},
    )
    with (here / "score_gemv_v3.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
