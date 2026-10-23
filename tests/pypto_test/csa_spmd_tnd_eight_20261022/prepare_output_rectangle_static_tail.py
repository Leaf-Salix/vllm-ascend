# SPDX-License-Identifier: Apache-2.0
"""Express the NZ column tail as separate static N192 and N64 tile calls."""

import ast
import copy
import hashlib
import json
from pathlib import Path


class RowTile(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "PROJ_A_ROW_TILE":
            node.id = "A_ROW_TILE"
        return node


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_v2.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    name = "output_projection_a_stages.py"
    original = ast.parse((here / "runs/physical_capacity_production_v1" / name).read_text())
    old = next(
        fn for fn in original.body if isinstance(fn, ast.FunctionDef) and fn.name == "o_projection_a_groups_stage_nz"
    )
    old_loop = next(n for n in old.body if isinstance(n, ast.For))
    start = next(
        i for i, n in enumerate(old_loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "xa_first"
    )
    text = source[name]
    fn = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "_rectangle_o_projection_a_groups_stage_nz"
    )
    tile = copy.deepcopy(fn)
    tile.name = "_rectangle_o_projection_a_tile_nz"
    tile.decorator_list = []
    scalars = [
        ast.arg(arg=n, annotation=ast.parse("pl.Scalar[pl.INDEX]", mode="eval").body)
        for n in ("g", "pa_src0", "pa_r0", "pa_rows", "out_col_g", "n0")
    ]
    constants = [copy.deepcopy(next(arg for arg in fn.args.args if arg.arg == n)) for n in ("A_ROW_TILE", "A_COL_TILE")]
    tile.args.args = copy.deepcopy(fn.args.args[:3]) + scalars + constants
    tile.body = [RowTile().visit(copy.deepcopy(n)) for n in old_loop.body[start:]] + ast.parse("return o_r_pad").body
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    stop = next(
        i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "xa_first"
    )
    arguments = "o_packed, wo_a, o_r_pad, g, pa_src0, pa_r0, pa_rows, out_col_g, n0, A_ROW_TILE"
    loop.body = (
        loop.body[:stop]
        + ast.parse(
            f"if nf < 5:\n    o_r_pad = {tile.name}({arguments}, 192)\n"
            f"else:\n    o_r_pad = {tile.name}({arguments}, 64)\n"
        ).body
    )
    fn.decorator_list = []
    lines = text.splitlines(keepends=True)
    lines[fn.lineno - 1 : fn.end_lineno] = [
        ast.unparse(ast.fix_missing_locations(tile))
        + "\n\n@pl.jit.inline(auto_scope=False)\n"
        + ast.unparse(ast.fix_missing_locations(fn))
        + "\n"
    ]
    source[name] = "".join(lines)
    target = here / "runs/output_rectangle_v3"
    target.mkdir(exist_ok=False)
    for filename, body in source.items():
        ast.parse(body)
        (target / filename).write_text(body)
    report = dict(
        parent,
        source=str(target.resolve()),
        parent="output_rectangle_v2.json",
        predecessor_failure="NZ lowering requires static last-dimension extent; dynamic valid N is unsupported",
        correction="The five full column tiles use N192; the final tile uses static N64; original K256 body shared",
        files={filename: hashlib.sha256(body.encode()).hexdigest() for filename, body in source.items()},
    )
    with (here / "output_rectangle_v3.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
