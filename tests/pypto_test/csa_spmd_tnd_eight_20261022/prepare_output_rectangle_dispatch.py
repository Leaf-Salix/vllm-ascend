# SPDX-License-Identifier: Apache-2.0
"""Separate bounded-N and original weight-view types behind a root-Tensor dispatch."""

import ast
import copy
import hashlib
import json
from pathlib import Path


class RectangleBody(ast.NodeTransformer):
    def visit_If(self, node):
        if ast.unparse(node.test) == "A_COL_TILE == 192":
            return [self.visit(child) for child in node.body]
        return self.generic_visit(node)


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    original = ast.parse((here / "runs/physical_capacity_production_v1/output_projection_a_stages.py").read_text())
    text = source["output_projection_a_stages.py"]
    lines = text.splitlines(keepends=True)
    names = {"o_projection_a_stage_nd", "o_projection_a_groups_stage_nz"}
    functions = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in reversed(functions):
        rectangle = RectangleBody().visit(copy.deepcopy(node))
        rectangle.name = "_rectangle_" + node.name
        rectangle.decorator_list = []
        old = copy.deepcopy(
            next(fn for fn in original.body if isinstance(fn, ast.FunctionDef) and fn.name == node.name)
        )
        old.name = "_original_" + node.name
        position = next(i for i, arg in enumerate(old.args.args) if arg.arg == "A_COL_TILE") + 1
        old.args.args.insert(
            position, ast.arg(arg="A_ROW_TILE", annotation=ast.parse("pl.constexpr", mode="eval").body)
        )
        old.decorator_list = []
        dispatch = copy.deepcopy(node)
        dispatch.decorator_list = []
        arguments = ", ".join(arg.arg for arg in node.args.args)
        dispatch.body = ast.parse(
            f"if A_COL_TILE == 192:\n    o_r_pad = {rectangle.name}({arguments})\n"
            f"else:\n    o_r_pad = {old.name}({arguments})\nreturn o_r_pad\n"
        ).body
        emitted = [ast.unparse(ast.fix_missing_locations(fn)) for fn in (rectangle, old, dispatch)]
        replacement = ("\n\n@pl.jit.inline(auto_scope=False)\n".join(emitted)) + "\n"
        lines[node.lineno - 1 : node.end_lineno] = [replacement]
    source["output_projection_a_stages.py"] = "".join(lines)
    target = here / "runs/output_rectangle_v2"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    report = dict(
        parent,
        source=str(target.resolve()),
        parent="output_rectangle_v1.json",
        predecessor_failure="Both constant-branch weight views are checked before pruning "
        "and have incompatible valid-shape types",
        correction="Separate math helpers return the same root output type; "
        "closed domain calls original helper unchanged",
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
    )
    with (here / "output_rectangle_v2.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
