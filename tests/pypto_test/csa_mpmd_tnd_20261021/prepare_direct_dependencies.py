# SPDX-License-Identifier: Apache-2.0
"""把已捕获的真实 WAIT 图等价写成 TaskId 边，准备私有手动依赖候选。

仅适用于首版固定的 task 划分；不从计时数据猜测生产者，不修改核内数学。
在提交队列前使用，完整图与映射另存供审阅。正式采用仍须设备验证。
"""

import ast
import json
import sys
from pathlib import Path


def main():
    experiment = Path(sys.argv[1]).resolve()
    evidence = Path(__file__).resolve().parent / "runs/matrix_v8_h131072_b16/l1_00_mpmd"
    graph = json.loads((evidence / "deps.json").read_text())
    names = json.loads((evidence / "names.json").read_text())["callable_id_to_name"]
    path = experiment / "operator/entry.py"
    tree = ast.parse(path.read_text())
    root = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode_csa_tnd")
    scopes = {}
    for node in ast.walk(root):
        if isinstance(node, ast.With):
            call = node.items[0].context_expr
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "spmd":
                hint = next(ast.literal_eval(k.value) for k in call.keywords if k.arg == "name_hint")
                scopes[hint] = node
    task_names = {}
    for task in graph["tasks"]:
        first_name = names[str(next(k for k in task["kernel_ids"] if k >= 0))]
        hint = {"index_score_topk_cube": "index_score_topk", "sparse_attn_cube": "sparse_attn_csa"}.get(
            first_name, first_name
        )
        if hint not in scopes:
            raise ValueError(f"捕获图与当前 task 划分不符：{hint}")
        task_names[task["task_id"]] = hint
    if len(task_names) != len(scopes) or len(set(task_names.values())) != len(scopes):
        raise ValueError("捕获图与源级 task 必须一一对应")
    parents = {hint: set() for hint in scopes}
    for edge in graph["edges"]:
        if "wait" in edge["flags"]:
            parents[task_names[edge["succ"]]].add(task_names[edge["pred"]])
        elif edge["flags"] != ["retain"]:
            raise ValueError(f"未处理的依赖语义：{edge}")
    ids = {}
    for hint, node in scopes.items():
        assigned = node.items[0].optional_vars
        ids[hint] = assigned.id if isinstance(assigned, ast.Name) else "dep_" + hint
        node.items[0].optional_vars = ast.Name(id=ids[hint], ctx=ast.Store())
    for hint, node in scopes.items():
        deps = [
            ast.parse("geometry_deps[0]" if parent == "csa_query_geometry" else ids[parent], mode="eval").body
            for parent in sorted(parents[hint])
        ]
        call = node.items[0].context_expr
        call.keywords = [k for k in call.keywords if k.arg != "deps"]
        call.keywords.append(ast.keyword(arg="deps", value=ast.List(elts=deps, ctx=ast.Load())))
    first = root.body.index(scopes["hc_widen_rms"])
    last = root.body.index(scopes["hc_post"])
    body = []
    for statement in root.body[first : last + 1]:
        body.append(statement)
        if statement is scopes["csa_row_offsets"]:
            body.extend(
                ast.parse(
                    f"geometry_deps = pl.array.create(1, pl.TASK_ID)\ngeometry_deps[0] = {ids['csa_row_offsets']}"
                ).body
            )
        if isinstance(statement, ast.If) and scopes["csa_query_geometry"] in statement.body:
            statement.body.extend(ast.parse(f"geometry_deps[0] = {ids['csa_query_geometry']}").body)
    manual = ast.With(items=[ast.withitem(context_expr=ast.parse("pl.manual_scope()", mode="eval").body)], body=body)
    root.body[first : last + 1] = [manual]
    ast.fix_missing_locations(tree)
    path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")
    (experiment / "direct_dependencies.json").write_text(
        json.dumps(
            {"source": str(evidence / "deps.json"), "tasks": {k: sorted(v) for k, v in parents.items()}},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"{len(scopes)} task，{sum(map(len, parents.values()))} 条去重 WAIT 边，核内 helper 未修改")


if __name__ == "__main__":
    main()
