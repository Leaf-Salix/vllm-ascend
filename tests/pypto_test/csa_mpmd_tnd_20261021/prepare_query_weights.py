# SPDX-License-Identifier: Apache-2.0
"""将head weights投影/规约纳入局部query MIX，保留原数学与多task外层。"""

import ast
import copy
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from prepare_priority_fusion import scope_name


def statements(text):
    return ast.parse(textwrap.dedent(text)).body


def insert_after_phase(function, phase, body):
    sites = [
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "publish_phase"
        and ast.unparse(node.value.args[1]) == phase
    ]
    assert len(sites) == 1, (function.name, phase)
    index = sites[0] + 1
    function.body[index:index] = statements(body)


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    entry_path = target / "entry.py"
    source = entry_path.read_text()
    entry = ast.parse(source)
    root = next(node for node in entry.body if isinstance(node, ast.FunctionDef) and node.name == "decode_csa_tnd")
    root_annotations = {arg.arg: arg.annotation for arg in root.args.args}
    scopes = {scope_name(node): node for node in ast.walk(root) if scope_name(node)}
    assert len(scopes) == 36
    query_path = target / "query_group.py"
    query = ast.parse(query_path.read_text())
    functions = {node.name: node for node in query.body if isinstance(node, ast.FunctionDef)}
    cube = functions["query_chain_cube"]
    vector = functions["query_chain_vector"]
    group = functions["indexer_query_chain"]

    def parameters(function, added, outputs):
        params = {arg.arg: copy.deepcopy(arg) for arg in function.args.args}
        constexpr = params.pop("QUERY_GROUP", None)
        for param in added:
            params[param] = ast.arg(arg=param, annotation=copy.deepcopy(root_annotations[param]))
        names = [param for param in params if param not in outputs] + outputs
        for param, arg in params.items():
            annotation = copy.deepcopy(root_annotations[param])
            is_inout = isinstance(annotation, ast.Subscript) and ast.unparse(annotation.value) == "pl.InOut"
            if param not in outputs and is_inout:
                annotation = annotation.slice
            if param in outputs:
                assert is_inout, param
            arg.annotation = annotation
        function.args.args = [params[param] for param in names] + ([constexpr] if constexpr else [])
        assert isinstance(function.body[-1], ast.Return)
        function.body[-1] = ast.Return(
            value=ast.Tuple(elts=[ast.Name(id=x, ctx=ast.Load()) for x in outputs], ctx=ast.Load())
        )
        return names + (["QUERY_GROUP"] if constexpr else [])

    cube_outputs = ["idx_query_acc", "idx_query_had", "weights_partial", "query_signal"]
    vector_outputs = [
        "idx_query_rope",
        "head_weights",
        "idx_query_i8",
        "idx_query_scale",
        "coefficients",
        "query_signal",
    ]
    group_outputs = [
        "idx_query_acc",
        "idx_query_rope",
        "idx_query_had",
        "weights_partial",
        "head_weights",
        "idx_query_i8",
        "idx_query_scale",
        "coefficients",
        "query_signal",
    ]
    cube_args = parameters(cube, ["normed", "weights_proj", "weights_partial"], cube_outputs)
    vector_args = parameters(vector, ["weights_partial"], vector_outputs)
    group_args = parameters(group, ["normed", "weights_proj", "weights_partial"], group_outputs)
    insert_after_phase(
        cube,
        "PROJECT",
        """
        weights_partial = indexer_weights_project_stage(normed, weights_proj, weights_partial, worker, GROUPS)
        query_signal = publish_phase(query_signal, WEIGHTS_PROJECT, worker, CORES, generation)
    """,
    )
    insert_after_phase(
        vector,
        "ROPE",
        """
        weight_units = WEIGHTS_OK * ((tokens + MM_ROW_TILE - 1) // MM_ROW_TILE)
        query_signal = wait_row_owners(
            query_signal, WEIGHTS_PROJECT, CORES, generation, 0, weight_units, 1, GROUPS, 0, 1
        )
        head_weights = indexer_weights_reduce_stage(weights_partial, head_weights, tokens, worker, VECTORS)
    """,
    )
    group.body = statements(f"""
        with pl.cluster(name_hint="indexer_query_chain"):
            {", ".join(cube_outputs)} = query_chain_cube({", ".join(cube_args)})
            {", ".join(vector_outputs)} = query_chain_vector({", ".join(vector_args)})
        return {", ".join(group_outputs)}
    """)
    for node in query.body:
        if isinstance(node, ast.ImportFrom) and node.module == "runtime_abi":
            node.names.append(ast.alias(name="WEIGHTS_PARTIAL_ROWS"))
        if isinstance(node, ast.ImportFrom) and node.module == "indexer_query_stages":
            node.names.extend(
                ast.alias(name=name)
                for name in (
                    "MM_ROW_TILE",
                    "WEIGHTS_OK",
                    "indexer_weights_project_stage",
                    "indexer_weights_reduce_stage",
                )
            )
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "QUERY_SIGNAL_COUNT":
            node.value = ast.parse("6 * CORES * 128", mode="eval").body
    first_function = next(index for index, node in enumerate(query.body) if isinstance(node, ast.FunctionDef))
    query.body.insert(first_function, ast.parse("WEIGHTS_PROJECT = 5").body[0])
    query.body[0].value.value += """

新增WEIGHTS_PROJECT阶段：AIC先发布query投影，再在AIV做RoPE时计算head weights四份partial。
AIV发布RoPE后等待真实partial owner，再按原左折叠规约。QUANT通知也覆盖本核head weights写回；
head weights的owner集合由ceil(T/16)行块确定，是min(T,48)个quant owner的子集，故系数在
等待QUANT后才能读取完整weights。所有参与者仍推进同一代际，信号工作区改为六个阶段槽。
"""
    query_path.write_text(
        "# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(ast.fix_missing_locations(query)) + "\n"
    )

    query_scope = scopes["indexer_query_chain"]
    query_scope.body = statements(f"{', '.join(group_outputs)} = indexer_query_chain({', '.join(group_args)})")
    lines = source.splitlines(keepends=True)
    edited = [scopes["indexer_weights_proj"], scopes["indexer_weights_reduce"], query_scope]
    for node in sorted(edited, key=lambda value: value.lineno, reverse=True):
        text = (
            textwrap.indent(ast.unparse(ast.fix_missing_locations(node)), "    ") + "\n" if node is query_scope else ""
        )
        lines[node.lineno - 1 : node.end_lineno] = [text]
    result = "".join(lines)
    result = result.replace("    weight_workers = GROUPS\n", "")
    result = result.replace(
        "        weight_workers = pl.min(GROUPS, WEIGHTS_OK * ((tokens + MM_ROW_TILE - 1) // MM_ROW_TILE))\n", ""
    )
    result = result.replace(
        "# 仅 Score 与 Attention 保留各自的 MIX Cube/双 Vector 流水。",
        "# Query、Score与Attention各有局部MIX；完整CSA保留独立task调度。",
    )
    entry_path.write_text(result)
    assert sum(scope_name(node) is not None for node in ast.walk(ast.parse(result))) == 34
    print(target)


if __name__ == "__main__":
    main()
