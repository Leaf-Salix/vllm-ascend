# SPDX-License-Identifier: Apache-2.0
"""为query MIX与Compressor保留并发核容量，统一更改分工及实际等待owner。"""

import ast
import copy
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from prepare_priority_fusion import scope_name


class QueryWorkers(ast.NodeTransformer):
    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            return node
        replacements = {"GROUPS": "QUERY_WORKERS", "VECTORS": "2 * QUERY_WORKERS", "CORES": "3 * QUERY_WORKERS"}
        return ast.parse(replacements[node.id], mode="eval").body if node.id in replacements else node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in ("query_chain_cube", "query_chain_vector"):
            node.args.append(ast.Name(id="QUERY_WORKERS", ctx=ast.Load()))
        return node


class PrefixWorkers(ast.NodeTransformer):
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("idx_project_workers", "main_project_workers"):
                if isinstance(node.value, ast.Name) and node.value.id == "GROUPS":
                    node.value.id = "PREFIX_WORKERS"
                elif isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "pl.min":
                    assert ast.unparse(node.value.args[0]) == "GROUPS"
                    node.value.args[0] = ast.Name(id="PREFIX_WORKERS", ctx=ast.Load())
        return self.generic_visit(node)

    def visit_With(self, node):
        if scope_name(node) == "indexer_query_chain":
            call = node.items[0].context_expr
            assert ast.unparse(call.args[0]) == "GROUPS"
            call.args[0] = ast.Name(id="QUERY_WORKERS", ctx=ast.Load())
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in ("compressor_project_stage", "indexer_compressor_project_stage"):
                assert ast.unparse(node.args[-1]) == "GROUPS"
                node.args[-1] = ast.Name(id="PREFIX_WORKERS", ctx=ast.Load())
            elif node.func.id == "indexer_query_chain":
                node.args.append(ast.Name(id="QUERY_WORKERS", ctx=ast.Load()))
        return node


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    if workers not in (8, 12, 16):
        raise ValueError("本实验只评估24核的常见资源分区")
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    query_path = target / "query_group.py"
    query_source = query_path.read_text()
    query_module = ast.parse(query_source)
    query_lines = query_source.splitlines(keepends=True)
    functions = [node for node in query_module.body if isinstance(node, ast.FunctionDef)]
    for node in reversed(functions):
        replacement = QueryWorkers().visit(copy.deepcopy(node))
        replacement.args.args.append(
            ast.arg(arg="QUERY_WORKERS", annotation=ast.parse("pl.constexpr", mode="eval").body)
        )
        begin = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list]) - 1
        query_lines[begin : node.end_lineno] = [ast.unparse(ast.fix_missing_locations(replacement)) + "\n"]
    query_source = "".join(query_lines).replace(
        "所有24个AIC/48个AIV每次共同推进代际，即使本轮无数据也不跳过。",
        "所有QUERY_WORKERS个AIC及两倍AIV每次共同推进代际，即使无数据也不跳过。\n"
        "信号保留最大72参与者容量，实际编号/阶段步长和全部owner按本次固定核数组合计算。",
    )
    query_path.write_text(query_source)

    path = target / "entry.py"
    source = path.read_text()
    root = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "decode_csa_tnd")
    gate = next(n for n in root.body if isinstance(n, ast.If) and ast.unparse(n.test) == "FUSED_DAG")
    replacement = copy.deepcopy(gate)
    replacement.body = ast.parse("idx_project_workers = PREFIX_WORKERS").body + [
        PrefixWorkers().visit(node) for node in replacement.body
    ]
    lines = source.splitlines(keepends=True)
    lines[gate.lineno - 1 : gate.end_lineno] = [
        textwrap.indent(ast.unparse(ast.fix_missing_locations(replacement)), "    ") + "\n"
    ]
    source = "".join(lines)
    needle = "    FUSED_DAG: pl.constexpr = True,\n"
    assert source.count(needle) == 1
    source = source.replace(
        needle, needle + "    QUERY_WORKERS: pl.constexpr = GROUPS,\n    PREFIX_WORKERS: pl.constexpr = GROUPS,\n"
    )
    # 小容量分支也须提供新helper参数，但它没有局部query MIX；其余编排不变。
    path.write_text(source)
    policy_path = target / "policy.py"
    policy = policy_path.read_text()
    replacements = {
        "    fused_dag: bool = True\n": "    fused_dag: bool = True\n    prefix_workers: int = 24\n",
        '            "FUSED_DAG": self.fused_dag,\n': (
            '            "FUSED_DAG": self.fused_dag,\n'
            '            "QUERY_WORKERS": self.prefix_workers,\n'
            '            "PREFIX_WORKERS": self.prefix_workers,\n'
        ),
        "        fused_dag=tokens >= FUSED_DAG_MIN_TOKENS,\n": (
            "        fused_dag=tokens >= FUSED_DAG_MIN_TOKENS,\n"
            f"        prefix_workers={workers} if tokens >= FUSED_DAG_MIN_TOKENS else 24,\n"
        ),
    }
    for before, after in replacements.items():
        assert policy.count(before) == 1, before
        policy = policy.replace(before, after)
    policy_path.write_text(policy)
    print(target)


if __name__ == "__main__":
    main()
