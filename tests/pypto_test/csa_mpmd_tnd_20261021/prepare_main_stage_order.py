# SPDX-License-Identifier: Apache-2.0
"""Main先投影/池化压缩链，再处理KV；保持全部数学调用和阶段边。"""

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_priority_fusion import scope_name


def call_of(node):
    return node.value if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) else None


def phase_of(node):
    call = call_of(node)
    if call is not None and ast.unparse(call.func) in ("publish_phase", "wait_row_owners"):
        return ast.unparse(call.args[1])
    return None


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    balanced_only = "--balanced-only" in sys.argv[3:]
    if any(value != "--balanced-only" for value in sys.argv[3:]):
        raise ValueError("仅支持--balanced-only")
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "front_pipelines.py"
    tree = ast.parse(path.read_text())
    for label in ("main_front_cube", "main_front_vector"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == label)
        before = copy.deepcopy(fn.body)
        if label.endswith("cube"):
            moving = [
                n
                for n in fn.body
                if phase_of(n) == "M_KV_PROJECT"
                or (call_of(n) is not None and ast.unparse(call_of(n).func) == "kv_project_stage")
            ]
            assert len(moving) == 2
            kept = [n for n in fn.body if n not in moving]
            position = next(i for i, n in enumerate(kept) if phase_of(n) == "M_PROJECT") + 1
        else:
            moving = [
                n
                for n in fn.body
                if phase_of(n) in ("M_KV_PROJECT", "M_KV_NORM")
                or (
                    call_of(n) is not None
                    and ast.unparse(call_of(n).func) in ("kv_norm_rope_stage", "original_kv_writeback_stage")
                )
            ]
            assert len(moving) == 5
            kept = [n for n in fn.body if n not in moving]
            position = next(i for i, n in enumerate(kept) if phase_of(n) == "M_QUERY")
        fn.body = kept[:position] + moving + kept[position:]
        # 仅整体移动已有语句；不漏阶段发布、等待、state读写屏障或返回值。
        assert sorted(ast.dump(n, include_attributes=False) for n in before) == sorted(
            ast.dump(n, include_attributes=False) for n in fn.body
        )
        if balanced_only:
            first = (
                next(
                    i
                    for i, n in enumerate(before)
                    if call_of(n) is not None and ast.unparse(call_of(n).func) == "begin_invocation"
                )
                + 1
            )
            assert isinstance(before[-1], ast.Return)
            fn.body = (
                fn.body[:first]
                + [
                    ast.If(
                        test=ast.Name(id="REORDER_MAIN", ctx=ast.Load()),
                        body=fn.body[first:-1],
                        orelse=before[first:-1],
                    )
                ]
                + [fn.body[-1]]
            )
            fn.args.args.append(ast.arg(arg="REORDER_MAIN", annotation=ast.parse("pl.constexpr", mode="eval").body))
    if balanced_only:
        chain = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main_front_chain")
        chain.args.args.append(ast.arg(arg="REORDER_MAIN", annotation=ast.parse("pl.constexpr", mode="eval").body))
        for n in ast.walk(chain):
            if isinstance(n, ast.Call) and ast.unparse(n.func) in ("main_front_cube", "main_front_vector"):
                n.args.append(ast.Name(id="REORDER_MAIN", ctx=ast.Load()))
        entry_path = target / "entry.py"
        source = entry_path.read_text()
        root = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "decode_csa_tnd")
        gate = next(n for n in root.body if isinstance(n, ast.If) and ast.unparse(n.test) == "FUSED_DAG")
        scope = next(n for n in gate.body if scope_name(n) == "main_front_chain")
        scope.body[0].value.args.append(ast.Name(id="BALANCED", ctx=ast.Load()))
        start = min([root.lineno] + [n.lineno for n in root.decorator_list]) - 1
        lines = source.splitlines(keepends=True)
        lines[start : root.end_lineno] = [ast.unparse(ast.fix_missing_locations(root)) + "\n"]
        entry_path.write_text("".join(lines))
    path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    print(target)


if __name__ == "__main__":
    main()
