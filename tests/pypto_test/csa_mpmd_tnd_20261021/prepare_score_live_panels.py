# SPDX-License-Identifier: Apache-2.0
"""小live固定M为宽N释放空间；保持每列K归约及1024列FIFO，不改叶计划。"""

import ast
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


class Panel(ast.NodeTransformer):
    def __init__(self, columns, prefetch_l0):
        self.columns = columns
        self.prefetch_l0 = prefetch_l0

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id == "qk_panel_cols":
                return ast.copy_location(ast.Constant(self.columns), node)
            if node.id == "key_prefetch_to_l0":
                return ast.copy_location(ast.Constant(self.prefetch_l0), node)
        return node


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "indexer_score_stages.py"
    source = path.read_text()
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store"
    )
    work = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "work_valid > 0")
    settings = {1: (256, False), 2: (128, True), 3: (128, False)}
    for live, (columns, prefetch) in settings.items():
        branch = next(
            n for n in ast.walk(work) if isinstance(n, ast.If) and ast.unparse(n.test) == f"buf_live == {live}"
        )
        module = Panel(columns, prefetch).visit(ast.Module(body=branch.body, type_ignores=[]))
        branch.body = module.body
    # M64/N256、M192/N128的Right总驻留按64KiB边界设计，Key仅预取到Mat。
    # M128/N128保留原Key L0双缓冲。最终资源分配仍由后端与设备编译验证。
    lines = source.splitlines(keepends=True)
    lines[work.lineno - 1 : work.end_lineno] = [
        textwrap.indent(ast.unparse(ast.fix_missing_locations(work)), " " * work.col_offset) + "\n"
    ]
    text = "".join(lines)
    ast.parse(text)
    path.write_text(text)
    print(path)


if __name__ == "__main__":
    main()
