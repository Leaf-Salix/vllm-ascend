# SPDX-License-Identifier: Apache-2.0
"""MPMD Score的少组分工按叶轮换组起点，避免24核条带固定承接同类query。"""

import ast
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "indexer_group_stages.py"
    source = path.read_text()
    fn = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "indexer_work_item"
    )
    nodes = [
        node for node in ast.walk(fn) if isinstance(node, ast.Assign) and ast.unparse(node) == "group = item % groups"
    ]
    assert len(nodes) == 1
    nodes[0].value = ast.parse("(item % groups + item // groups) % groups", mode="eval").body
    fn.body[0].value.value = """少组时每叶轮换组起点，多组保留原分工；空输入不除零。

对固定leaf，group=(ordinal+leaf)%groups是全体组的循环置换，因此每个
(group,leaf)仍只执行一次。只改变worker归属和执行顺序，pair_arena仍按
原query/leaf写入，最终Top-K仍按原叶序合并。AIC和两个AIV共用此映射。

这里仅用于MPMD：后继由TMR等待整个Score task，不依赖单SPMD的原owner
通知公式。不能单独把这个helper搬回单SPMD而不同时修正其producer等待。
"""
    lines = source.splitlines(keepends=True)
    start = min([fn.lineno, *(node.lineno for node in fn.decorator_list)])
    lines[start - 1 : fn.end_lineno] = [textwrap.dedent(ast.unparse(ast.fix_missing_locations(fn))) + "\n"]
    path.write_text("".join(lines))
    print(path)


if __name__ == "__main__":
    main()
