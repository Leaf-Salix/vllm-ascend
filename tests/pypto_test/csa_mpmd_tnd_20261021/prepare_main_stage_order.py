# SPDX-License-Identifier: Apache-2.0
"""Main先投影/池化压缩链，再处理KV；保持全部数学调用和阶段边。"""

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path


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
    path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    print(target)


if __name__ == "__main__":
    main()
