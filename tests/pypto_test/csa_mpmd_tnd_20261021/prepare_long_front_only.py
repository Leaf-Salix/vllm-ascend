# SPDX-License-Identifier: Apache-2.0
"""短历史恢复原Query分组；长历史继续已验证的Query分组与Main重排组合。"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "front_pipelines.py"
    tree = ast.parse(path.read_text())
    vector = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main_front_vector")
    branch = next(n for n in vector.body if isinstance(n, ast.If) and ast.unparse(n.test) == "REORDER_MAIN")
    changed = 0
    for node in ast.walk(ast.Module(body=branch.orelse, type_ignores=[])):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "front_q_dequant_rope_stage"
        ):
            node.func.id = "q_dequant_rope_stage"
            changed += 1
    assert changed == 1
    path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    print(target)


if __name__ == "__main__":
    main()
