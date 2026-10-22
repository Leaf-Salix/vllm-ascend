# SPDX-License-Identifier: Apache-2.0
"""宽N候选的输出调度消融：等待原前驱就绪再派发，保留sync_start。"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_query_contiguous import replace_function


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "entry.py"
    text = path.read_text()
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "decode_csa_tnd")
    count = 0
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "spmd":
            continue
        keywords = {k.arg: k for k in node.keywords}
        hint = keywords.get("name_hint")
        if hint is not None and isinstance(hint.value, ast.Constant) and hint.value.value == "output_group_chain":
            assert keywords["allow_early_resolve"].value.value is True
            assert keywords["sync_start"].value.value is True
            keywords["allow_early_resolve"].value = ast.Constant(False)
            count += 1
    assert count == 3, count
    path.write_text(replace_function(text, fn))
    print(target)


if __name__ == "__main__":
    main()
