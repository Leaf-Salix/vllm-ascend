# SPDX-License-Identifier: Apache-2.0
"""使两条前段MIX在QR完成后再选择空闲核，保留sync_start及核内实现。"""

import ast
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from prepare_priority_fusion import scope_name


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "entry.py"
    source = path.read_text()
    root = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "decode_csa_tnd")
    gate = next(n for n in root.body if isinstance(n, ast.If) and ast.unparse(n.test) == "FUSED_DAG")
    scope = next(n for n in gate.body if scope_name(n) == "qproj_a_rms_quant")
    keyword = next(k for k in scope.items[0].context_expr.keywords if k.arg == "allow_early_resolve")
    assert keyword.value.value is True
    keyword.value = ast.Constant(False)
    lines = source.splitlines(keepends=True)
    lines[scope.lineno - 1 : scope.end_lineno] = [
        textwrap.indent(ast.unparse(ast.fix_missing_locations(scope)), "        ") + "\n"
    ]
    path.write_text("".join(lines))
    print(path)


if __name__ == "__main__":
    main()
