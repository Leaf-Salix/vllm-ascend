# SPDX-License-Identifier: Apache-2.0
"""在局部query候选上先启动共同前驱Q_A，再放行独立的前段投影。"""

import ast
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from prepare_priority_fusion import scope_name, set_keyword


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "entry.py"
    source = path.read_text()
    scopes = {scope_name(node): node for node in ast.walk(ast.parse(source)) if scope_name(node)}
    assert len(scopes) == 36 and "indexer_query_chain" in scopes
    qa = scopes["qproj_a"]
    assert qa.items[0].optional_vars is None
    qa.items[0].optional_vars = ast.Name(id="qa_tid", ctx=ast.Store())
    changed = [qa]
    # 仅调度优先级；这些投影仍从原normed读取，无新增数学或Host准备。
    # Q_A是Q_B和Indexer query共同前驱，比等待整个query MIX更早释放。
    for name in ("indexer_compressor_kv_score", "indexer_weights_proj", "compressor_kv_score", "kv_proj"):
        scope = scopes[name]
        assert not any(item.arg == "deps" for item in scope.items[0].context_expr.keywords)
        set_keyword(scope, "deps", ast.parse("[qa_tid]", mode="eval").body)
        changed.append(scope)
    lines = source.splitlines(keepends=True)
    for node in sorted(changed, key=lambda value: value.lineno, reverse=True):
        lines[node.lineno - 1 : node.end_lineno] = [
            textwrap.indent(ast.unparse(ast.fix_missing_locations(node)), "    ") + "\n"
        ]
    path.write_text("".join(lines))
    print(path)


if __name__ == "__main__":
    main()
