# SPDX-License-Identifier: Apache-2.0
"""核对关闭候选后的编排AST等于基线，且数学/底层信号文件逐字节不变。"""

import ast
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
CONTROL = "ATTENTION_TRANSITIVE_DEPENDENCIES"


class DisableCandidate(ast.NodeTransformer):
    def visit_If(self, node):
        if ast.unparse(node.test) == f"not {CONTROL}":
            return [self.visit(statement) for statement in node.body]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.args.args = [arg for arg in node.args.args if arg.arg != CONTROL]
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def main():
    unchanged = []
    source_files = []
    for path in sorted(PACKAGE.glob("*.py")):
        relative = str(path.relative_to(ROOT))
        original = subprocess.check_output(["git", "show", f"593c5d25:{relative}"], cwd=ROOT)
        current = path.read_bytes()
        source_files.append(dict(path=relative, size_bytes=len(current)))
        if path.name not in ("entry.py", "policy.py", "cube_schedule.py", "vector_schedule.py"):
            assert original == current, path.name
            unchanged.append(path.name)
        if path.name in ("cube_schedule.py", "vector_schedule.py"):
            baseline = DisableCandidate().visit(ast.parse(original.decode()))
            disabled = DisableCandidate().visit(ast.parse(current.decode()))
            assert ast.dump(baseline) == ast.dump(disabled), path.name
    result = dict(
        passed=True,
        baseline="593c5d25",
        disabled_schedule_ast_equal=True,
        byte_equal_math_and_sync_files=unchanged,
        source_files=source_files,
    )
    (HERE / "source_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "source_files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
