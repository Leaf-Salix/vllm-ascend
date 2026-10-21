# SPDX-License-Identifier: Apache-2.0
"""按已有 token 容量选择完整多 task 编排，不在结果中手选实验。"""

import ast
import copy
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


def decode_function(module):
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "decode_csa_tnd")


def main():
    here = Path(__file__).resolve().parent
    name, large_name, small_name = sys.argv[1:4]
    large = here / "experiments" / large_name / "operator"
    small = here / "experiments" / small_name / "operator"
    large_source = (large / "entry.py").read_text()
    small_source = (small / "entry.py").read_text()
    large_module, small_module = ast.parse(large_source), ast.parse(small_source)
    large_root, small_root = decode_function(large_module), decode_function(small_module)
    large_functions = {n.name: n for n in large_module.body if isinstance(n, ast.FunctionDef)}
    for node in small_module.body:
        if isinstance(node, ast.FunctionDef) and node.name != small_root.name:
            assert ast.dump(node) == ast.dump(large_functions[node.name]), node.name
    small_args = {arg.arg for arg in small_root.args.args}
    large_args = {arg.arg for arg in large_root.args.args}
    assert large_args - small_args == {"query_signal"} and not small_args - large_args
    assert isinstance(large_root.body[-1], ast.Return) and isinstance(small_root.body[-1], ast.Return)
    large_return, small_return = large_root.body[-1], small_root.body[-1]
    assert [ast.unparse(n) for n in large_return.value.elts[:-1]] == [ast.unparse(n) for n in small_return.value.elts]
    assert ast.unparse(large_return.value.elts[-1]) == "query_signal"
    common_count = 0
    for left, right in zip(large_root.body[:-1], small_root.body[:-1]):
        if ast.dump(left) != ast.dump(right):
            break
        common_count += 1
    # 公共Tensor视图只定义一次，避免前端在两个constexpr分支中重复统一符号shape。
    assert common_count >= 14

    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(large, target, dirs_exist_ok=True)
    gate = ast.If(
        test=ast.Name(id="FUSED_DAG", ctx=ast.Load()),
        body=copy.deepcopy(large_root.body[common_count:-1]),
        orelse=copy.deepcopy(small_root.body[common_count:-1]),
    )
    lines = large_source.splitlines(keepends=True)
    lines[large_root.body[common_count].lineno - 1 : large_root.body[-1].end_lineno] = [
        textwrap.indent(ast.unparse(ast.fix_missing_locations(gate)), "    ") + "\n",
        textwrap.indent(ast.unparse(large_return), "    ") + "\n",
    ]
    source = "".join(lines)
    needle = "    LIMIT_EMPTY_WORKERS: pl.constexpr = False,\n"
    assert source.count(needle) == 1
    source = source.replace(needle, needle + "    FUSED_DAG: pl.constexpr = True,\n")
    # 只补较小编排需要而较大编排未导入的符号，拒绝同名不同来源。
    imported = {}
    for node in large_module.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported[alias.asname or alias.name] = (node.level, node.module, alias.name)
    extra_imports = []
    for node in small_module.body:
        if isinstance(node, ast.ImportFrom):
            missing = []
            for alias in node.names:
                key, origin = alias.asname or alias.name, (node.level, node.module, alias.name)
                if key in imported:
                    assert imported[key] == origin, key
                else:
                    missing.append(copy.deepcopy(alias))
                    imported[key] = origin
            if missing:
                extra_imports.append(ast.ImportFrom(module=node.module, names=missing, level=node.level))
    source = source.replace(
        "import pypto.language as pl\n",
        "import pypto.language as pl\n" + "\n".join(ast.unparse(node) for node in extra_imports) + "\n",
        1,
    )
    (target / "entry.py").write_text(source)

    policy_path = target / "policy.py"
    policy = policy_path.read_text()
    empty_field = "    limit_empty_workers: bool = False\n"
    empty_constexpr = '            "LIMIT_EMPTY_WORKERS": self.limit_empty_workers,\n'
    replacements = {
        "NONEMPTY_WORKER_MAX_TOKENS = 64\n": "NONEMPTY_WORKER_MAX_TOKENS = 64\n"
        "# 两个M32容量起使用局部融合编排；小容量保留已测更快的独立阶段。\n"
        "FUSED_DAG_MIN_TOKENS = 64\n",
        empty_field: empty_field + "    fused_dag: bool = True\n",
        empty_constexpr: empty_constexpr + '            "FUSED_DAG": self.fused_dag,\n',
        "    return CSAMPMDPolicy(\n": "    return CSAMPMDPolicy(\n        fused_dag=tokens >= FUSED_DAG_MIN_TOKENS,\n",
    }
    for before, after in replacements.items():
        assert policy.count(before) == 1, before
        policy = policy.replace(before, after)
    policy_path.write_text(policy)
    binding_path = target / "binding.py"
    binding = binding_path.read_text()
    needle = '"query_signal": ((QUERY_SIGNAL_COUNT,), torch.int32),'
    assert binding.count(needle) == 1
    binding_path.write_text(
        binding.replace(needle, '"query_signal": ((QUERY_SIGNAL_COUNT if self.policy.fused_dag else 1,), torch.int32),')
    )
    print(target)


if __name__ == "__main__":
    main()
