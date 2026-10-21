# SPDX-License-Identifier: Apache-2.0
"""移植已设备验证的连续head反量化，独立constexpr限定长G6资源域。"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

from query_contiguous_source import transform


def add_arg(fn, name, default=False):
    fn.args.args.append(ast.arg(arg=name, annotation=ast.parse("pl.constexpr", mode="eval").body))
    if default:
        fn.args.defaults.append(ast.Constant(False))


def replace_function(source, fn):
    start = min([fn.lineno] + [n.lineno for n in fn.decorator_list]) - 1
    lines = source.splitlines(keepends=True)
    lines[start : fn.end_lineno] = [ast.unparse(ast.fix_missing_locations(fn)) + "\n"]
    return "".join(lines)


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "query_projection_stages.py"
    source = transform(path.read_text(), revision=4)
    wrapper = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "q_dequant_rope_stage"
    )
    add_arg(wrapper, "CONTIGUOUS_HEADS", default=True)
    wrapper.body[0].test = ast.BoolOp(
        op=ast.And(), values=[ast.Name(id="CONTIGUOUS_HEADS", ctx=ast.Load()), wrapper.body[0].test]
    )
    path.write_text(replace_function(source, wrapper))
    path = target / "front_pipelines.py"
    source = path.read_text()
    for fn_name, callee in (("main_front_vector", "q_dequant_rope_stage"), ("main_front_chain", "main_front_vector")):
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
        add_arg(fn, "QUERY_CONTIGUOUS_HEADS")
        calls = [
            n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == callee
        ]
        assert len(calls) == 1
        calls[0].args.append(ast.Name(id="QUERY_CONTIGUOUS_HEADS", ctx=ast.Load()))
        source = replace_function(source, fn)
    path.write_text(source)
    path = target / "entry.py"
    source = path.read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "decode_csa_tnd")
    add_arg(fn, "QUERY_CONTIGUOUS_HEADS", default=True)
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "main_front_chain"
    ]
    assert len(calls) == 1
    calls[0].args.append(ast.Name(id="QUERY_CONTIGUOUS_HEADS", ctx=ast.Load()))
    # JIT依赖调用不读取constexpr的签名默认值；旧独立task显式关闭该新策略。
    for call in ast.walk(fn):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "q_dequant_rope_stage":
            call.args.append(ast.Constant(False))
    path.write_text(replace_function(source, fn))
    path = target / "policy.py"
    source = path.read_text()
    needle = "    output_work_balance: bool = False\n"
    assert source.count(needle) == 1
    source = source.replace(needle, needle + "    query_contiguous_heads: bool = False\n")
    needle = '            "OUTPUT_WORK_BALANCE": self.output_work_balance,\n'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '            "QUERY_CONTIGUOUS_HEADS": self.query_contiguous_heads,\n')
    needle = "    return CSAMPMDPolicy(\n"
    assert source.count(needle) == 1
    source = source.replace(
        needle, needle + "        query_contiguous_heads=long_history and group == 6 and 64 <= tokens <= 96,\n"
    )
    path.write_text(source)
    for filename in ("entry.py", "front_pipelines.py", "query_projection_stages.py", "policy.py"):
        ast.parse((target / filename).read_text())
    print(target)


if __name__ == "__main__":
    main()
