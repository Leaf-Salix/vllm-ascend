# SPDX-License-Identifier: Apache-2.0
"""移植已验证的NZ Q_B M64/N512/K128，在多task的Main AIC中独立验证。"""

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_query_contiguous import add_arg, replace_function


def function(text, name):
    return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)


def append_call(fn, callee, parameter):
    calls = [
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == callee
    ]
    assert len(calls) == 1, (fn.name, callee, len(calls))
    calls[0].args.append(ast.Name(id=parameter, ctx=ast.Load()))


class WideConstants(ast.NodeTransformer):
    names = {
        "QPROJ_PIPE_M_TILE": "QB_WIDE_M_TILE",
        "QPROJ_PIPE_K_TILE": "QB_WIDE_K_TILE",
        "QPROJ_MM_N_TILE": "QB_WIDE_N_TILE",
        "QPROJ_N_BLOCKS": "QB_WIDE_N_BLOCKS",
    }

    def visit_Name(self, node):
        node.id = self.names.get(node.id, node.id)
        return node


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "query_projection_stages.py"
    text = path.read_text()
    old = function(text, "_q_b_nz_stage")
    wide = WideConstants().visit(copy.deepcopy(old))
    wide.name = "_q_b_nz_wide_stage"
    wide.body[0] = ast.Expr(value=ast.Constant("NZ Q_B M64/N512/K128；INT32归约不溢出，列唯一归属。"))
    add_arg(old, "QB_WIDE_N")
    old.body[1:-1] = [
        ast.If(
            test=ast.Name(id="QB_WIDE_N", ctx=ast.Load()),
            body=ast.parse(
                "q_proj_i32 = _q_b_nz_wide_stage(wq_b, qr_i8_matmul, q_proj_i32, tokens, worker_id, WORKERS)"
            ).body,
            orelse=old.body[1:-1],
        )
    ]
    text = replace_function(text, old)
    old_nd = function(text, "_q_b_nd_stage")
    add_arg(old_nd, "QB_WIDE_N")
    text = replace_function(text, old_nd)
    marker = "@pl.jit.inline(auto_scope=False)\ndef _q_b_nz_stage("
    assert text.count(marker) == 1
    text = text.replace(marker, ast.unparse(ast.fix_missing_locations(wide)) + "\n\n\n" + marker)
    marker = "QPROJ_PIPE_K_TILE = 256\n"
    assert text.count(marker) == 1
    text = text.replace(
        marker,
        marker
        + "QB_WIDE_M_TILE = 64\nQB_WIDE_N_TILE = 512\nQB_WIDE_K_TILE = 128\n"
        "QB_WIDE_N_BLOCKS = H * HEAD_DIM // QB_WIDE_N_TILE\n",
    )
    path.write_text(text)
    path = target / "front_pipelines.py"
    text = path.read_text()
    for name, callee in (("main_front_cube", "q_b_stage"), ("main_front_chain", "main_front_cube")):
        fn = function(text, name)
        add_arg(fn, "QB_WIDE_N")
        append_call(fn, callee, "QB_WIDE_N")
        text = replace_function(text, fn)
    path.write_text(text)
    path = target / "entry.py"
    text = path.read_text()
    fn = function(text, "decode_csa_tnd")
    add_arg(fn, "QB_WIDE_N", default=True)
    append_call(fn, "main_front_chain", "QB_WIDE_N")
    append_call(fn, "q_b_stage", "QB_WIDE_N")
    path.write_text(replace_function(text, fn))
    path = target / "policy.py"
    text = path.read_text()
    needle = "    prefix_workers: int = 24\n"
    assert text.count(needle) == 1
    text = text.replace(needle, needle + "    qb_wide_n: bool = False\n")
    needle = '            "PREFIX_WORKERS": self.prefix_workers,\n'
    assert text.count(needle) == 1
    text = text.replace(needle, needle + '            "QB_WIDE_N": self.qb_wide_n,\n')
    needle = "    return CSAMPMDPolicy(\n"
    assert text.count(needle) == 1
    text = text.replace(
        needle,
        needle
        + "        # 一个M64内的长G6域；不改变真实query或有效行数。\n"
        "        qb_wide_n=long_history and group == 6 and 32 <= tokens <= 64,\n",
    )
    path.write_text(text)
    print(target)


if __name__ == "__main__":
    main()
