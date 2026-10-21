# SPDX-License-Identifier: Apache-2.0
"""在两个M32行组区间让QA/KV填满24 AIC，并裁掉KV RMS空worker。"""

import ast
import copy
import shutil
import subprocess
import sys
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
    root = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "decode_csa_tnd")
    gate = next(n for n in root.body if isinstance(n, ast.If) and ast.unparse(n.test) == "FUSED_DAG")
    scopes = {scope_name(n): n for n in gate.body if scope_name(n)}
    early_kv = copy.deepcopy(scopes["kv_proj"])
    call = early_kv.items[0].context_expr
    call.keywords = [k for k in call.keywords if k.arg != "deps"]
    call.args[0] = ast.parse(
        "(KV_HEAD_DIM // KV_N_TILE) * pl.min(KV_OM, pl.max(1, tokens // KV_FIXED_SMALL_M_TILE))",
        mode="eval",
    ).body
    rebuilt = []
    for node in gate.body:
        label = scope_name(node)
        if label == "hc_gate_mix_norm":
            ready = copy.deepcopy(node)
            set_keyword(ready, "allow_early_resolve", ast.Constant(False))
            rebuilt.append(ast.If(test=ast.Name(id="KV_QA_OVERLAP", ctx=ast.Load()), body=[ready], orelse=[node]))
        elif label == "qproj_a":
            rebuilt.append(ast.If(test=ast.Name(id="KV_QA_OVERLAP", ctx=ast.Load()), body=[early_kv], orelse=[]))
            rebuilt.append(node)
        elif label == "kv_proj":
            rebuilt.append(
                ast.If(
                    test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="KV_QA_OVERLAP", ctx=ast.Load())),
                    body=[node],
                    orelse=[],
                )
            )
        else:
            if label == "kv_norm_rope":
                node.items[0].context_expr.args[0] = ast.parse(
                    "pl.min((tokens + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE, VECTORS)", mode="eval"
                ).body
            rebuilt.append(node)
    gate.body = rebuilt
    root.args.args.append(ast.arg(arg="KV_QA_OVERLAP", annotation=ast.parse("pl.constexpr", mode="eval").body))
    root.args.defaults.append(ast.Constant(False))
    start = min([root.lineno] + [n.lineno for n in root.decorator_list]) - 1
    lines = source.splitlines(keepends=True)
    lines[start : root.end_lineno] = [ast.unparse(ast.fix_missing_locations(root)) + "\n"]
    source = "".join(lines).replace(
        "import pypto.language as pl\n",
        "import pypto.language as pl\nfrom .kv_stages import KV_FIXED_SMALL_M_TILE, KV_RMS_T_TILE\n",
        1,
    )
    path.write_text(source)
    policy_path = target / "policy.py"
    policy = policy_path.read_text()
    policy = policy.replace(
        "\n\n@dataclass",
        "\n\n# QA每M32行组8核、KV每组4核；两个行组恰好占24 AIC。\n"
        "QA_KV_ROW_TILE = 32\nQA_KV_OVERLAP_MIN_TOKENS = 2 * QA_KV_ROW_TILE\n"
        "QA_KV_OVERLAP_END_TOKENS = 3 * QA_KV_ROW_TILE\n\n\n@dataclass",
        1,
    )
    needle = "    fused_dag: bool = True\n"
    assert policy.count(needle) == 1
    policy = policy.replace(needle, needle + "    kv_qa_overlap: bool = False\n")
    needle = '            "FUSED_DAG": self.fused_dag,\n'
    policy = policy.replace(needle, needle + '            "KV_QA_OVERLAP": self.kv_qa_overlap,\n')
    needle = "    return CSAMPMDPolicy(\n"
    policy = policy.replace(
        needle, needle + "        kv_qa_overlap=QA_KV_OVERLAP_MIN_TOKENS <= tokens < QA_KV_OVERLAP_END_TOKENS,\n", 1
    )
    policy_path.write_text(policy)
    print(target)


if __name__ == "__main__":
    main()
