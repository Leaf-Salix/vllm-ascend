# SPDX-License-Identifier: Apache-2.0
"""反哺冻结SPMD的M256 query Hadamard，并同步调整真实producer等待集合。"""

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    reference = here / "sources/84751e34/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    module = ast.parse((reference / "projection_overlap_stages.py").read_text())
    helper = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_query_hadamard_four_tokens")
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "query_group.py"
    tree = ast.parse(path.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = [
        n
        for n in ast.walk(functions["query_chain_cube"])
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "indexer_query_hadamard_stage"
    ]
    assert len(calls) == 1
    calls[0].func = ast.Name(id=helper.name, ctx=ast.Load())
    waits = [
        n
        for n in ast.walk(functions["query_chain_vector"])
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "wait_row_owners" and ast.unparse(n.args[1]) == "HADAMARD"
    ]
    assert len(waits) == 1 and ast.unparse(waits[0].args[5]) == "tokens"
    waits[0].args[5] = ast.parse("(tokens + 3) // 4", mode="eval").body
    first = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.FunctionDef))
    tree.body.insert(first, copy.deepcopy(helper))
    tree.body[0].value.value += (
        "\nM256 Hadamard直接复用冻结84751e34的四query helper；仍按每行完整K128计算，"
        "真实尾行由valid_shape约束。producer单位改为ceil(T/4)，等待集合一起更新。"
        "只影响融合query路径；小容量独立task保持原数学和分工。\n"
    )
    path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    print(path)


if __name__ == "__main__":
    main()
