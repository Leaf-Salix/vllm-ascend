# SPDX-License-Identifier: Apache-2.0
"""Indexer完整M8块改为8 head一组，并将固定RoPE gather索引外提到worker级。"""

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path


class WiderHeads(ast.NodeTransformer):
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id == "DQ_ROPE_H_TILE":
            return ast.copy_location(ast.Constant(8), node)
        return node


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "indexer_query_stages.py"
    source = path.read_text()
    original = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "indexer_query_dequant_rope_stage"
    )
    fn = WiderHeads().visit(copy.deepcopy(original))
    fn.name = "indexer_query_dequant_rope_wide_stage"
    worker_guard = next(n for n in fn.body if isinstance(n, ast.If))
    outer_index = next(i for i, n in enumerate(worker_guard.body) if isinstance(n, ast.For))
    outer = worker_guard.body[outer_index]
    branch = next(n for n in outer.body if isinstance(n, ast.If))
    head_index = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.For))
    # 前三项只是当前八行的scale/cos/sin；后面直到head循环的索引完全不依赖输入。
    setup = branch.body[3:head_index]
    assert setup and not any(
        isinstance(n, ast.Name) and n.id in ("dq_t0", "hg", "qr_scale", "cos", "sin")
        for statement in setup
        for n in ast.walk(statement)
    )
    # 旧浮点swap只为尾行服务；挪入尾分支，完整M8不再计算一份不用的索引。
    tail_setup = worker_guard.body[1:outer_index]
    branch.orelse = tail_setup + branch.orelse
    branch.body = branch.body[:3] + branch.body[head_index:]
    worker_guard.body = worker_guard.body[:1] + setup + [outer]
    old_guard = next(n for n in original.body if isinstance(n, ast.If))
    old_outer = next(n for n in old_guard.body if isinstance(n, ast.For))
    old_branch = next(n for n in old_outer.body if isinstance(n, ast.If))
    old_head = next(n for n in old_branch.body if isinstance(n, ast.For))
    new_head = next(n for n in branch.body if isinstance(n, ast.For))
    assert [ast.dump(n, include_attributes=False) for n in old_head.body] == [
        ast.dump(n, include_attributes=False) for n in new_head.body
    ]
    path.write_text(source + "\n\n" + ast.unparse(ast.fix_missing_locations(fn)) + "\n")
    path = target / "front_pipelines.py"
    tree = ast.parse(path.read_text())
    vector = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "index_front_vector")
    for i, node in enumerate(vector.body):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "indexer_query_dequant_rope_stage"
        ):
            wide = copy.deepcopy(node)
            wide.value.func.id = "indexer_query_dequant_rope_wide_stage"
            vector.body[i] = ast.If(
                test=ast.parse("tokens >= 2 * DEQUANT_T_TILE and tokens % DEQUANT_T_TILE == 0", mode="eval").body,
                body=[wide],
                orelse=[node],
            )
            break
    else:
        raise ValueError("未找到Indexer RoPE调用")
    tree.body.insert(
        1,
        ast.ImportFrom(
            module="indexer_query_stages", level=1, names=[ast.alias(name="indexer_query_dequant_rope_wide_stage")]
        ),
    )
    # 当前16个Indexer AIV下，两条分支均有>=16个单位，原Cube等待集合仍完整且恰好覆盖。
    path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    print(target)


if __name__ == "__main__":
    main()
