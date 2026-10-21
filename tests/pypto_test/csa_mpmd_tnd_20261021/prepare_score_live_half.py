# SPDX-License-Identifier: Apache-2.0
"""长G6 Score对实际query<=3的组跳过全零后半计算；不改变叶计划和FIFO。"""

import ast
import copy
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


class FirstHalfOnly(ast.NodeTransformer):
    second_half = {
        "split_query1",
        "split_coefficient1",
        "split_qk_acc1",
        "split_qk1",
        "split_previous1",
        "split_current1",
        "split_right1",
        "split_right1_last",
        "split_ws_done",
        "split_ws_done_last",
    }

    def __init__(self):
        self.removed = []

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in self.second_half:
                self.removed.append(node.targets[0].id)
                return None
        return self.generic_visit(node)

    def visit_Name(self, node):
        replacement = {"split_ws_done": "split_ws_first", "split_ws_done_last": "split_ws_first_last"}.get(node.id)
        return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node) if replacement else node

    def visit_BinOp(self, node):
        if ast.unparse(node) == "query_group_size * IDX_N_HEADS":
            return ast.copy_location(ast.Name(id="SPLIT_QK_ROWS", ctx=ast.Load()), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        if ast.unparse(node.func) == "pl.unroll" and len(node.args) == 1:
            if ast.unparse(node.args[0]) == "query_group_size":
                node.args[0] = ast.parse("SPLIT_QK_ROWS // IDX_N_HEADS", mode="eval").body
        return self.generic_visit(node)


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "indexer_score_stages.py"
    source = path.read_text()
    tree = ast.parse(source)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_indexer_score_cube_split_store"
    )
    work = [node for node in ast.walk(fn) if isinstance(node, ast.If) and ast.unparse(node.test) == "work_valid > 0"]
    assert len(work) == 1
    valid = work[0]
    full = copy.deepcopy(valid.body)
    transformer = FirstHalfOnly()
    half_module = transformer.visit(ast.Module(body=copy.deepcopy(full), type_ignores=[]))
    assert set(transformer.removed) == transformer.second_half
    half = half_module.body

    # 所有信号/排空仍逐字保持同一调用，store仍写六行；只裁去恒零的QK/WS后半。
    def calls(body, name):
        return [
            ast.dump(node)
            for node in ast.walk(ast.Module(body=body, type_ignores=[]))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == name
        ]

    for name in ("pl.system.sync_set", "pl.system.sync_wait"):
        assert calls(full, name) == calls(half, name), name
    assert len(calls(full, "pl.store")) == len(calls(half, "pl.store")) == 2
    assert not any(isinstance(node, ast.Name) and node.id in transformer.second_half for node in ast.walk(half_module))
    # 两分支的Q Tile物理M不同，局部名隔离，避免前端构造无后继用途的异形phi。
    parameters = {arg.arg for arg in fn.args.args}
    locals_to_rename = {
        node.id
        for node in ast.walk(half_module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id not in parameters
    }
    for node in ast.walk(half_module):
        if isinstance(node, ast.Name) and node.id in locals_to_rename:
            node.id = "live_half_" + node.id
    valid.body = [
        ast.If(test=ast.parse("buf_live <= SPLIT_QK_ROWS // IDX_N_HEADS", mode="eval").body, body=half, orelse=full)
    ]
    assert ast.dump(ast.Module(body=valid.body[0].orelse, type_ignores=[])) == ast.dump(
        ast.Module(body=full, type_ignores=[])
    )
    lines = source.splitlines(keepends=True)
    comment = """# live<=3时系数矩阵后半严格为零，不读取下一请求，也不改变六行分数FIFO。
# 只计算固定M192的前半QK及对应WS，避免触碰已知动态Acc行距限制。
# live>=4完整保留原双半链；跳过加零是否逐bit等价仍需设备数值验证。
"""
    replacement = (
        textwrap.indent(comment + ast.unparse(ast.fix_missing_locations(valid)), " " * valid.col_offset) + "\n"
    )
    lines[valid.lineno - 1 : valid.end_lineno] = [replacement]
    path.write_text("".join(lines))
    print(path)


if __name__ == "__main__":
    main()
