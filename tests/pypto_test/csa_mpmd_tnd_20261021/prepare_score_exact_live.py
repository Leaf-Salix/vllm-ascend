# SPDX-License-Identifier: Apache-2.0
"""长G6 Score按真实live分支使用固定M64/128/192，保留两半流水和六行FIFO。"""

import ast
import copy
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


class HalfRows(ast.NodeTransformer):
    def __init__(self, rows):
        self.rows = rows

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id == "SPLIT_QK_ROWS":
            return ast.copy_location(ast.Constant(self.rows), node)
        return node


class SecondRows(ast.NodeTransformer):
    """只缩小第二半Tile尺寸，GM/L1中的第二半偏移仍然是192。"""

    names = {
        "split_query1",
        "split_coefficient1",
        "split_qk_acc1",
        "split_previous1",
        "split_current1",
        "split_right1",
        "split_right1_last",
    }

    def __init__(self, rows):
        self.rows = rows
        self.changed = set()

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in self.names and isinstance(node.value, ast.Call):
                for arg in node.value.args:
                    if isinstance(arg, ast.List):
                        for i, item in enumerate(arg.elts):
                            if isinstance(item, ast.Name) and item.id == "SPLIT_QK_ROWS":
                                arg.elts[i] = ast.Constant(self.rows)
                                self.changed.add(name)
        return self.generic_visit(node)


def isolate(body, parameters, label):
    module = ast.Module(body=body, type_ignores=[])
    locals_ = {
        n.id
        for n in ast.walk(module)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id not in parameters
    }
    for node in ast.walk(module):
        if isinstance(node, ast.Name) and node.id in locals_:
            node.id = label + node.id
    return module.body


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "indexer_score_stages.py"
    source = path.read_text()
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store"
    )
    valid = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "work_valid > 0")
    original = valid.body[0]
    assert isinstance(original, ast.If) and ast.unparse(original.test) == "buf_live <= SPLIT_QK_ROWS // IDX_N_HEADS"
    parameters = {a.arg for a in fn.args.args}
    cases = {}
    for live in range(1, 7):
        template = original.body if live <= 3 else original.orelse
        module = ast.Module(body=copy.deepcopy(template), type_ignores=[])
        if live <= 3:
            module = HalfRows(live * 64).visit(module)
        elif live < 6:
            transformer = SecondRows((live - 3) * 64)
            module = transformer.visit(module)
            assert transformer.changed == transformer.names
        # 阶段FIFO仍固定六行，tile ACK、发布次数和排空循环没有裁剪。
        for call in ("pl.store", "pl.system.sync_set", "pl.system.sync_wait"):
            before = sum(isinstance(n, ast.Call) and ast.unparse(n.func) == call for s in template for n in ast.walk(s))
            after = sum(isinstance(n, ast.Call) and ast.unparse(n.func) == call for n in ast.walk(module))
            assert before == after
        cases[live] = isolate(module.body, parameters, f"live{live}_")
    branch = cases[6]
    for live in range(5, 0, -1):
        branch = [ast.If(test=ast.parse(f"buf_live == {live}", mode="eval").body, body=cases[live], orelse=branch)]
    valid.body = branch
    lines = source.splitlines(keepends=True)
    replacement = textwrap.indent(ast.unparse(ast.fix_missing_locations(valid)), " " * valid.col_offset) + "\n"
    lines[valid.lineno - 1 : valid.end_lineno] = [replacement]
    text = "".join(lines)
    text = text.replace(
        "# live<=3时系数矩阵后半严格为零，不读取下一请求，也不改变六行分数FIFO。\n"
        "        # 只计算固定M192的前半QK及对应WS，避免触碰已知动态Acc行距限制。\n"
        "        # live>=4完整保留原双半链；跳过加零是否逐bit等价仍需设备数值验证。",
        "# 各live分支使用固定物理M，避免动态Acc行距问题；只跳过零系数对应的head。\n"
        "        # 六行FIFO、两半矩阵乘累加次序、ACK和排空保持原协议。",
    )
    ast.parse(text)
    path.write_text(text)
    print(path)


if __name__ == "__main__":
    main()
