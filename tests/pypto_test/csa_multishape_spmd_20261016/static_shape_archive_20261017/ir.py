# SPDX-License-Identifier: Apache-2.0
"""读取包内数学和编排源码；只提供生成器需要的 AST 操作。"""

import ast
import collections
from pathlib import Path


def functions_from_ir(path):
    """返回唯一 program 类及其函数，仅读源码，不导入或执行 PyPTO。"""
    module = ast.parse(Path(path).read_text())
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1:
        raise ValueError(f"数学源码必须有且仅有一个 program 类：{path}")
    functions = [node for node in classes[0].body if isinstance(node, ast.FunctionDef)]
    by_name = {function.name: function for function in functions}
    if len(by_name) != len(functions):
        raise ValueError(f"数学源码包含重名函数：{path}")
    return module, by_name


class ExplicitWorker(ast.NodeTransformer):
    """把逻辑核编号显式化，不改变数学、访存、事件或流水。"""

    REPLACEMENTS = {
        "pl.tile.get_block_idx": "original_block_idx",
        "pl.tile.get_block_num": "original_block_num",
        "pl.tile.get_subblock_idx": "original_subblock_idx",
        "pl.tile.get_subblock_num": "original_subblock_num",
    }

    def __init__(self):
        self.used = collections.Counter()

    def visit_Call(self, node):
        name = ast.unparse(node.func)
        if name in self.REPLACEMENTS:
            if node.args or node.keywords:
                raise ValueError(f"不支持带参数的逻辑核编号：{ast.unparse(node)}")
            target = self.REPLACEMENTS[name]
            self.used[target] += 1
            return ast.copy_location(ast.Name(id=target, ctx=ast.Load()), node)
        return self.generic_visit(node)
