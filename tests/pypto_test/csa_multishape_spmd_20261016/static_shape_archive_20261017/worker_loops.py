# SPDX-License-Identifier: Apache-2.0
"""多波次逻辑分片复用同一核体，不改变阶段前后的通知与等待。"""

import ast


def loop_rebound_blocks(source, layout):
    """保持等待/发布位置，把一个控制位置承接的逻辑分片放进核内循环。"""
    module = ast.parse(source)
    rewritten = 0
    for role in ("aic", "aiv"):
        fn = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "csa_static_" + role
        )
        for stmt in fn.body:
            if not isinstance(stmt, ast.If) or not ast.unparse(stmt.test).startswith("mapped_block_"):
                continue
            index = int(stmt.test.left.id.removeprefix("mapped_block_"))
            members = layout["batches"][role][index]["members"]
            strides = {row["logical_block_stride"] for row in members}
            counts = {row["logical_block_num"] for row in members}
            assert len(strides) == len(counts) == 1
            stride, count = next(iter(strides)), next(iter(counts))
            if stride == count:
                continue
            # signal_lowering 生成 6 项控制语句、3 项读侧维护、原数学体、
            # 3 项写侧维护与最后的条件发布。仅对数学体加循环。
            math_body = stmt.body[9:-4]
            old_name, new_name = f"block_{index}", f"logical_block_{index}"

            class RenameBlock(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == old_name:  # noqa: B023 本轮同步执行，不保存闭包。
                        return ast.copy_location(ast.Name(id=new_name, ctx=node.ctx), node)  # noqa: B023
                    return node

            # PTOAS 当前对“外层 INDEX 循环变量直接作内层起点”的 lowering
            # 会产生 emitc.for 类型不一致。改为固定波数 + 动态 block 偏移，
            # 数学上的 block 集合与执行顺序不变；尾波使用显式范围保护。
            loop = ast.parse(
                f"for block_wave_{index} in pl.range({(count + stride - 1) // stride}):\n"
                f"    {new_name} = {old_name} + block_wave_{index} * {stride}\n"
                f"    if {new_name} < {count}:\n        pass"
            ).body[0]
            loop.body[1].body = [RenameBlock().visit(node) for node in math_body]
            stmt.body = stmt.body[:9] + [loop] + stmt.body[-4:]
            rewritten += 1
    ast.fix_missing_locations(module)
    return "# 按形状特化的 CSA 单 SPMD；新增分片在核内循环，复用核体。\n" + ast.unparse(module) + "\n", rewritten
