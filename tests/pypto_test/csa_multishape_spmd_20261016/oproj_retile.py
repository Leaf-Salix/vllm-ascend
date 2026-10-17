# SPDX-License-Identifier: Apache-2.0
"""输出投影列块候选：复用冻结原算子已有的 N256 数学分支。

只改 O-A 列划分与相应逻辑分片数，K256归约及其顺序保持；不改原动态对照。
每档重新分核，输出和缓存仍必须通过原 exact 门槛，不能从数学等价推断浮点逐位相等。
"""

import ast
import copy
import json
import re
from pathlib import Path


def stable_name(name):
    return re.sub(r"_inline\d+", "", name)


class StableNames(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=stable_name(node.id), ctx=node.ctx), node)


class ExpectedN256(ast.NodeTransformer):
    """逐位置定义允许的列块变化；行轴、K归约、dtype和cache policy不动。"""

    def visit_AnnAssign(self, node):
        node = self.generic_visit(node)
        name = stable_name(node.target.id) if isinstance(node.target, ast.Name) else ""
        if name.startswith(("pa_rb__", "nf__")):
            if not isinstance(node.value, ast.BinOp) or not isinstance(node.value.right, ast.Constant) or node.value.right.value != 8:
                raise ValueError("原O-A列分片公式改变")
            node.value.right.value = 4
        if name.startswith("n0__"):
            if not isinstance(node.value, ast.BinOp) or not isinstance(node.value.right, ast.Constant) or node.value.right.value != 128:
                raise ValueError("原O-A列偏移公式改变")
            node.value.right.value = 256
        return node

    def visit_List(self, node):
        node = self.generic_visit(node)
        # 仅两个3维局部形状含列轴：权重[1,K,N]、累加器[1,M,N]。
        if len(node.elts) == 3 and all(isinstance(x, ast.Constant) for x in node.elts):
            values = [x.value for x in node.elts]
            if values in ([1, 256, 128], [1, 128, 128]):
                node.elts[-1].value = 256
        return node


def use_n256(schedule, catalog, output):
    path = Path(catalog["logical_source_ir"])
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef))
    functions = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    target = functions["proj_a_mm_2"]
    selected = [t for t in schedule["tasks"]
                if t["kernel_names"][0] in {"proj_a_mm", "proj_a_mm_0", "proj_a_mm_1"}]
    if not selected:
        raise ValueError("本规格没有待替换的N128输出投影")
    audit = []
    changed = set()
    for task in selected:
        name = task["kernel_names"][0]
        original = functions[name]
        if name not in changed:
            expected = ExpectedN256().visit(copy.deepcopy(ast.Module(body=original.body, type_ignores=[])))
            actual = copy.deepcopy(ast.Module(body=target.body, type_ignores=[]))
            if ast.dump(StableNames().visit(copy.deepcopy(expected))) != ast.dump(
                    StableNames().visit(copy.deepcopy(actual))):
                raise ValueError("N256原分支还包含列划分以外的数学变化：" + name)
            old_params = {stable_name(p.arg): p.arg for p in original.args.args}
            new_params = {stable_name(p.arg): p.arg for p in target.args.args}
            if old_params.keys() != new_params.keys():
                raise ValueError("原N128/N256的入参不一致")
            mapping = {new_params[key]: old_params[key] for key in old_params}

            class BindOriginalParameters(ast.NodeTransformer):
                def visit_Name(self, node):
                    return ast.copy_location(ast.Name(id=mapping.get(node.id, node.id), ctx=node.ctx), node)

            original.body = BindOriginalParameters().visit(actual).body
            names = list(ast.walk(ast.Module(body=original.body, type_ignores=[])))
            loads = {n.id for n in names if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            stores = {n.id for n in names if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            if loads - stores - {p.arg for p in original.args.args} - {"pl"}:
                raise ValueError("N256核体存在未重新绑定的变量")
            changed.add(name)
        call = schedule["call_bindings"][task["task_id"]]
        if not any(isinstance(a["value"], dict) and a["value"].get("layout") == "NZ"
                   for a in call["arguments"]):
            raise ValueError("当前候选只用于读取逻辑NZ核体的生成路径")
        tokens = call["arguments"][0]["value"]
        before = ((tokens + 127) // 128) * 8
        after = ((tokens + 127) // 128) * 4
        if task["block_num"] != before or call["block_num"] != before:
            raise ValueError("原列分片数量不符合N128/行128的约定")
        task.update(block_num=after, member_count=after)
        call["block_num"] = after
        audit.append({"task": task["task_id"], "kernel": name,
                      "original_blocks": before, "n256_blocks": after,
                      "row_tile": 128, "k_tile": 256, "k_order_unchanged": True,
                      "full_output_column_coverage": [0, 1024],
                      "source_branch": "proj_a_mm_2"})
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    updated = output / "logical_n256.py"
    updated.write_text("# Copyright (c) PyPTO Contributors.\n"
                       "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                       "# 从冻结CSA逻辑IR复用N256列分支，详见同目录audit.json。\n"
                       + ast.unparse(module) + "\n")
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    updated_catalog = copy.deepcopy(catalog)
    updated_catalog["logical_source_ir"] = str(updated)
    return updated_catalog, audit
