# SPDX-License-Identifier: Apache-2.0
"""小档O-A的M对齐实际行数/K512预取候选；仍保留N128和原K方向顺序。

只适用于整档T<=96。M按16行对齐；更大档继续保留原M128/K256。
这是测试目录的可选变体，CPU通过后还必须检查实际K内层及整层exact。
"""

import ast
import copy
import json
from pathlib import Path


def use_k512(schedule, catalog, output):
    tokens = schedule["root_arguments"]["x_hc"]["shape"][0]
    rows = (tokens + 15) // 16 * 16
    if not 1 <= tokens <= 96:
        raise ValueError("O-A K512候选要求T<=96，较大档保留原分块")
    module = ast.parse(Path(catalog["logical_source_ir"]).read_text())
    functions = {node.name: node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)}
    selected = {task["kernel_names"][0] for task in schedule["tasks"]
                if task["kernel_names"][0] in {"proj_a_mm", "proj_a_mm_0", "proj_a_mm_1"}}
    if not selected:
        raise ValueError("当前档位没有原N128 O-A函数")
    audit = []
    for name in sorted(selected):
        fn = functions[name]
        before = copy.deepcopy(fn)
        changes = {"activation_shape": 0, "activation_valid_k": 0, "weight_shape": 0,
                   "acc_shape": 0, "k_offset": 0, "pipeline_extent": 0}

        class Retile(ast.NodeTransformer):
            def visit_List(self, node):
                node = self.generic_visit(node)
                values = [x.value if isinstance(x, ast.Constant) else None for x in node.elts]
                if values == [128, 256]:
                    node.elts[0].value, node.elts[1].value = rows, 512
                    changes["activation_shape"] += 1
                elif values == [1, 256, 128]:
                    node.elts[1].value = 512
                    changes["weight_shape"] += 1
                elif values == [1, 128, 128]:
                    node.elts[1].value = rows
                    changes["acc_shape"] += 1
                elif (len(values) == 2 and values[1] == 256
                      and isinstance(node.elts[0], ast.Name)
                      and node.elts[0].id.startswith("pa_rows_")):
                    node.elts[1].value = 512
                    changes["activation_valid_k"] += 1
                return node

            def visit_AnnAssign(self, node):
                node = self.generic_visit(node)
                if isinstance(node.target, ast.Name) and node.target.id.startswith("k0_"):
                    if not isinstance(node.value, ast.BinOp) or ast.unparse(node.value.right) != "256":
                        raise ValueError("O-A原K偏移表达式改变")
                    node.value.right.value = 512
                    changes["k_offset"] += 1
                return node

            def visit_Call(self, node):
                node = self.generic_visit(node)
                if ast.unparse(node.func) == "pl.pipeline":
                    if [ast.unparse(x) for x in node.args] != ["1", "16"]:
                        raise ValueError("O-A原外层K循环改变")
                    node.args[1].value = 8
                    changes["pipeline_extent"] += 1
                return node

        Retile().visit(fn)
        expected = {"activation_shape": 4, "activation_valid_k": 4, "weight_shape": 4,
                    "acc_shape": 3, "k_offset": 1, "pipeline_extent": 1}
        if changes != expected:
            raise ValueError(f"O-A变换位置不符：{name}: {changes}")
        # 变换只触及所列形状/循环；所有参数、N索引、行索引及全局目标保持原样。
        if ast.dump(fn.args) != ast.dump(before.args) or ast.dump(fn.returns) != ast.dump(before.returns):
            raise ValueError("O-A根参数描述符意外改变")
        audit.append({"kernel": name, "tokens": tokens, "m_tile": rows, "n_tile": 128,
                      "k_prefetch_tile": 512, "original_k_prefetch_tile": 256,
                      "outer_k_blocks": 8, "expected_inner_k_tile": 128,
                      "changes": changes, "logical_members_unchanged": True,
                      "global_views_unchanged": True, "device_validated": False})
    ast.fix_missing_locations(module)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / "logical_oproj_k512.py"
    destination.write_text("# Copyright (c) PyPTO Contributors.\n"
                           "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                           + ast.unparse(module) + "\n")
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["logical_source_ir"] = str(destination)
    return result, audit
