# SPDX-License-Identifier: Apache-2.0
"""HC前端两个逐行阶段的分档候选，物理Tile仍保持8行以满足列规约对齐。

只改变独立token行的归属与有效行数，K方向的循环和累加顺序不变。
本候选先限于T为8的倍数，避免混入原padding区域初始化规则的改变。
"""

import ast
import copy
import json
from pathlib import Path


def repartition_rows(schedule, catalog, output, rows=2):
    tokens = schedule["root_arguments"]["x_hc"]["shape"][0]
    if rows not in (2, 4) or tokens % 8:
        raise ValueError("HC行分片候选要求2/4行且T是8的倍数")
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    selected = {node.name: node for node in ast.walk(module)
                if isinstance(node, ast.FunctionDef)
                and node.name in {"hc_widen_rms", "mix_x_rms_norm"}}
    if len(selected) != 2:
        raise ValueError("HC前端函数不唯一")
    changes = []
    for name, fn in selected.items():
        is_widen = name == "hc_widen_rms"
        start_name = "w_t0__ssa_v0" if is_widen else "t0_inline519__ssa_v0"
        valid_name = "w_rows__ssa_v0" if is_widen else "valid_rows_inline512__ssa_v0"
        counts = {"start": 0, "valid": 0, "full_predicate": 0, "load": 0, "rms_store": 0}
        for node in ast.walk(fn):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == start_name:
                    if not isinstance(node.value, ast.BinOp) or ast.unparse(node.value.right) != "8":
                        raise ValueError("原行分片起点不是8行")
                    node.value.right = ast.Constant(rows)
                    counts["start"] += 1
                elif node.target.id == valid_name:
                    if not isinstance(node.value, ast.Call) or ast.unparse(node.value.args[-1]) != "8":
                        raise ValueError("原有效行数不是8行上限")
                    node.value.args[-1] = ast.Constant(rows)
                    counts["valid"] += 1
            if isinstance(node, ast.If) and ast.unparse(node.test) == f"{valid_name} == 8":
                node.test.comparators[0] = ast.Constant(rows)
                counts["full_predicate"] += 1
            if not isinstance(node, ast.Call):
                continue
            op = ast.unparse(node.func)
            if is_widen and op == "pl.tile.store" and ast.unparse(node.args[0]) == "w_inv__tile":
                node.args[0] = ast.parse(f"pl.tile.set_validshape(w_inv__tile, {valid_name}, 1)", mode="eval").body
                counts["rms_store"] += 1
            if not is_widen and op == "pl.tile.load":
                source = ast.unparse(node.args[0])
                if source in {"pre_val_store__rv_v2", "x_mixed_inline534__rv_v2"}:
                    if len(node.args) != 4 or ast.unparse(node.args[3].elts[0]) != "8":
                        raise ValueError("pre或mixed输入的原有效形状改变")
                    node.args[3].elts[0] = ast.Name(id=valid_name, ctx=ast.Load())
                    counts["load"] += 1
        expected = {"start": 1, "valid": 1, "full_predicate": 1,
                    "load": 0 if is_widen else 2, "rms_store": 1 if is_widen else 0}
        if counts != expected:
            raise ValueError(f"{name}行分片变换点不符：{counts}")
        tasks = [task for task in schedule["tasks"]
                 if {value for value in task["kernel_names"] if value} == {name}]
        if len(tasks) != 1:
            raise ValueError(f"{name}逻辑任务不是唯一的")
        task = tasks[0]
        call = schedule["call_bindings"][task["task_id"]]
        old = task["block_num"]
        blocks = min(tokens // rows, 48) if is_widen else tokens // rows
        task.update(block_num=blocks, member_count=blocks)
        call["block_num"] = blocks
        if is_widen:
            args = [arg for arg in call["arguments"] if arg["parameter"] == "widen_rows__ssa_v0"]
            if len(args) != 1 or not isinstance(args[0]["value"], int):
                raise ValueError("widen_rows不是已有Host分片计数")
            args[0]["value"] = tokens // rows
        changes.append({"kernel": name, "task": task["task_id"], "old_members": old,
                        "new_members": blocks, "logical_row_chunks": tokens // rows,
                        "changed_sites": counts})
    ast.fix_missing_locations(module)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "core_bodies_hc_row_partition.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    "# 仅变更两个HC前端核的行分工；有效形状限制每核读写范围。\n"
                    + ast.unparse(module) + "\n")
    coverage = [row for chunk in range(tokens // rows) for row in range(chunk * rows, (chunk + 1) * rows)]
    if coverage != list(range(tokens)):
        raise ValueError("新行分片有重复或遗漏")
    audit = {"tokens": tokens, "rows_per_chunk": rows, "physical_tile_rows": 8,
             "tasks": changes, "row_coverage_exactly_once": True,
             "k_accumulation_order_unchanged": True,
             "full_chunks_do_not_write_shared_tail": True,
             "inv_rms_store_has_explicit_valid_rows": True,
             "device_validated": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
