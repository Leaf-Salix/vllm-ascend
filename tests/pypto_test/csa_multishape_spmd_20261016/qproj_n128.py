# SPDX-License-Identifier: Apache-2.0
"""主Q投影N128候选，保留K256 INT8累加与20个逻辑worker。

原N256的Right是256×256 INT8，单块占满64KiB，K双缓冲无法让两块
Right同时存活。N128把Right降至32KiB、Acc降至64KiB，检查编译器
能否实际重叠MTE1与Cube。代价是列块数加倍；必须以完整CSA实测选择。
"""

import ast
import collections
import copy
import json
from pathlib import Path


def use_n128(schedule, catalog, output):
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("主Q N128先验证B16/S6，不泛化尚未验证的行块")
    tree = ast.parse(Path(catalog["logical_source_ir"]).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "qproj_matmul")
    before_args = ast.dump(fn.args)
    changes = {"round_count": 0, "column_offset": 0, "weight_shapes": 0, "acc_shapes": 0}
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    old = ast.unparse(loop.iter.args[0])
    expected = "(19 - qproj_worker_inline8514__ssa_v0 + 128) // 20"
    if old != expected:
        raise ValueError("原主Q不是128列块/20worker：" + old)
    loop.iter.args[0] = ast.parse(old.replace("+ 128", "+ 256"), mode="eval").body
    changes["round_count"] = 1
    for node in ast.walk(fn):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        if name.startswith("w_col0_"):
            if not isinstance(node.value, ast.BinOp) or ast.unparse(node.value.right) != "256":
                raise ValueError("原Q列偏移改变")
            node.value.right = ast.Constant(128)
            changes["column_offset"] += 1
        elif name.startswith(("wq_first_", "wq_chunk_")):
            for value in ast.walk(node):
                if isinstance(value, ast.List) and ast.unparse(value) == "[256, 256]":
                    value.elts[1] = ast.Constant(128)
                    changes["weight_shapes"] += 1
        elif name.startswith("col_acc_"):
            for value in ast.walk(node.annotation):
                if isinstance(value, ast.List) and ast.unparse(value) == "[128, 256]":
                    value.elts[1] = ast.Constant(128)
                    changes["acc_shapes"] += 1
    if changes != {"round_count": 1, "column_offset": 1, "weight_shapes": 4, "acc_shapes": 3}:
        raise ValueError("主Q N128修改位置不符：" + str(changes))
    if ast.dump(fn.args) != before_args:
        raise ValueError("不能改主Q根参数描述符")
    columns = collections.Counter(w + r * 20 for w in range(20)
                                  for r in range((19 - w + 256) // 20))
    if columns != collections.Counter(range(256)):
        raise ValueError("列块遗漏或重复")
    kloop = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.pipeline"]
    if len(kloop) != 1 or [ast.unparse(n) for n in kloop[0].args] != ["256", "1024", "256"]:
        raise ValueError("K累加顺序改变")
    tasks = [t for t in schedule["tasks"] if t["kernel_names"][0] == "qproj_matmul"]
    if len(tasks) != 1 or tasks[0]["block_num"] != 20:
        raise ValueError("主Q N128保持20个原逻辑worker")
    ast.fix_missing_locations(tree)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "logical_qproj_n128.py"
    source.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      + ast.unparse(tree) + "\n")
    audit = {"old_n": 256, "new_n": 128, "k_tile": 256, "workers": 20,
             "column_tiles": 256, "column_coverage_exactly_once": 32768,
             "root_and_gm_layout_unchanged": True, "math_task_dependencies_unchanged": True,
             "int8_accumulate_order_unchanged": True, "changes": changes,
             "right_bytes_per_tile": 32768, "acc_bytes_per_tile": 65536,
             "double_buffer_realized": False, "device_exact_passed": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["logical_source_ir"] = str(source)
    return result, audit
