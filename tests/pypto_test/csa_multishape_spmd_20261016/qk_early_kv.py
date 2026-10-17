# SPDX-License-Identifier: Apache-2.0
"""仅T96候选：在现选向量化索引读取上评估既有提前KV通知策略。

生产阈值144有旧实测依据，不将其视为错误。这里仅改变三个同步策略条件，
数学、GM访问、三槽流水及所有FFTS事件保持；是否适合本档由完整验证决定。
"""

import ast
import copy
import json
from pathlib import Path


def use_early_kv(schedule, catalog, output):
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("提前KV通知候选只验证T96")
    tree = ast.parse(Path(catalog["body_bundle"]).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "qk_pv_aiv")
    before = ast.dump(tree)
    changed = []
    expected = {"144 <= t_dim_inline4335__ssa_v0": 1,
                "t_dim_inline4335__ssa_v0 < 144": 2}
    found = {key: 0 for key in expected}
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        text = ast.unparse(node.test)
        if text not in expected:
            continue
        constants = [n for n in ast.walk(node.test) if isinstance(n, ast.Constant) and n.value == 144]
        if len(constants) != 1:
            raise ValueError("同步条件不含唯一阈值")
        constants[0].value = 96
        changed.append(constants[0])
        found[text] += 1
    if found != expected:
        raise ValueError(f"同步策略分支数量改变：{found}")
    candidate = ast.unparse(ast.fix_missing_locations(tree))
    for node in changed:
        node.value = 144
    if ast.dump(tree) != before:
        raise ValueError("除三个同步阈值外出现其他改变")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "core_bodies_qk_early_kv.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    + candidate + "\n")
    audit = {"changed_function": fn.name, "old_threshold_tokens": 144, "new_threshold_tokens": 96,
             "changed_conditions": found, "all_other_ast_unchanged": True,
             "qk_transfer_slots": 3, "qk_pre_launch": 2,
             "mathematical_ops_and_gm_accesses_unchanged": True,
             "requires_selected_vector_index_cache": True, "device_validated": False,
             "scope": "分档策略实验，不宣称生产阈值错误或提前通知一定提速"}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
