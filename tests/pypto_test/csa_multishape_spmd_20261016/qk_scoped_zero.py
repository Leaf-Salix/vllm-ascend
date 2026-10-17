# SPDX-License-Identifier: Apache-2.0
"""将 QK 的 SWA 零初始化收进分支，避免压缩块多执行一次 64 KiB 清零。

原始代码在 SWA/压缩分支之前清零，压缩分支进入后又清零同名 Tile。
只移动前一份初始化；逐行 gather、负索引留零、数学与 FFTS 保持原样。
"""

import ast
import copy
import json
from pathlib import Path


def scope_zero(catalog, output):
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    fn = next(node for node in ast.walk(module)
              if isinstance(node, ast.FunctionDef) and node.name == "qk_pv_aiv")

    def is_zero(node):
        return (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "qk_kv_half_inline4272__ssa_v0"
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "pl.tile.full")

    moved = 0
    for parent in list(ast.walk(fn)):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for index in range(len(body) - 1):
            current, branch = body[index:index + 2]
            if (is_zero(current) and isinstance(branch, ast.If)
                    and ast.unparse(branch.test) == "qk_sb_inline4367__ssa_v1 == 0"):
                if (not is_zero(branch.orelse[0])
                        or ast.dump(current) != ast.dump(branch.orelse[0])):
                    raise ValueError("压缩块不是覆盖同名 Tile 的相同初始化")
                branch.body.insert(0, body.pop(index))
                moved += 1
                break
    if moved != 1:
        raise ValueError(f"预期唯一 SWA/压缩分支，实际移动 {moved} 处")
    ast.fix_missing_locations(module)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "core_bodies_qk_scoped_zero.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    "# QK 初始化限定到各自分支；保留逐行负索引处理。\n"
                    + ast.unparse(module) + "\n")
    audit = {"kernel": fn.name, "moved_initializations": moved,
             "compressed_block_zero_fills_before": 2,
             "compressed_block_zero_fills_after": 1,
             "swa_block_zero_fills_before_and_after": 1,
             "tile_bytes": 64 * 512 * 2,
             "gather_math_ffts_unchanged": True,
             "negative_index_behavior_unchanged": True,
             "extra_gm_buffer": False, "device_validated": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
