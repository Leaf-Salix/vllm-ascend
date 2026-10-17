# SPDX-License-Identifier: Apache-2.0
"""QK压缩KV按行MGATHER候选；仅作用于已有UB地址缓存的AIV数学体。

A3的MGATHER仍通过Scalar提交逐行MTE2，不能声称一次硬件gather完成。
这里的目的在于缩短编译后的逐行控制流，让现有ISA宏管理搬运内部同步。
负稀疏索引保留符号位成为越界哨兵，由gather_oob=zero填零。
"""

import ast
import copy
import json
from pathlib import Path


def use_mgather(schedule, catalog, output):
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    fn = next(node for node in ast.walk(module)
              if isinstance(node, ast.FunctionDef) and node.name == "qk_pv_aiv")
    names = {node.id for node in ast.walk(fn) if isinstance(node, ast.Name)}
    if "qk_meta_physical_rows" not in names or "qk_gather_rows" in names:
        raise ValueError("MGATHER候选需要恰好一份已有QK地址缓存")
    rows = schedule["root_arguments"]["cmp_kv"]["shape"][0] * 32
    if not 0 < rows < 2**31:
        raise ValueError("负索引哨兵要求物理KV行数处于INT32正数范围")
    replacement = ast.parse("""
qk_gather_negative = pl.tile.minimums(qk_meta_raw, 0)
qk_gather_rows = pl.tile.or_(qk_meta_physical_rows, qk_gather_negative)
qk_kv_half_inline4272__rv_v17 = pl.tile.mgather(
    cmp_kv_flat_inline4388__ssa_v0, qk_gather_rows,
    coalesce='row', gather_oob='zero', target_memory=pl.Mem.Vec)
""").body
    hits = 0

    class ReplaceLoop(ast.NodeTransformer):
        def visit_For(self, node):
            nonlocal hits
            if "qk_row_inline4264__idx_v0" not in {
                    n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}:
                return self.generic_visit(node)
            if ast.unparse(node.iter.args[0]) != "64":
                raise ValueError("原压缩块不是64行")
            hits += 1
            return copy.deepcopy(replacement)

    ReplaceLoop().visit(fn)
    if hits != 1:
        raise ValueError("未找到唯一64行压缩KV循环")
    # 原逻辑在SWA/压缩分支外先分配64KiB零Tile，压缩分支再分配一次。
    # MGATHER自己产出完整64行，保留那份跨分支活跃的零Tile会使UB超过上限。
    # 把SWA所需的初始零Tile移入SWA分支，压缩分支由MGATHER的zero语义覆盖。
    def is_initial_zero(node):
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
            if (is_initial_zero(current) and isinstance(branch, ast.If)
                    and ast.unparse(branch.test) == "qk_sb_inline4367__ssa_v1 == 0"):
                if not is_initial_zero(branch.orelse[0]):
                    raise ValueError("压缩KV分支的初始零Tile结构改变")
                branch.body.insert(0, current)
                branch.orelse.pop(0)
                body.pop(index)
                moved += 1
                break
    if moved != 1:
        raise ValueError("SWA初始零Tile的分支移动不是唯一的")
    ast.fix_missing_locations(module)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "core_bodies_qk_mgather.py"
    target.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      "# QK压缩KV的MGATHER候选，不改变数学与跨核协议。\n"
                      + ast.unparse(module) + "\n")
    audit = {"kernel": fn.name, "rows_per_block": 64,
             "physical_kv_rows": rows, "gather_oob": "zero",
             "negative_sentinel": "physical_row | min(raw_sparse_index, 0)，负数符号位保持",
             "valid_index_address_unchanged": True,
             "swa_zero_tile_scoped_to_swa_branch": True,
             "qk_pv_math_unchanged": True, "ffts_unchanged": True,
             "extra_gm_buffer": False, "host_metadata_read": False,
             "device_validated": False,
             "scope": "A3现有ISA宏内仍逐行DMA，性能和负索引行为须实测"}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(target)
    return result, audit
