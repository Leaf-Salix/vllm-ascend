# SPDX-License-Identifier: Apache-2.0
"""QK压缩KV全有效块的搬运候选；含负索引的块保留原回退。

前提是qk_index_cache已生成当前64行的UB地址。先做INT32最小值规约，
确认整块索引非负后，每8行展开DMA，省去逐行原始索引读取及条件跳转。
页表、稀疏索引和有效性仍逐步读取device Tensor，不假设历史长度固定。
"""

import ast
import copy
import json
from pathlib import Path


def use_full_block(catalog, output):
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    fn = next(node for node in ast.walk(module)
              if isinstance(node, ast.FunctionDef) and node.name == "qk_pv_aiv")
    names = {node.id for node in ast.walk(fn) if isinstance(node, ast.Name)}
    if "qk_meta_physical_rows" not in names or "qk_full_min" in names:
        raise ValueError("需要恰好一份已有QK地址缓存")
    old_result = "qk_kv_half_inline4272__rv_v17"
    lines = [
        # row_min结果为列主序，INT32的列跨度须32B；使用8个相同的行。
        "qk_full_base = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)",
        "qk_full_rows = pl.tile.col_expand(qk_full_base, qk_meta_raw)",
        "qk_full_tmp = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)",
        "qk_full_min = pl.tile.row_min(qk_full_rows, qk_full_tmp)",
        "if pl.tile.read(qk_full_min, [0, 0]) >= 0:",
        "    for (qk_fast_chunk, (qk_fast_iter,)) in pl.range(8, init_values=(qk_kv_half_inline4272__ssa_v0,)):",
    ]
    for lane in range(8):
        previous = "qk_fast_iter" if lane == 0 else f"qk_fast_tile_{lane - 1}"
        lines.extend([
            f"        qk_fast_row_{lane} = qk_fast_chunk * 8 + {lane}",
            f"        qk_fast_src_{lane} = pl.cast(pl.tile.read(qk_meta_physical_rows, [0, qk_fast_row_{lane}]), pl.INDEX)",
            f"        qk_fast_tile_{lane} = pl.tile.gather_row({previous}, cmp_kv_flat_inline4388__ssa_v0, "
            f"[qk_fast_row_{lane}, 0], [qk_fast_src_{lane}, 0], [1, 512], transpose=False)",
        ])
    lines.extend([
        "        qk_fast_result = pl.yield_(qk_fast_tile_7)",
        f"    {old_result} = pl.yield_(qk_fast_result)",
        "else:",
        "    pass",
    ])
    replacement = ast.parse("\n".join(lines)).body
    hits = 0

    class RenameResult(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == old_result:
                node.id = "qk_fallback_result"
            return node

    class ReplaceLoop(ast.NodeTransformer):
        def visit_For(self, node):
            nonlocal hits
            if "qk_row_inline4264__idx_v0" not in {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}:
                return self.generic_visit(node)
            if ast.unparse(node.iter.args[0]) != "64":
                raise ValueError("原压缩块不是64行")
            hits += 1
            fallback = RenameResult().visit(copy.deepcopy(node))
            result = copy.deepcopy(replacement)
            result[-1].orelse = [fallback] + ast.parse(
                f"{old_result} = pl.yield_(qk_fallback_result)").body
            return result

    ReplaceLoop().visit(fn)
    if hits != 1:
        raise ValueError("未找到唯一压缩行循环")
    ast.fix_missing_locations(module)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "core_bodies_qk_full_block.py"
    target.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      "# 全有效稀疏块搬运候选；负索引块保留原循环。\n"
                      + ast.unparse(module) + "\n")
    audit = {"kernel": fn.name, "rows": 64, "unroll_factor": 8,
             "validity": "device INT32 row_min >= 0",
             "fallback": "原64行条件循环，逐节点保留，仅结果变量重命名",
             "qk_math_unchanged": True, "ffts_unchanged": True,
             "extra_gm_buffer": False, "device_validated": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(target)
    return result, audit
