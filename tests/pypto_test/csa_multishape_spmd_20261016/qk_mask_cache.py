# SPDX-License-Identifier: Apache-2.0
"""QK/PV AIV缓存本核query的有效块标记，消除循环内GM Scalar读。

标记由前驱plan任务生成，QK期间只读。仍从当前device Tensor通过MTE2加载，
只保存在本核UB；不添加Host回读或GM展开缓冲，不直接删同步指令。
"""

import ast
import copy
import json
from pathlib import Path


def cache_qk_masks(schedule, catalog, output):
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    fn = next(node for node in ast.walk(module)
              if isinstance(node, ast.FunctionDef) and node.name == "qk_pv_aiv")
    tokens = schedule["root_arguments"]["x_hc"]["shape"][0]
    rows = (tokens + 23) // 24
    if not 1 <= rows <= 16:
        raise ValueError("当前QK掩码缓存支持1..16个query/核")
    mask = "valid_block_mask_inline4413__ssa_v0"
    core = "qk_core_inline4347__ssa_v0"
    hits = 0

    class ReplaceReads(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal hits
            if (isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == "pl.tensor.read"
                    and ast.unparse(node.args[0]) == mask):
                indices = node.args[1]
                if not isinstance(indices, ast.List) or len(indices.elts) != 2:
                    raise ValueError("QK有效块读取维度改变")
                row, col = map(ast.unparse, indices.elts)
                hits += 1
                return ast.parse(f"pl.tile.read(qk_flags_cached, [(({row}) - {core}) // 24, {col}])",
                                 mode="eval").body
            return self.generic_visit(node)

    ReplaceReads().visit(fn)
    if hits != 6:
        raise ValueError(f"QK有效块读取点变化：{hits} != 6")
    prefix = ast.parse(f"""
qk_flags_empty = pl.tile.create([{rows}, 16], dtype=pl.INT32, target_memory=pl.Mem.Vec)
for (qk_flags_q, (qk_flags_iter,)) in pl.range(qk_query_count_inline4331__ssa_v0, init_values=(qk_flags_empty,)):
    qk_flags_row = pl.tile.gather_row(qk_flags_iter, {mask}, [qk_flags_q, 0],
        [{core} + qk_flags_q * 24, 0], [1, 16], transpose=False)
    qk_flags_cached = pl.yield_(qk_flags_row)
""").body
    loops = [i for i, node in enumerate(fn.body) if isinstance(node, ast.For)]
    if len(loops) != 1 or "qk_tick_inline4279__idx_v0" not in ast.unparse(fn.body[loops[0]].target):
        raise ValueError("QK tick主循环不唯一")
    fn.body[loops[0]:loops[0]] = prefix
    ast.fix_missing_locations(module)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "core_bodies_qk_mask_cache.py"
    target.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      "# 当前调用的QK有效块标记缓存在本核UB，不直接删同步指令。\n"
                      + ast.unparse(module) + "\n")
    audit = {"kernel": fn.name, "tokens": tokens, "ub_rows": rows, "ub_bytes": rows * 16 * 4,
             "gm_scalar_read_sites_replaced": hits, "per_query_columns": 16,
             "source": "原plan任务生成的device有效块标记，每次QK重新读取",
             "guards_math_ffts_unchanged": True, "extra_gm_buffer": False,
             "direct_sync_deletion": False, "device_validated": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    updated = copy.deepcopy(catalog)
    updated["body_bundle"] = str(target)
    return updated, audit
