# SPDX-License-Identifier: Apache-2.0
"""把已有RoPE逐行gather合并为一次批量gather，不假定输入索引一定为相邻交换。

原始第r行的列索引c，改为连续Tile中的r*64+c。索引仍来自原Tile，
数学、GM访问、尾块、FFTS和跨核通知不变。这里只操作实验catalog。
"""

import ast
import copy
import json
from pathlib import Path


def use_flat_gather(catalog, output):
    tree = ast.parse(Path(catalog["body_bundle"]).read_text())
    before = ast.dump(tree)
    expected = {"kv_rms_norm_rope": 32, "rmsnorm_rope_cache_write": 16,
                "rmsnorm_rope": 16}
    found = {}
    replacements = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name not in expected:
            continue
        if any(isinstance(n, ast.Name) and n.id.startswith("rope_fg_") for n in ast.walk(fn)):
            raise ValueError("重复应用RoPE批量gather")

        class Rewrite(ast.NodeTransformer):
            def visit_For(self, node):
                if ast.unparse(node.target) != "(gather_lv, (gather_ia,))":
                    return self.generic_visit(node)
                rows = expected[fn.name]
                if ast.unparse(node.iter) != f"pl.range({rows}, init_values=(gather_acc_init,))":
                    raise ValueError("RoPE行数或循环状态改变")
                if len(node.body) != 6 or any(not isinstance(n, ast.AnnAssign) for n in node.body):
                    raise ValueError("原始gather循环形状改变")
                inp, idx, scratch, gather, assemble, result = node.body
                for stmt in (inp, idx):
                    if (ast.unparse(stmt.value.func) != "pl.tile.slice" or
                            [ast.unparse(x) for x in stmt.value.args[1:]] !=
                            ["[1, 64]", "[gather_lv, 0]", "[1, 64]"]):
                        raise ValueError("原始gather源切片改变")
                if (ast.unparse(scratch.value) !=
                        "pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)" or
                        ast.unparse(gather.value) !=
                        "pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)" or
                        ast.unparse(assemble.value) !=
                        "pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])" or
                        ast.unparse(result.value) != "pl.yield_(gather_asmbl)" or
                        ast.unparse(result.target) != "gather_rv"):
                    raise ValueError("原始gather数学或写回改变")
                count = rows * 64
                source = ast.unparse(inp.value.args[0])
                indices = ast.unparse(idx.value.args[0])
                replacement = ast.parse(f"""
rope_fg_values = pl.tile.reshape({source}, [1, {count}])
rope_fg_columns = pl.tile.reshape({indices}, [1, {count}])
rope_fg_linear = pl.tile.ci(pl.const(0, pl.INT32), [1, {count}], dtype=pl.INT32, descending=False)
rope_fg_rows = pl.tile.shls(pl.tile.shrs(rope_fg_linear, 6), 6)
rope_fg_absolute = pl.tile.add(rope_fg_rows, rope_fg_columns)
rope_fg_scratch = pl.tile.create([1, {count}], dtype=pl.INT32, target_memory=pl.Mem.Vec)
rope_fg_selected = pl.tile.gather(rope_fg_values, rope_fg_absolute, rope_fg_scratch)
gather_rv = pl.tile.reshape(rope_fg_selected, [{rows}, 64])
""").body
                found[fn.name] = {"rows": rows, "columns": 64,
                                  "source": source, "indices": indices,
                                  "old_gather_executions_per_loop": rows,
                                  "new_gather_executions_per_loop": 1}
                replacements.append((replacement, copy.deepcopy(node)))
                return replacement

        Rewrite().visit(fn)
    if set(found) != set(expected) or len(replacements) != 3:
        raise ValueError("未精确找到三处RoPE逐行gather")
    # 删除改动块并放回原循环，证明修改范围外的全部AST保持原样。
    restored = copy.deepcopy(tree)
    original_loops = {name: old for name, (_, old) in zip(found, replacements)}
    for fn in ast.walk(restored):
        if not isinstance(fn, ast.FunctionDef) or fn.name not in expected:
            continue

        class Restore(ast.NodeTransformer):
            def visit_Assign(self, node):
                name = ast.unparse(node.targets[0])
                if name == "rope_fg_values":
                    return copy.deepcopy(original_loops[fn.name])
                if name.startswith("rope_fg_") or name == "gather_rv":
                    return None
                return self.generic_visit(node)

        Restore().visit(fn)
    if ast.dump(restored) != before:
        raise ValueError("替换循环以外的AST改变")
    # 任意合法列c都保持同一行；不是只检查相邻交换样本。
    checked = 0
    for rows in expected.values():
        for row in range(rows):
            for column in range(64):
                for selection in range(64):
                    flat = row * 64 + column
                    absolute = ((flat >> 6) << 6) + selection
                    if divmod(absolute, 64) != (row, selection):
                        raise AssertionError("批量索引跨行")
                    checked += 1
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "core_bodies_rope_flat_gather.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    + ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    audit = {"functions": found, "all_other_ast_unchanged": True,
             "all_valid_column_selections_checked": checked,
             "input_index_values_preserved": True, "gm_and_sync_unchanged": True,
             "device_validated": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
