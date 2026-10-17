# SPDX-License-Identifier: Apache-2.0
"""QK的每AIV 64行KV拆成两个32行块，交接槽、数学、通知位置保持。

缩小本地搬运Tile，两个块分别写原GM交接区的相邻行。只有完整64行写入后
才到原KV_READY发布点。SWA有效行与负页/负稀疏索引仍遵循原规则。
"""

import ast
import copy
import json
from pathlib import Path


def producer_node(function):
    matches = [node for node in ast.walk(function) if isinstance(node, ast.If)
               and any(isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
                       and child.target.id == "qk_kv_half_inline4272__ssa_v0" for child in node.body)]
    if len(matches) != 1:
        raise ValueError("原KV准备块不唯一")
    return matches[0]


def use_chunked_gather(catalog, output):
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    functions = {node.name: node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)}
    function = functions["qk_pv_aiv"]
    before = copy.deepcopy(function)
    block = producer_node(function)
    if len(block.body) != 5:
        raise ValueError("原KV准备块不再是slot/row/full/branch/store结构")
    slot, row, init, branch, store = copy.deepcopy(block.body)
    if ast.unparse(branch.test) != "qk_sb_inline4367__ssa_v1 == 0":
        raise ValueError("SWA/压缩分支改变")
    if ast.unparse(store.value.func) != "pl.tile.store":
        raise ValueError("原KV交接不是tile.store")
    old_store = store.value
    loop = next(node for node in branch.orelse if isinstance(node, ast.For))
    prefix = [copy.deepcopy(node) for node in branch.orelse
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id.startswith("qk_meta_")]
    if len(prefix) != 9:
        raise ValueError("本候选需要已有64行QK地址缓存")
    # Tile缩小不改变head维；所有Tile phi/yield维度同步缩小。
    class HalfRows(ast.NodeTransformer):
        def visit_List(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node) == "[64, 512]":
                node.elts[0] = ast.Constant(32)
            return node

    small_init = HalfRows().visit(init)
    swa = ast.Module(body=branch.body[:-1], type_ignores=[])
    # 原分支末尾yield的是外层合流值；在各自块内直接写GM，无需合流64行Tile。
    if ast.unparse(branch.body[-1].value.func) != "pl.yield_":
        raise ValueError("原SWA分支结果结构改变")
    swa_result = copy.deepcopy(branch.body[-1].value.args[0])
    changes = {"swa_lane_offsets": 0, "swa_row_limit": 0, "cmp_read_offsets": 0}

    class SWAChunk(HalfRows):
        def visit_Name(self, node):
            if node.id == "qk_lane_kv_inline4438__ssa_v0" and isinstance(node.ctx, ast.Load):
                changes["swa_lane_offsets"] += 1
                return ast.copy_location(ast.parse(
                    "qk_lane_kv_inline4438__ssa_v0 + qk_chunk * 32", mode="eval").body, node)
            return node

        def visit_AnnAssign(self, node):
            node = self.generic_visit(node)
            if node.target.id == "qk_rows_inline4268__ssa_v0":
                if ast.unparse(node.value.args[-1]) != "64":
                    raise ValueError("SWA有效行上限改变")
                node.value.args[-1] = ast.Constant(32)
                changes["swa_row_limit"] += 1
            return node

    swa = SWAChunk().visit(swa)
    cmp_loop = HalfRows().visit(copy.deepcopy(loop))
    if ast.unparse(cmp_loop.iter.args[0]) != "64":
        raise ValueError("原压缩搬运不是64行循环")
    cmp_loop.iter.args[0] = ast.Constant(32)
    for node in ast.walk(cmp_loop):
        if isinstance(node, ast.AnnAssign) and node.target.id == "qk_cmp_k_inline4263__ssa_v0":
            node.value = ast.BinOp(node.value, ast.Add(), ast.parse("qk_chunk * 32", mode="eval").body)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.tile.read":
            if ast.unparse(node.args[0]) not in ("qk_meta_raw", "qk_meta_physical_rows"):
                raise ValueError("压缩分支出现未知的Tile Scalar读")
            old = node.args[1].elts[1]
            node.args[1].elts[1] = ast.BinOp(old, ast.Add(), ast.parse("qk_chunk * 32", mode="eval").body)
            changes["cmp_read_offsets"] += 1
    cmp_result = copy.deepcopy(branch.orelse[-1].value.args[0])

    def chunk_loop(body, result):
        output_call = copy.deepcopy(old_store)
        output_call.args[0] = result
        output_call.args[1].elts[0] = ast.BinOp(output_call.args[1].elts[0], ast.Add(),
                                             ast.parse("qk_chunk * 32", mode="eval").body)
        result_loop = ast.parse("for qk_chunk in pl.range(2):\n    pass\n").body[0]
        result_loop.body = [copy.deepcopy(small_init)] + body + [ast.Expr(output_call)]
        return result_loop

    branch.body = [chunk_loop(swa.body, swa_result)]
    branch.orelse = prefix + [chunk_loop([cmp_loop], cmp_result)]
    block.body = [slot, row, branch]
    if changes != {"swa_lane_offsets": 15, "swa_row_limit": 1, "cmp_read_offsets": 2}:
        raise ValueError("KV分块变换点不符：" + str(changes))
    # 移去前后两版唯一KV准备块后，函数其余部分必须完全相同。
    original_block = producer_node(before)
    original_block.body = [ast.Pass()]
    saved = block.body
    block.body = [ast.Pass()]
    if ast.dump(before) != ast.dump(function):
        raise ValueError("意外改变了KV准备以外的数学或通信")
    block.body = saved
    ast.fix_missing_locations(module)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "core_bodies_qk_chunked_gather.py"
    source.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      + ast.unparse(module) + "\n")
    (directory / "producer.py").write_text(ast.unparse(ast.Module(body=block.body, type_ignores=[])) + "\n")
    audit = {"chunk_rows": 32, "chunks_per_aiv": 2, "producer_changes": changes,
             "other_math_and_communication_ast_unchanged": True,
             "native_views_and_gm_transfer_layout_unchanged": True,
             "kv_ready_stays_after_both_chunk_stores": True,
             "device_validated": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(source)
    return result, audit
