# SPDX-License-Identifier: Apache-2.0
"""两个原1024分数块合为一次2048交接，保留N64展开及每2048段Top-K。

每叶不改变候选分区。AIC只计算实际需要的1024子块；AIV的尾段span同时
受本叶剩余有效数约束，不能读入相邻叶或尚未写入的score_transfer尾部。
扩大的是私有双槽GM工作区；Native cache与所有数学根入参布局不变。
"""

import ast
import copy
import json
from pathlib import Path


TRANSFER = "buf_score_transfer_inline8845__ssa_v0"


def change_aic(fn):
    iterations = next(n for n in ast.walk(fn) if isinstance(n, ast.AnnAssign)
                      and ast.unparse(n.target) == "buf_score_iters_inline8863__ssa_v0")
    iterations.value = ast.parse("(buf_valid_count_inline8796__ssa_v0 + 2047) // 2048", mode="eval").body
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For)
                and ast.unparse(n.target) == "buf_score_step_inline8719__idx_v0")
    if len(loop.body) != 434:
        raise ValueError("原1024分数循环结构改变")
    wait, begin, transfer = loop.body[:3]
    ready = loop.body[-1]
    math = loop.body[3:-1]
    if "sync_wait(1" not in ast.unparse(wait) or "sync_set(0" not in ast.unparse(ready):
        raise ValueError("原FFTS事件改变")
    begin.value = ast.parse("buf_score_step_inline8719__idx_v0 * 2048 + transfer_subblock * 1024", mode="eval").body
    stores = [n for n in ast.walk(ast.Module(body=math, type_ignores=[]))
              if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.tile.store"
              and ast.unparse(n.args[2]) == TRANSFER]
    if len(stores) != 16:
        raise ValueError("原N64分数store不是16处")
    for store in stores:
        old = store.args[1].elts[1]
        store.args[1].elts[1] = ast.BinOp(left=old, op=ast.Add(), right=ast.parse(
            "transfer_subblock * 1024", mode="eval").body)
    compute = ast.If(test=ast.parse("buf_score_begin_inline8717__ssa_v1 < buf_valid_count_inline8796__ssa_v0",
                                   mode="eval").body, body=math, orelse=[])
    subloop = ast.For(target=ast.Name(id="transfer_subblock", ctx=ast.Store()),
                      iter=ast.parse("pl.range(2)", mode="eval").body,
                      body=[begin, compute], orelse=[])
    loop.body = [wait, transfer, subloop, ready]


def vector_prefix():
    lines = [
        "single_score_begin_inline8724__ssa_v0 = single_begin_inline8697__ssa_v0 + single_step_inline8751__idx_v0 * 2048",
        "single_transfer_row_inline8662__ssa_v0 = buf_worker_inline8783__ssa_v0 * 12 + single_step_inline8751__idx_v0 % 2 * 6 + buf_score_lane_inline8716__ssa_v0 * 3",
        "transfer_scale_bytes = pl.tile.create([1, 4096], dtype=pl.INT8, target_memory=pl.Mem.Vec)",
        "for transfer_page, (transfer_scale_iter,) in pl.range(64, init_values=(transfer_scale_bytes,)):",
        "    transfer_logical = pl.min(single_score_begin_inline8724__ssa_v0 // 32 + transfer_page, pl.max((single_cache_len_inline8773__ssa_v0 - 1) // 32, 0))",
        "    transfer_physical = pl.max(pl.cast(pl.tensor.read(idx_block_table__ssa_v0, [single_batch_inline8814__ssa_v0, transfer_logical]), pl.INDEX), 0)",
        "    transfer_scale_new = pl.tile.gather_row(transfer_scale_iter, idx_native_kv_cache__rv_v2, [0, transfer_page * 64], [transfer_physical, 4096], [1, 64], transpose=False)",
        "    transfer_scale_done = pl.yield_(transfer_scale_new)",
        "transfer_scale_half = pl.tile.reinterpret_view(transfer_scale_done, dtype=pl.FP16)",
        "transfer_kv_scale = pl.tile.cast(transfer_scale_half, target_type=pl.FP32, mode='round')",
        "pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)",
        "transfer_scores = pl.tile.load(buf_score_transfer_inline8845__ssa_v0, [single_transfer_row_inline8662__ssa_v0, 0], [3, 2048], [3, 2048], target_memory=pl.Mem.Vec)",
        "pl.system.sync_set(1, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV)",
        "transfer_segments = pl.tile.create([3, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)",
    ]
    for query in range(3):
        lines.extend([
            f"transfer_row_{query} = pl.tile.extract(transfer_scores, {query}, 0, [1, 2048], target_memory=pl.Mem.Vec)",
            f"transfer_scaled_{query} = pl.tile.mul(transfer_row_{query}, transfer_kv_scale)",
            f"transfer_segments = pl.tile.assemble(transfer_segments, transfer_scaled_{query}, [{query}, 0])",
        ])
    lines.extend([
        "single_segment_rows_inline8736__ssa_v0 = transfer_segments",
        "single_segment_begin_inline8760__ssa_v0 = single_score_begin_inline8724__ssa_v0",
        "single_segment_span_inline8784__ssa_v0 = pl.min(2048, single_valid_inline8808__ssa_v0 - single_step_inline8751__idx_v0 * 2048)",
    ])
    return ast.parse("\n".join(lines)).body


def change_aiv(fn):
    item = next(n for n in ast.walk(fn) if isinstance(n, ast.For)
                and ast.unparse(n.target) == "single_item_inline8866__idx_v0")
    leaf = item.body[-1]
    count = next(n for n in leaf.body if isinstance(n, ast.AnnAssign)
                 and ast.unparse(n.target) == "single_steps_inline8664__ssa_v0")
    count.value = ast.parse("(single_valid_inline8808__ssa_v0 + 2047) // 2048", mode="eval").body
    initial = next(n for n in leaf.body if isinstance(n, ast.AnnAssign)
                   and ast.unparse(n.target) == "single_segments_inline8768__ssa_v0")
    leaf.body.remove(initial)
    loop = next(n for n in leaf.body if isinstance(n, ast.For)
                and ast.unparse(n.target).startswith("(single_step"))
    topk_guard = next(n for n in loop.body if isinstance(n, ast.If))
    if ast.unparse(topk_guard.test) != "single_step_inline8751__idx_v0 % 2 == 1 or single_step_inline8751__idx_v0 + 1 == single_steps_inline8664__ssa_v0":
        raise ValueError("原TopK的2048候选段边界改变")
    if (not isinstance(topk_guard.body[-1], ast.AnnAssign)
            or ast.unparse(topk_guard.body[-1].value) != "pl.yield_(single_roots_inline8822__phi_v14)"):
        raise ValueError("原TopK合并出口改变")
    # 排序/合并数学保持；旧循环每两轮产出一个段，新的每轮就是一个段。
    # 因而 root 初始化必须从旧 step<2 改为新 step==0，否则第二段覆盖第一段。
    topk = topk_guard.body[3:-1]
    original_topk = [ast.dump(n) for n in topk]
    root_init = [n for root in topk for n in ast.walk(root)
                 if isinstance(n, ast.Compare)
                 and ast.unparse(n) == "single_step_inline8751__idx_v0 < 2"]
    if len(root_init) != 3:
        raise ValueError("原三query的root初始化条件改变")
    for condition in root_init:
        condition.ops = [ast.Eq()]
        condition.comparators = [ast.Constant(value=0)]
    loop.target = ast.parse("single_step_inline8751__idx_v0, (single_roots_inline8822__iter_v1,) = (0, (0,))").body[0].targets[0]
    loop.iter = ast.parse("pl.range(single_steps_inline8664__ssa_v0, init_values=(single_roots_inline8822__ssa_v0,))", mode="eval").body
    loop.body = vector_prefix() + topk + ast.parse(
        "single_roots_inline8822__rv_v2 = pl.yield_(single_roots_inline8822__phi_v14)").body
    # 反向恢复这三处条件后，其他数学AST须逐项保持。
    restored = copy.deepcopy(topk)
    for root in restored:
        for node in ast.walk(root):
            if isinstance(node, ast.Compare) and ast.unparse(node) == "single_step_inline8751__idx_v0 == 0":
                node.ops = [ast.Lt()]
                node.comparators = [ast.Constant(value=2)]
    if [ast.dump(n) for n in restored] != original_topk:
        raise AssertionError("除root首段判定外，TopK排序和root合并不应改变")
    if any(isinstance(n, ast.Name) and n.id.startswith("single_segments_inline8768") for n in ast.walk(fn)):
        raise ValueError("旧两步segments缓存仍被使用")
    return len(topk)


def use_transfer_2048(schedule, catalog, output):
    call = next(c for c in schedule["call_bindings"].values()
                if c["logical_function"] == "indexer_score_topk_native_pair")
    arg = next(a for a in call["arguments"] if a["parameter"] == TRANSFER)
    allocation = arg["value"]["allocation"]
    if arg["value"]["shape"] != [288, 1024] or arg["value"]["offset"] != 0:
        raise ValueError("原分数工作区合同改变")
    refs = [(tid, a) for tid, c in schedule["call_bindings"].items() for a in c["arguments"]
            if isinstance(a["value"], dict) and a["value"].get("allocation") == allocation]
    if len(refs) != 1:
        raise ValueError("分数工作区不再是Indexer任务独占")
    temp = next(a for a in schedule["temporary_allocations"] if a["allocation"] == allocation)
    old_bytes = temp["bytes"]
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    names = ["indexer_score_topk_native_pair_aic", "indexer_score_topk_native_pair_aiv"]
    functions = {n.name: n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}
    before = {name: ast.dump(fn) for name, fn in functions.items()}
    change_aic(functions[names[0]])
    topk_nodes = change_aiv(functions[names[1]])

    class ResizeTransfer(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if ast.unparse(node) == "pl.Tensor[[288, 1024], pl.FP32]":
                return ast.parse("pl.Tensor[[288, 2048], pl.FP32]", mode="eval").body
            return self.generic_visit(node)

    for name in names:
        ResizeTransfer().visit(functions[name])
    changed = [name for name, fn in functions.items() if ast.dump(fn) != before[name]]
    if set(changed) != set(names):
        raise AssertionError("变更超出Indexer生产/消费两函数")
    for ref in (arg["value"], temp):
        ref["shape"] = [288, 2048]
        ref["strides"] = [2048, 1]
    temp["bytes"] = old_bytes * 2
    arg["annotation"] = arg["annotation"].replace("[288, 1024]", "[288, 2048]")
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "core_bodies_indexer_transfer_2048.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    + ast.unparse(module) + "\n")
    audit = {"changed_functions": changed, "score_transfer_allocation": allocation,
             "score_transfer_bytes_before": old_bytes, "score_transfer_bytes_after": temp["bytes"],
             "new_score_block": 2048, "math_subblock": 1024, "qk_panel_n": 64,
             "only_existing_valid_subblocks_computed": True, "topk_math_nodes_preserved": topk_nodes,
             "root_init_conditions_remapped": 3,
             "topk_segment_bound_by_leaf_valid_remaining": True,
             "leaf_partition_and_pair_arena_unchanged": True,
             "root_abi_and_native_layouts_unchanged": True, "private_workspace_resized": True,
             "ffts_one_ready_consumed_per_2048": True, "ffts_two_slot_drain_preserved": True,
             "device_validated": False}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
