# SPDX-License-Identifier: Apache-2.0
"""Indexer最后一个1024候选块仅剩1..64项时，只计算第一个N64面板。

分支来自已有device valid_count，不从Host读取长度或写死131072。原16面板
路径逐AST保留；QK K128、WS K384、FIX的FP16缩放/ReLU、FFTS及AIV TopK
均不变。未写的960列属于内部score_transfer无效尾部，消费者先逐元素缩放，
再按各query可见长度fillpad；它们不能进入排序。它们不是已初始化的新有效数据。
"""

import ast
import copy
import json
from pathlib import Path


def short_tail_math():
    lines = [
        "tail_key_l1 = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)",
    ]
    # 与原N64完全相同的两个Native页，保留非法页到0及cache末页夹取规则。
    for page in range(2):
        prefix = f"tail_page{page}"
        lines.extend([
            f"{prefix}_logical = pl.min((buf_logical_begin_inline8767__ssa_v1 + "
            f"buf_score_begin_inline8717__ssa_v1 + {page * 32}) // 32, "
            "pl.max((buf_cache_len_inline8809__ssa_v0 - 1) // 32, 0))",
            f"{prefix}_physical = pl.max(pl.cast(pl.tensor.read(idx_block_table__ssa_v0, "
            f"[buf_batch_idx_inline8687__ssa_v0, {prefix}_logical]), pl.INDEX), 0)",
            f"{prefix}_byte = {prefix}_physical * native_page_bytes_inline8726__ssa_v0",
            f"if {prefix}_byte % 128 == 0:",
            f"    tail_key_l1 = pl.tile.gather_row(tail_key_l1, idx_kv_cache_inline8854__ssa_v0, "
            f"[{page * 32}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
            "else:",
            f"    tail_key_l1 = pl.tile.gather_row(tail_key_l1, idx_kv_cache_shift64_inline8790__ssa_v0, "
            f"[{page * 32}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
        ])
    lines.extend([
        "tail_key = pl.tile.move(pl.tile.transpose_view(tail_key_l1), target_memory=pl.Mem.Right)",
        "tail_qk = pl.tile.matmul(buf_query_left_inline8850__ssa_v0, tail_key)",
        "tail_score_empty = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)",
        "tail_score = pl.tile.assemble(tail_score_empty, tail_qk, [0, 0], "
        "pre_quant=0.0009765625, pre_relu=True)",
        "tail_ws_right = pl.tile.move(tail_score, target_memory=pl.Mem.Right)",
        "tail_ws = pl.tile.matmul(buf_coefficient_pair_inline8676__ssa_v0, tail_ws_right)",
        "pl.tile.store(pl.tile.set_validshape(tail_ws, 6, 64), "
        "[buf_transfer_row_inline8782__ssa_v0, 0], buf_score_transfer_inline8845__ssa_v0)",
    ])
    return ast.parse("\n".join(lines)).body


def use_short_tail(schedule, catalog, output):
    call = next(c for c in schedule["call_bindings"].values()
                if c["logical_function"] == "indexer_score_topk_native_pair")
    transfer = next(a["value"] for a in call["arguments"]
                    if a["parameter"] == "buf_score_transfer_inline8845__ssa_v0")
    if transfer["shape"] != [288, 1024]:
        raise ValueError("短尾候选只适用于长历史S6的原N64/1024分数块")
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    before = {n.name: ast.dump(n) for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}
    fn = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
              and n.name == "indexer_score_topk_native_pair_aic")
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For)
                and ast.unparse(n.target) == "buf_score_step_inline8719__idx_v0")
    if (len(loop.body) != 434 or not isinstance(loop.body[0], ast.If)
            or "sync_wait(1" not in ast.unparse(loop.body[0])
            or ast.unparse(loop.body[1].target) != "buf_score_begin_inline8717__ssa_v1"
            or ast.unparse(loop.body[2].target) != "buf_transfer_row_inline8782__ssa_v0"
            or ast.unparse(loop.body[-1]) != "pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)"):
        raise ValueError("原N64分数循环/FFTS合同改变")
    original = loop.body[3:-1]
    original_ast = [ast.dump(n) for n in original]
    branch = ast.If(test=ast.parse(
        "buf_valid_count_inline8796__ssa_v0 - buf_score_begin_inline8717__ssa_v1 <= 64",
        mode="eval").body, body=short_tail_math(), orelse=original)
    loop.body = loop.body[:3] + [branch] + loop.body[-1:]
    if [ast.dump(n) for n in branch.orelse] != original_ast:
        raise AssertionError("原1024路径发生变化")
    changed = [n.name for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
               and ast.dump(n) != before[n.name]]
    if changed != [fn.name]:
        raise AssertionError(f"除Indexer Cube外出现变化：{changed}")
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "core_bodies_indexer_short_tail.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    + ast.unparse(module) + "\n")
    audit = {"changed_functions": changed, "fast_path_valid_count": [1, 64],
             "device_length_branch": True, "fallback_ast_unchanged": True,
             "qk_m_n_k": [384, 64, 128], "ws_m_n_k": [16, 64, 384],
             "fix_fp16_quant_relu_unchanged": True, "ffts_wait_set_drain_unchanged": True,
             "aiv_topk_unchanged": True, "new_gm_buffers": 0, "host_readback": False,
             "unwritten_transfer_columns": [64, 1024],
             "unwritten_columns_are_masked_before_sort": True,
             "device_validated": False}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
