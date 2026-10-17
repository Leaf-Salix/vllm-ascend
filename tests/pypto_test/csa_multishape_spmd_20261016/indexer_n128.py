# SPDX-License-Identifier: Apache-2.0
"""长历史S6 Indexer的N128候选：两组三query共用一次Key加载。

直接使用M384/N128会超过128KiB Acc。本候选每次计算三个query，
两个M192/N128 QK共用Key Right；下一Key预取到L1，不占第二个Right槽。
WS Right占48KiB，加Key的16KiB正好64KiB。WS去掉另一组三query的
零系数项；数学等价不能代替逐位验证，仍使用原输出和Top-K闸门。
"""

import ast
import copy
import json
from pathlib import Path


def n128_math():
    lines = []
    # 两个Query Left各24KiB，两个系数Left各6KiB，合计60KiB。
    for half in range(2):
        lines.extend([
            f"n128_query_{half} = pl.tile.extract(buf_query_vector_inline8886__ssa_v0, "
            f"{half * 192}, 0, [192, 128], target_memory=pl.Mem.Left)",
            f"n128_coef_gm_{half} = pl.tile.load(coefficients_inline8733__ssa_v0, "
            f"[buf_query_inline8778__phi_v2 // 6 * 16 + {half * 3}, {half * 192}], "
            "[16, 192], [3, 192], target_memory=pl.Mem.Mat)",
            f"n128_coef_{half} = pl.tile.move(n128_coef_gm_{half}, target_memory=pl.Mem.Left)",
        ])
    lines.extend([
        "for n128_step in pl.range(buf_score_iters_inline8863__ssa_v0):",
        "    if n128_step >= 2:",
        "        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)",
        "    n128_begin = buf_logical_begin_inline8767__ssa_v1 + n128_step * 1024",
        "    n128_transfer = buf_worker_inline8783__ssa_v0 * 12 + n128_step % 2 * 6",
        "    n128_key_pool = pl.tile.create([256, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)",
    ])

    def load_key(panel):
        slot = panel % 2 * 128
        for page in range(4):
            prefix = f"n128_p{panel}_page{page}"
            lines.extend([
                f"    {prefix}_logical = pl.min((n128_begin + {panel * 128 + page * 32}) // 32, "
                "pl.max((buf_cache_len_inline8809__ssa_v0 - 1) // 32, 0))",
                f"    {prefix}_physical = pl.max(pl.cast(pl.tensor.read(idx_block_table__ssa_v0, "
                f"[buf_batch_idx_inline8687__ssa_v0, {prefix}_logical]), pl.INDEX), 0)",
                f"    {prefix}_byte = {prefix}_physical * native_page_bytes_inline8726__ssa_v0",
                f"    if {prefix}_byte % 128 == 0:",
                f"        n128_key_pool = pl.tile.gather_row(n128_key_pool, idx_kv_cache_inline8854__ssa_v0, "
                f"[{slot + page * 32}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
                "    else:",
                f"        n128_key_pool = pl.tile.gather_row(n128_key_pool, idx_kv_cache_shift64_inline8790__ssa_v0, "
                f"[{slot + page * 32}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
            ])

    def extract_key(panel):
        lines.append(f"    n128_key_{panel} = pl.tile.extract(pl.tile.transpose_view(n128_key_pool), "
                     f"0, {panel % 2 * 128}, [128, 128], target_memory=pl.Mem.Right)")

    def reduce_half(panel, half):
        prefix = f"n128_ws_{panel}_{half}"
        lines.extend([
            f"    {prefix}_right = pl.tile.move(n128_score_l1_{panel}_{half}, target_memory=pl.Mem.Right)",
            f"    {prefix}_acc = pl.tile.matmul(n128_coef_{half}, {prefix}_right)",
            f"    pl.tile.store(pl.tile.set_validshape({prefix}_acc, 3, 128), "
            f"[n128_transfer + {half * 3}, {panel * 128}], buf_score_transfer_inline8845__ssa_v0)",
        ])

    load_key(0)
    extract_key(0)
    for panel in range(8):
        if panel < 7:
            load_key(panel + 1)
        for half in range(2):
            lines.append(f"    n128_qk_{panel}_{half} = pl.tile.matmul(n128_query_{half}, n128_key_{panel})")
            if half == 0 and panel:
                reduce_half(panel - 1, 1)
            elif half == 1:
                reduce_half(panel, 0)
            lines.extend([
                f"    n128_score_empty_{panel}_{half} = pl.tile.create([192, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)",
                f"    n128_score_l1_{panel}_{half} = pl.tile.assemble(n128_score_empty_{panel}_{half}, n128_qk_{panel}_{half}, "
                "[0, 0], pre_quant=0.0009765625, pre_relu=True)",
            ])
        if panel < 7:
            extract_key(panel + 1)
    reduce_half(7, 1)
    lines.append("    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)")
    return ast.parse("\n".join(lines)).body


def use_indexer_n128(schedule, catalog, output):
    call = next(call for call in schedule["call_bindings"].values()
                if call["logical_function"] == "indexer_score_topk_native_pair")
    transfer = next(arg["value"] for arg in call["arguments"]
                    if arg["parameter"] == "buf_score_transfer_inline8845__ssa_v0")
    if transfer["shape"] != [288, 1024]:
        raise ValueError("本候选只接长历史S6的1024分数块")
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    fn = next(node for node in ast.walk(module)
              if isinstance(node, ast.FunctionDef) and node.name == "indexer_score_topk_native_pair_aic")
    item = next(node for node in fn.body if isinstance(node, ast.For)
                and ast.unparse(node.target).startswith("buf_item"))
    leaf = item.body[-1]
    if not isinstance(leaf, ast.If) or len(leaf.body) != 10:
        raise ValueError("冻结Indexer叶计算结构改变")
    expected_targets = ["buf_query_vector_inline8886__ssa_v0", "buf_query_left_inline8850__ssa_v0"]
    if [leaf.body[i].target.id for i in (4, 6)] != expected_targets:
        raise ValueError("原Q加载或Left绑定改变")
    before_drain = ast.dump(leaf.body[-1])
    leaf.body = leaf.body[:5] + n128_math() + [leaf.body[-1]]
    if ast.dump(leaf.body[-1]) != before_drain:
        raise ValueError("改变了原FFTS排空")
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "core_bodies_indexer_n128.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    "# Indexer N128/逐query WS候选，精度与性能需设备验证。\n"
                    + ast.unparse(module) + "\n")
    audit = {"kernel": fn.name, "query_group": 6, "old_n": 64, "new_n": 128,
             "old_ws_k": 384, "new_ws_k": 192, "ws_per_panel": 2,
             "qk_m": 192, "qk_per_panel": 2, "next_key_prefetch": "L1",
             "qk_int8_acc32": True, "fixpipe_quant_relu_unchanged": True,
             "ffts_step_count_unchanged": True, "leaf_partition_unchanged": True,
             "topk_aiv_unchanged": True, "root_and_gm_scratch_unchanged": True,
             "theoretical_live_bytes": {"left": 61440, "right": 65536, "acc": 106496},
             "capacity_check_scope": "理论同时存活，不替代实际规划和CCEC编译",
             "exact_equivalence_validated": False}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
