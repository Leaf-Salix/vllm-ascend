# SPDX-License-Identifier: Apache-2.0
"""Indexer M192/N64：两份分数同时存活，给Cube/FIX重叠留下L0C空间。

原M384/N64单份INT32分数占96KiB，128KiB Acc容不下第二份。本候选
每组三个query，两份各48KiB；先提交两次QK，再写回第一份，源级活跃
区间要求两份分数不别名。具体地址和同步必须核对生成代码，不能把
源级交错当成实际硬件重叠。叶计划、1024分数块、Vector TopK及FFTS不变。
"""

import ast
import copy
import json
from pathlib import Path


def score_math():
    lines = []
    for half in range(2):
        lines.extend([
            f"pp_query_{half} = pl.tile.extract(buf_query_vector_inline8886__ssa_v0, "
            f"{half * 192}, 0, [192, 128], target_memory=pl.Mem.Left)",
            f"pp_coef_gm_{half} = pl.tile.load(coefficients_inline8733__ssa_v0, "
            f"[buf_query_inline8778__phi_v2 // 6 * 16 + {half * 3}, {half * 192}], "
            "[16, 192], [3, 192], target_memory=pl.Mem.Mat)",
            f"pp_coef_{half} = pl.tile.move(pp_coef_gm_{half}, target_memory=pl.Mem.Left)",
        ])
    lines.extend([
        "for pp_step in pl.range(buf_score_iters_inline8863__ssa_v0):",
        "    if pp_step >= 2:",
        "        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)",
        "    pp_begin = buf_logical_begin_inline8767__ssa_v1 + pp_step * 1024",
        "    pp_transfer = buf_worker_inline8783__ssa_v0 * 12 + pp_step % 2 * 6",
        "    pp_key_pool = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)",
    ])

    def prefetch_key(panel):
        slot = panel % 2 * 64
        for page in range(2):
            prefix = f"pp_p{panel}_page{page}"
            lines.extend([
                f"    {prefix}_logical = pl.min((pp_begin + {panel * 64 + page * 32}) // 32, "
                "pl.max((buf_cache_len_inline8809__ssa_v0 - 1) // 32, 0))",
                f"    {prefix}_physical = pl.max(pl.cast(pl.tensor.read(idx_block_table__ssa_v0, "
                f"[buf_batch_idx_inline8687__ssa_v0, {prefix}_logical]), pl.INDEX), 0)",
                f"    {prefix}_byte = {prefix}_physical * native_page_bytes_inline8726__ssa_v0",
                f"    if {prefix}_byte % 128 == 0:",
                f"        pp_key_pool = pl.tile.gather_row(pp_key_pool, idx_kv_cache_inline8854__ssa_v0, "
                f"[{slot + page * 32}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
                "    else:",
                f"        pp_key_pool = pl.tile.gather_row(pp_key_pool, idx_kv_cache_shift64_inline8790__ssa_v0, "
                f"[{slot + page * 32}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
            ])
        lines.append(f"    pp_key_{panel} = pl.tile.extract(pl.tile.transpose_view(pp_key_pool), "
                     f"0, {slot}, [128, 64], target_memory=pl.Mem.Right)")

    def write_score(panel, half):
        lines.extend([
            f"    pp_score_empty_{panel}_{half} = pl.tile.create([192, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)",
            f"    pp_score_l1_{panel}_{half} = pl.tile.assemble(pp_score_empty_{panel}_{half}, "
            f"pp_qk_{panel}_{half}, [0, 0], pre_quant=0.0009765625, pre_relu=True)",
        ])

    def reduce_half(panel, half):
        prefix = f"pp_ws_{panel}_{half}"
        lines.extend([
            f"    {prefix}_right = pl.tile.move(pp_score_l1_{panel}_{half}, target_memory=pl.Mem.Right)",
            f"    {prefix}_acc = pl.tile.matmul(pp_coef_{half}, {prefix}_right)",
            f"    pl.tile.store(pl.tile.set_validshape({prefix}_acc, 3, 64), "
            f"[pp_transfer + {half * 3}, {panel * 64}], buf_score_transfer_inline8845__ssa_v0)",
        ])

    prefetch_key(0)
    for panel in range(16):
        if panel < 15:
            prefetch_key(panel + 1)
        lines.append(f"    pp_qk_{panel}_0 = pl.tile.matmul(pp_query_0, pp_key_{panel})")
        if panel:
            reduce_half(panel - 1, 1)
        lines.append(f"    pp_qk_{panel}_1 = pl.tile.matmul(pp_query_1, pp_key_{panel})")
        # QK0在QK1之后仍需FIX写回；真实活跃区间阻止两份Acc复用。
        write_score(panel, 0)
        reduce_half(panel, 0)
        write_score(panel, 1)
    reduce_half(15, 1)
    lines.append("    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)")
    return ast.parse("\n".join(lines)).body


def use_score_pingpong(schedule, catalog, output):
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("M192/N64双Acc候选先限定T96长历史")
    call = next(call for call in schedule["call_bindings"].values()
                if call["logical_function"] == "indexer_score_topk_native_pair")
    transfer = next(arg["value"] for arg in call["arguments"]
                    if arg["parameter"] == "buf_score_transfer_inline8845__ssa_v0")
    if transfer["shape"] != [288, 1024]:
        raise ValueError("候选要求原S6长历史1024分数交接块")
    tree = ast.parse(Path(catalog["body_bundle"]).read_text())
    fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
              and node.name == "indexer_score_topk_native_pair_aic")
    item = next(node for node in fn.body if isinstance(node, ast.For)
                and ast.unparse(node.target).startswith("buf_item"))
    leaf = item.body[-1]
    if (not isinstance(leaf, ast.If) or len(leaf.body) != 10
            or leaf.body[4].target.id != "buf_query_vector_inline8886__ssa_v0"
            or leaf.body[6].target.id != "buf_query_left_inline8850__ssa_v0"):
        raise ValueError("原Indexer叶数学体结构改变")
    drain = ast.dump(leaf.body[-1])
    leaf.body = leaf.body[:5] + score_math() + leaf.body[-1:]
    assert ast.dump(leaf.body[-1]) == drain
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "core_bodies_indexer_score_pingpong.py"
    target.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      "# Indexer M192/N64双Acc实验；保留原叶计划及FFTS。\n"
                      + ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    audit = {"kernel": fn.name, "old_qk_tile": [384, 64, 128],
             "new_qk_tile": [192, 64, 128], "query_groups_per_panel": 2,
             "panels_per_score_block": 16, "qk_acc_simultaneously_live_bytes": 98304,
             "query_left_and_coefficient_bytes": 61440,
             "right_upper_bound_bytes": 65536, "acc_upper_bound_bytes": 106496,
             "original_leaf_plan": True, "original_page_clamp": True,
             "original_score_scale_relu_and_gm_layout": True,
             "original_ffts_and_aiv_topk": True, "host_metadata_read": False,
             "precision_proven": False, "hardware_overlap_proven": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(target)
    return result, audit
