# SPDX-License-Identifier: Apache-2.0
"""N64循环流水：固定L1槽位接收DPS返回值，Right只在本轮内使用。

与旧rolled候选区别：不携带Right Tile或新建的score Tile过循环回边。
Key以三槽L1预取，score以两槽L1原位写入；先QK当前块，再WS上一块。
保留M384/N64/K128和WS K384、原叶计划、1024分数交接与AIV Top-K。
该写法改变Right预取提前量，实际分配/同步和整层性能都需要独立验证。
"""

import ast
import copy
import json
from pathlib import Path


def ring_math():
    lines = [
        "for ring_step in pl.range(buf_score_iters_inline8863__ssa_v0):",
        "    if ring_step >= 2:",
        "        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)",
        "    ring_begin = buf_logical_begin_inline8767__ssa_v1 + ring_step * 1024",
        "    ring_transfer = buf_worker_inline8783__ssa_v0 * 12 + ring_step % 2 * 6",
        "    ring_key_pool = pl.tile.create([192, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)",
        "    ring_score_pool = pl.tile.create([384, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)",
    ]

    def pages(indent, panel, prefix):
        # 真实Native页地址与旧N64体一致。L1槽是本候选的私有组织方式。
        for page in range(2):
            tag = f"{prefix}_{page}"
            code = [
                f"{tag}_logical = pl.min((ring_begin + ({panel}) * 64 + {page * 32}) // 32, "
                "pl.max((buf_cache_len_inline8809__ssa_v0 - 1) // 32, 0))",
                f"{tag}_physical = pl.max(pl.cast(pl.tensor.read(idx_block_table__ssa_v0, "
                f"[buf_batch_idx_inline8687__ssa_v0, {tag}_logical]), pl.INDEX), 0)",
                f"{tag}_byte = {tag}_physical * native_page_bytes_inline8726__ssa_v0",
                f"if {tag}_byte % 128 == 0:",
                f"    ring_key_pool = pl.tile.gather_row(ring_key_pool, idx_kv_cache_inline8854__ssa_v0, "
                f"[({panel}) % 3 * 64 + {page * 32}, 0], [{tag}_byte // 128, 0], [32, 128], transpose=False)",
                "else:",
                f"    ring_key_pool = pl.tile.gather_row(ring_key_pool, idx_kv_cache_shift64_inline8790__ssa_v0, "
                f"[({panel}) % 3 * 64 + {page * 32}, 0], [{tag}_byte // 128, 0], [32, 128], transpose=False)",
            ]
            lines.extend(" " * indent + line for line in code)

    pages(4, "0", "ring_prime")
    pages(4, "1", "ring_second")
    lines.extend([
        "    ring_key0 = pl.tile.extract(pl.tile.transpose_view(ring_key_pool), 0, 0, [128, 64], target_memory=pl.Mem.Right)",
        "    ring_qk0 = pl.tile.matmul(buf_query_left_inline8850__ssa_v0, ring_key0)",
        "    ring_score_pool = pl.tile.assemble(ring_score_pool, ring_qk0, [0, 0], pre_quant=0.0009765625, pre_relu=True)",
        "    for ring_panel in pl.range(1, 15):",
        "        ring_current_key = pl.tile.extract(pl.tile.transpose_view(ring_key_pool), 0, ring_panel % 3 * 64, [128, 64], target_memory=pl.Mem.Right)",
    ])
    pages(8, "ring_panel + 1", "ring_next")
    lines.extend([
        "        ring_ws_right = pl.tile.extract(ring_score_pool, 0, (ring_panel - 1) % 2 * 64, [384, 64], target_memory=pl.Mem.Right)",
        "        ring_qk = pl.tile.matmul(buf_query_left_inline8850__ssa_v0, ring_current_key)",
        "        ring_ws = pl.tile.matmul(buf_coefficient_pair_inline8676__ssa_v0, ring_ws_right)",
        "        ring_score_pool = pl.tile.assemble(ring_score_pool, ring_qk, [0, ring_panel % 2 * 64], pre_quant=0.0009765625, pre_relu=True)",
        "        pl.tile.store(pl.tile.set_validshape(ring_ws, 6, 64), [ring_transfer, (ring_panel - 1) * 64], buf_score_transfer_inline8845__ssa_v0)",
        # 末轮不再预取：显式剥离，避免“更新池/原样沿用池”合流插入Mat自拷贝。
        "    ring_final_key = pl.tile.extract(pl.tile.transpose_view(ring_key_pool), 0, 0, [128, 64], target_memory=pl.Mem.Right)",
        "    ring_final_ws_right = pl.tile.extract(ring_score_pool, 0, 0, [384, 64], target_memory=pl.Mem.Right)",
        "    ring_final_qk = pl.tile.matmul(buf_query_left_inline8850__ssa_v0, ring_final_key)",
        "    ring_final_ws = pl.tile.matmul(buf_coefficient_pair_inline8676__ssa_v0, ring_final_ws_right)",
        "    ring_score_pool = pl.tile.assemble(ring_score_pool, ring_final_qk, [0, 64], pre_quant=0.0009765625, pre_relu=True)",
        "    pl.tile.store(pl.tile.set_validshape(ring_final_ws, 6, 64), [ring_transfer, 896], buf_score_transfer_inline8845__ssa_v0)",
        "    ring_last_right = pl.tile.extract(ring_score_pool, 0, 64, [384, 64], target_memory=pl.Mem.Right)",
        "    ring_last_ws = pl.tile.matmul(buf_coefficient_pair_inline8676__ssa_v0, ring_last_right)",
        "    pl.tile.store(pl.tile.set_validshape(ring_last_ws, 6, 64), [ring_transfer, 960], buf_score_transfer_inline8845__ssa_v0)",
        "    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)",
    ])
    return ast.parse("\n".join(lines)).body


def use_l1_ring(schedule, catalog, output):
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("N64固定L1环先验证T96")
    call = next(c for c in schedule["call_bindings"].values()
                if c["logical_function"] == "indexer_score_topk_native_pair")
    transfer = next(a["value"] for a in call["arguments"]
                    if a["parameter"] == "buf_score_transfer_inline8845__ssa_v0")
    if transfer["shape"] != [288, 1024]:
        raise ValueError("要求原1024分数交接布局")
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    functions = {n.name: n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}
    before = {name: ast.dump(fn) for name, fn in functions.items()}
    fn = functions["indexer_score_topk_native_pair_aic"]
    item = next(n for n in fn.body if isinstance(n, ast.For)
                and ast.unparse(n.target).startswith("buf_item"))
    leaf = item.body[-1]
    if (not isinstance(leaf, ast.If) or len(leaf.body) != 10
            or ast.unparse(leaf.body[6].target) != "buf_query_left_inline8850__ssa_v0"
            or ast.unparse(leaf.body[7].target) != "buf_coefficient_pair_inline8676__ssa_v0"):
        raise ValueError("原Indexer叶结构改变")
    prefix, drain = [ast.dump(n) for n in leaf.body[:8]], ast.dump(leaf.body[-1])
    leaf.body = leaf.body[:8] + ring_math() + leaf.body[-1:]
    assert [ast.dump(n) for n in leaf.body[:8]] == prefix and ast.dump(leaf.body[-1]) == drain
    changed = [name for name, value in functions.items() if ast.dump(value) != before[name]]
    assert changed == [fn.name]
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "core_bodies_indexer_l1_ring.py"
    target.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      + ast.unparse(module) + "\n")
    audit = {"changed_functions": changed, "qk_mnk": [384, 64, 128],
             "ws_mnk": [16, 64, 384], "runtime_panels_per_1024": 16,
             "key_l1_slots": 3, "score_l1_slots": 2,
             "key_l1_bytes": 24576, "score_l1_bytes": 98304,
             "loop_right_tiles_are_local": True, "dps_return_values_received": True,
             "last_qk_peeled_to_avoid_optional_key_pool_update": True,
             "original_leaf_plan_and_1024_transfer_and_aiv_and_drain_unchanged": True,
             "physical_codegen_checked": False, "device_validated": False,
             "warning": "Right预取提前量与原展开不同，不能从源顺序推出性能"}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(target)
    return result, audit
