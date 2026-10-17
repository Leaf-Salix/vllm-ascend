# SPDX-License-Identifier: Apache-2.0
"""原N64的16个展开面板改为首块剥离、15次核内循环、末块WS排空。

当前为保留的CPU失败候选：v1遇到同地址Mat move，v2的循环Key携带被
转换成A3不支持的Right→Right move。未接设备driver，不可用于性能结论。

不缩小QK/WS的数学Tile，不改页地址与FFTS。循环携带当前Key Right和上一
FP16 score；下一Key的L1临时区在当前迭代中完整写入再提取，不需跨迭代携带。
提前取下一Key后才计算当前QK，地址是否真实双缓冲
必须另查实际IR/C++，不能用源码的变量名保证不别名。
"""

import ast
import copy
import json
from pathlib import Path


def rolled_math():
    lines = [
        "for roll_step in pl.range(buf_score_iters_inline8863__ssa_v0):",
        "    if roll_step >= 2:",
        "        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)",
        "    roll_begin = buf_logical_begin_inline8767__ssa_v1 + roll_step * 1024",
        "    roll_transfer = buf_worker_inline8783__ssa_v0 * 12 + roll_step % 2 * 6",
        "    roll_pool = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)",
    ]

    def pages(indent, panel, pool, output):
        text = []
        for page in range(2):
            prefix = output + str(page)
            text.extend([
                f"{prefix}_logical = pl.min((roll_begin + ({panel}) * 64 + {page * 32}) // 32, "
                "pl.max((buf_cache_len_inline8809__ssa_v0 - 1) // 32, 0))",
                f"{prefix}_physical = pl.max(pl.cast(pl.tensor.read(idx_block_table__ssa_v0, "
                f"[buf_batch_idx_inline8687__ssa_v0, {prefix}_logical]), pl.INDEX), 0)",
                f"{prefix}_byte = {prefix}_physical * native_page_bytes_inline8726__ssa_v0",
                f"if {prefix}_byte % 128 == 0:",
                f"    {pool} = pl.tile.gather_row({pool}, idx_kv_cache_inline8854__ssa_v0, "
                f"[({panel}) % 2 * 64 + {page * 32}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
                "else:",
                f"    {pool} = pl.tile.gather_row({pool}, idx_kv_cache_shift64_inline8790__ssa_v0, "
                f"[({panel}) % 2 * 64 + {page * 32}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
            ])
        lines.extend(" " * indent + t for t in text)

    pages(4, "0", "roll_pool", "roll_prime")
    lines.append("    roll_key0 = pl.tile.extract(pl.tile.transpose_view(roll_pool), 0, 0, [128, 64], target_memory=pl.Mem.Right)")
    pages(4, "1", "roll_pool", "roll_second")
    lines.extend([
        "    roll_key1 = pl.tile.extract(pl.tile.transpose_view(roll_pool), 0, 64, [128, 64], target_memory=pl.Mem.Right)",
        "    roll_qk0 = pl.tile.matmul(buf_query_left_inline8850__ssa_v0, roll_key0)",
        "    roll_score0_empty = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)",
        "    roll_score0 = pl.tile.assemble(roll_score0_empty, roll_qk0, [0, 0], pre_quant=0.0009765625, pre_relu=True)",
        "    for roll_panel, (roll_key_iter, roll_score_iter) in pl.range(1, 16, init_values=(roll_key1, roll_score0)):",
        "        if roll_panel + 1 < 16:",
        "            roll_next_pool = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)",
    ])
    pages(12, "roll_panel + 1", "roll_next_pool", "roll_next")
    lines.extend([
        "            roll_prefetched = pl.tile.extract(pl.tile.transpose_view(roll_next_pool), 0, (roll_panel + 1) % 2 * 64, [128, 64], target_memory=pl.Mem.Right)",
        "            roll_key_phi = pl.yield_(roll_prefetched)",
        "        else:",
        "            roll_key_phi = pl.yield_(roll_key_iter)",
        "        roll_ws_right = pl.tile.move(roll_score_iter, target_memory=pl.Mem.Right)",
        "        roll_qk = pl.tile.matmul(buf_query_left_inline8850__ssa_v0, roll_key_iter)",
        "        roll_ws = pl.tile.matmul(buf_coefficient_pair_inline8676__ssa_v0, roll_ws_right)",
        "        roll_score_empty = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)",
        "        roll_score = pl.tile.assemble(roll_score_empty, roll_qk, [0, 0], pre_quant=0.0009765625, pre_relu=True)",
        "        pl.tile.store(pl.tile.set_validshape(roll_ws, 6, 64), [roll_transfer, (roll_panel - 1) * 64], buf_score_transfer_inline8845__ssa_v0)",
        "        roll_key_last, roll_score_last = pl.yield_(roll_key_phi, roll_score)",
        "    roll_last_right = pl.tile.move(roll_score_last, target_memory=pl.Mem.Right)",
        "    roll_last_ws = pl.tile.matmul(buf_coefficient_pair_inline8676__ssa_v0, roll_last_right)",
        "    pl.tile.store(pl.tile.set_validshape(roll_last_ws, 6, 64), [roll_transfer, 960], buf_score_transfer_inline8845__ssa_v0)",
        "    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)",
    ])
    return ast.parse("\n".join(lines)).body


def use_rolled_panels(schedule, catalog, output):
    call = next(c for c in schedule["call_bindings"].values()
                if c["logical_function"] == "indexer_score_topk_native_pair")
    transfer = next(a["value"] for a in call["arguments"]
                    if a["parameter"] == "buf_score_transfer_inline8845__ssa_v0")
    if transfer["shape"] != [288, 1024]:
        raise ValueError("循环面板只适用于长历史S6原N64")
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    before = {n.name: ast.dump(n) for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}
    fn = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
              and n.name == "indexer_score_topk_native_pair_aic")
    item = next(n for n in fn.body if isinstance(n, ast.For)
                and ast.unparse(n.target).startswith("buf_item"))
    leaf = item.body[-1]
    if (not isinstance(leaf, ast.If) or len(leaf.body) != 10
            or ast.unparse(leaf.body[7].target) != "buf_coefficient_pair_inline8676__ssa_v0"
            or not isinstance(leaf.body[8], ast.For)):
        raise ValueError("原叶计算结构改变")
    original_prefix = [ast.dump(n) for n in leaf.body[:8]]
    original_drain = ast.dump(leaf.body[-1])
    leaf.body = leaf.body[:8] + rolled_math() + [leaf.body[-1]]
    assert [ast.dump(n) for n in leaf.body[:8]] == original_prefix
    assert ast.dump(leaf.body[-1]) == original_drain
    changed = [n.name for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
               and ast.dump(n) != before[n.name]]
    assert changed == [fn.name]
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "core_bodies_indexer_rolled_panels.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    + ast.unparse(module) + "\n")
    audit = {"changed_functions": changed, "panel_n": 64, "runtime_panels_per_step": 16,
             "qk_and_ws_static_matmul_sites": 4, "qk_and_ws_runtime_matmuls_per_step": 32,
             "query_coefficient_loads_and_leaf_plan_unchanged": True,
             "ffts_step_count_and_drain_unchanged": True, "aiv_topk_unchanged": True,
             "root_and_gm_scratch_unchanged": True,
             "loop_carried_tiles": ["key_right", "previous_score_l1"],
             "next_key_l1_fully_written_in_same_iteration": True,
             "physical_buffer_disjointness_validated": False, "device_validated": False}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
