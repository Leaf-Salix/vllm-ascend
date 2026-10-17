# SPDX-License-Identifier: Apache-2.0
"""N64面板分组循环：组内展开，组末排空WS，不跨循环携带Right Tile。

只缩减展开代码，不减少每步16次QK/16次WS。组间需要重新启动Key/score
流水，可能抵消代码缩短的收益；不得仅凭代码大小判断性能。
"""

import ast
import copy
import json
from pathlib import Path


def grouped_math(group_size):
    lines = [
        "for gp_step in pl.range(buf_score_iters_inline8863__ssa_v0):",
        "    if gp_step >= 2:",
        "        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)",
        "    gp_begin = buf_logical_begin_inline8767__ssa_v1 + gp_step * 1024",
        "    gp_transfer = buf_worker_inline8783__ssa_v0 * 12 + gp_step % 2 * 6",
        f"    for gp_group in pl.range({16 // group_size}):",
        f"        gp_base = gp_group * {group_size * 64}",
        "        gp_pool = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)",
    ]

    def load_key(panel):
        for page in range(2):
            prefix = f"gp_p{panel}_page{page}"
            lines.extend([
                f"        {prefix}_logical = pl.min((gp_begin + gp_base + {panel * 64 + page * 32}) // 32, "
                "pl.max((buf_cache_len_inline8809__ssa_v0 - 1) // 32, 0))",
                f"        {prefix}_physical = pl.max(pl.cast(pl.tensor.read(idx_block_table__ssa_v0, "
                f"[buf_batch_idx_inline8687__ssa_v0, {prefix}_logical]), pl.INDEX), 0)",
                f"        {prefix}_byte = {prefix}_physical * native_page_bytes_inline8726__ssa_v0",
                f"        if {prefix}_byte % 128 == 0:",
                f"            gp_pool = pl.tile.gather_row(gp_pool, idx_kv_cache_inline8854__ssa_v0, "
                f"[{panel % 2 * 64 + page * 32}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
                "        else:",
                f"            gp_pool = pl.tile.gather_row(gp_pool, idx_kv_cache_shift64_inline8790__ssa_v0, "
                f"[{panel % 2 * 64 + page * 32}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
            ])
        lines.append(f"        gp_key_{panel} = pl.tile.extract(pl.tile.transpose_view(gp_pool), "
                     f"0, {panel % 2 * 64}, [128, 64], target_memory=pl.Mem.Right)")

    def ws_load(panel):
        lines.append(f"        gp_ws_right_{panel} = pl.tile.extract(gp_score_{panel}, "
                     "0, 0, [384, 64], target_memory=pl.Mem.Right)")

    def ws_matmul(panel):
        lines.append(f"        gp_ws_{panel} = pl.tile.matmul(buf_coefficient_pair_inline8676__ssa_v0, gp_ws_right_{panel})")

    def ws_store(panel):
        lines.append(f"        pl.tile.store(pl.tile.set_validshape(gp_ws_{panel}, 6, 64), "
                     f"[gp_transfer, gp_base + {panel * 64}], buf_score_transfer_inline8845__ssa_v0)")

    load_key(0)
    for panel in range(group_size):
        if panel + 1 < group_size:
            load_key(panel + 1)
        if panel:
            ws_load(panel - 1)
        lines.append(f"        gp_qk_{panel} = pl.tile.matmul(buf_query_left_inline8850__ssa_v0, gp_key_{panel})")
        if panel:
            ws_matmul(panel - 1)
        lines.extend([
            f"        gp_empty_{panel} = pl.tile.create([384, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)",
            f"        gp_score_{panel} = pl.tile.assemble(gp_empty_{panel}, gp_qk_{panel}, "
            "[0, 0], pre_quant=0.0009765625, pre_relu=True)",
        ])
        if panel:
            ws_store(panel - 1)
    ws_load(group_size - 1)
    ws_matmul(group_size - 1)
    ws_store(group_size - 1)
    lines.append("    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)")
    return ast.parse("\n".join(lines)).body


def use_panel_groups(schedule, catalog, output, group_size):
    if group_size not in (2, 4, 8):
        raise ValueError("N64面板组仅允许2/4/8；其他取值应使用原展开版")
    call = next(c for c in schedule["call_bindings"].values()
                if c["logical_function"] == "indexer_score_topk_native_pair")
    transfer = next(a["value"] for a in call["arguments"]
                    if a["parameter"] == "buf_score_transfer_inline8845__ssa_v0")
    if transfer["shape"] != [288, 1024]:
        raise ValueError("分组循环只适用于长历史S6原N64")
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
    prefix = [ast.dump(n) for n in leaf.body[:8]]
    drain = ast.dump(leaf.body[-1])
    leaf.body = leaf.body[:8] + grouped_math(group_size) + [leaf.body[-1]]
    assert [ast.dump(n) for n in leaf.body[:8]] == prefix
    assert ast.dump(leaf.body[-1]) == drain
    changed = [n.name for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
               and ast.dump(n) != before[n.name]]
    assert changed == [fn.name]
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "core_bodies_indexer_panel_groups.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    + ast.unparse(module) + "\n")
    audit = {"changed_functions": changed, "panel_n": 64, "panel_group_size": group_size,
             "groups_per_step": 16 // group_size, "qk_ws_static_matmul_sites": group_size * 2,
             "qk_ws_runtime_matmuls_per_step": 32, "loop_carried_tiles": [],
             "query_coefficient_loads_and_leaf_plan_unchanged": True,
             "ffts_step_count_and_drain_unchanged": True, "aiv_topk_unchanged": True,
             "root_and_gm_scratch_unchanged": True, "group_boundaries_drain_ws": True,
             "physical_buffer_disjointness_validated": False, "device_validated": False}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
