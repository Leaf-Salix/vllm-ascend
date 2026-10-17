# SPDX-License-Identifier: Apache-2.0
"""Indexer M384/N80：保持六query和WS K384，1024列改为13个panel。

QK Acc占120KiB，WS Acc占5KiB；不以缩短K或拆query改变归约。
WS Right占60KiB，不能与10KiB的Key Right同时存活。只把下个Key
预取到L1，当前QK消费Key后才取前一panel的WS Right。此约束可能抵消
减少panel的收益，必须检查实际规划和完整L1，不预记性能改善。
"""

import ast
import copy
import json
from pathlib import Path


def n80_math():
    lines = [
        "n80_query = pl.tile.move(buf_query_vector_inline8886__ssa_v0, target_memory=pl.Mem.Left)",
        "n80_coef_l1 = pl.tile.load(coefficients_inline8733__ssa_v0, "
        "[buf_query_inline8778__phi_v2 // 6 * 16, 0], [16, 384], [16, 384], target_memory=pl.Mem.Mat)",
        "n80_coef = pl.tile.move(n80_coef_l1, target_memory=pl.Mem.Left)",
        "for n80_step in pl.range(buf_score_iters_inline8863__ssa_v0):",
        "    if n80_step >= 2:",
        "        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)",
        "    n80_begin = buf_logical_begin_inline8767__ssa_v1 + n80_step * 1024",
        "    n80_transfer = buf_worker_inline8783__ssa_v0 * 12 + n80_step % 2 * 6",
        "    n80_pool = pl.tile.create([160, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)",
    ]

    def load_key(panel):
        # Native的32行页含尾部scale；80列panel交替从页内0/16行开始。
        # 每段都只访问单页，不把scale字节当成Key；沿用两份0/64字节偏移视图。
        cursor = 0
        while cursor < 80:
            offset = (panel * 80 + cursor) % 32
            count = min(32 - offset, 80 - cursor)
            prefix = f"n80_p{panel}_r{cursor}"
            lines.extend([
                f"    {prefix}_logical = pl.min((n80_begin + {panel * 80 + cursor}) // 32, "
                "pl.max((buf_cache_len_inline8809__ssa_v0 - 1) // 32, 0))",
                f"    {prefix}_physical = pl.max(pl.cast(pl.tensor.read(idx_block_table__ssa_v0, "
                f"[buf_batch_idx_inline8687__ssa_v0, {prefix}_logical]), pl.INDEX), 0)",
                f"    {prefix}_byte = {prefix}_physical * native_page_bytes_inline8726__ssa_v0 + {offset * 128}",
                f"    if {prefix}_byte % 128 == 0:",
                f"        n80_pool = pl.tile.gather_row(n80_pool, idx_kv_cache_inline8854__ssa_v0, "
                f"[{panel % 2 * 80 + cursor}, 0], [{prefix}_byte // 128, 0], [{count}, 128], transpose=False)",
                "    else:",
                f"        n80_pool = pl.tile.gather_row(n80_pool, idx_kv_cache_shift64_inline8790__ssa_v0, "
                f"[{panel % 2 * 80 + cursor}, 0], [{prefix}_byte // 128, 0], [{count}, 128], transpose=False)",
            ])
            cursor += count

    def extract_key(panel):
        lines.append(f"    n80_key_{panel} = pl.tile.extract(pl.tile.transpose_view(n80_pool), "
                     f"0, {panel % 2 * 80}, [128, 80], target_memory=pl.Mem.Right)")

    def reduce_score(panel):
        columns = min(80, 1024 - panel * 80)
        lines.extend([
            f"    n80_ws_right_{panel} = pl.tile.move(n80_score_{panel}, target_memory=pl.Mem.Right)",
            f"    n80_ws_{panel} = pl.tile.matmul(n80_coef, n80_ws_right_{panel})",
        ])
        return [
            f"    pl.tile.store(pl.tile.set_validshape(n80_ws_{panel}, 6, {columns}), "
            f"[n80_transfer, {panel * 80}], buf_score_transfer_inline8845__ssa_v0)",
        ]

    load_key(0)
    for panel in range(13):
        extract_key(panel)
        if panel < 12:
            load_key(panel + 1)
        lines.append(f"    n80_qk_{panel} = pl.tile.matmul(n80_query, n80_key_{panel})")
        stores = reduce_score(panel - 1) if panel else []
        lines.extend([
            f"    n80_empty_{panel} = pl.tile.create([384, 80], dtype=pl.FP16, target_memory=pl.Mem.Mat)",
            f"    n80_score_{panel} = pl.tile.assemble(n80_empty_{panel}, n80_qk_{panel}, "
            "[0, 0], pre_quant=0.0009765625, pre_relu=True)",
            *stores,
        ])
    lines.extend(reduce_score(12))
    lines.append("    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)")
    return ast.parse("\n".join(lines)).body


def use_indexer_n80(schedule, catalog, output):
    call = next(c for c in schedule["call_bindings"].values()
                if c["logical_function"] == "indexer_score_topk_native_pair")
    transfer = next(a["value"] for a in call["arguments"]
                    if a["parameter"] == "buf_score_transfer_inline8845__ssa_v0")
    if transfer["shape"] != [288, 1024]:
        raise ValueError("N80仅接长历史S6的原1024列分数块")
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    function = next(n for n in ast.walk(module)
                    if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_topk_native_pair_aic")
    item = next(n for n in function.body if isinstance(n, ast.For)
                and ast.unparse(n.target).startswith("buf_item"))
    leaf = item.body[-1]
    if not isinstance(leaf, ast.If) or len(leaf.body) != 10:
        raise ValueError("冻结Indexer叶结构改变")
    if leaf.body[4].target.id != "buf_query_vector_inline8886__ssa_v0":
        raise ValueError("冻结Query加载改变")
    leaf.body = leaf.body[:5] + n80_math() + [leaf.body[-1]]
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "core_bodies_indexer_n80.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    "# Indexer N80实验；精度与性能以设备验证为准。\n" + ast.unparse(module) + "\n")
    audit = {"kernel": function.name, "old_n": 64, "new_n": 80,
             "panel_count": 13, "last_stored_columns": 64, "qk_m": 384, "ws_k": 384,
             "qk_and_ws_matmuls_per_1024": 26, "native_page_segments_per_1024": 39,
             "query_and_ws_reduction_order_unchanged": True,
             "fixpipe_quant_relu_unchanged": True, "ffts_and_leaf_plan_unchanged": True,
             "topk_aiv_unchanged": True, "root_and_gm_scratch_unchanged": True,
             "theoretical_live_bytes": {"left": 61440, "right_peak": 61440, "acc": 128000},
             "key_and_ws_right_must_not_overlap": True,
             "exact_equivalence_validated": False}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(path)
    return result, audit
