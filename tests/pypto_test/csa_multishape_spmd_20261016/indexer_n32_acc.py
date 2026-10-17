# SPDX-License-Identifier: Apache-2.0
"""Indexer N32双QK累加区：保留M384与WS K384，仅沿候选列拆分。

原N64的一个QK输出占96KiB，L0C无法同时保留两个。本候选每份48KiB，
先提交下一面板QK，再把当前QK通过原FP16 FIX/ReLU写入L1并执行WS。
32面板在源码中展开，不携带Right或Mat Tile跨循环。数学调用数翻倍，
因此容量满足/存在双缓冲不等于更快，必须以完整CSA无插桩L1验证。
"""

import ast
import copy
import json
from pathlib import Path


def score_math():
    lines = [
        "for n32_step in pl.range(buf_score_iters_inline8863__ssa_v0):",
        "    if n32_step >= 2:",
        "        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)",
        "    n32_begin = buf_logical_begin_inline8767__ssa_v1 + n32_step * 1024",
        "    n32_transfer = buf_worker_inline8783__ssa_v0 * 12 + n32_step % 2 * 6",
        "    n32_pool = pl.tile.create([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)",
    ]

    def load_key(panel):
        prefix = f"n32_p{panel}"
        row = panel % 4 * 32
        lines.extend([
            f"    {prefix}_logical = pl.min((n32_begin + {panel * 32}) // 32, "
            "pl.max((buf_cache_len_inline8809__ssa_v0 - 1) // 32, 0))",
            f"    {prefix}_physical = pl.max(pl.cast(pl.tensor.read(idx_block_table__ssa_v0, "
            f"[buf_batch_idx_inline8687__ssa_v0, {prefix}_logical]), pl.INDEX), 0)",
            f"    {prefix}_byte = {prefix}_physical * native_page_bytes_inline8726__ssa_v0",
            f"    if {prefix}_byte % 128 == 0:",
            f"        n32_pool = pl.tile.gather_row(n32_pool, idx_kv_cache_inline8854__ssa_v0, "
            f"[{row}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
            "    else:",
            f"        n32_pool = pl.tile.gather_row(n32_pool, idx_kv_cache_shift64_inline8790__ssa_v0, "
            f"[{row}, 0], [{prefix}_byte // 128, 0], [32, 128], transpose=False)",
            f"    n32_key_{panel} = pl.tile.extract(pl.tile.transpose_view(n32_pool), "
            f"0, {row}, [128, 32], target_memory=pl.Mem.Right)",
        ])

    load_key(0)
    load_key(1)
    lines.append("    n32_qk_0 = pl.tile.matmul(buf_query_left_inline8850__ssa_v0, n32_key_0)")
    for panel in range(32):
        # 当前和下一QK的Acc须同时存活，给Cube与前一块FIX留下重叠机会。
        if panel < 31:
            lines.append(f"    n32_qk_{panel+1} = pl.tile.matmul(buf_query_left_inline8850__ssa_v0, n32_key_{panel+1})")
        lines.extend([
            f"    n32_blank_{panel} = pl.tile.create([384, 32], dtype=pl.FP16, target_memory=pl.Mem.Mat)",
            f"    n32_score_{panel} = pl.tile.assemble(n32_blank_{panel}, n32_qk_{panel}, "
            "[0, 0], pre_quant=0.0009765625, pre_relu=True)",
        ])
        if panel < 30:
            load_key(panel + 2)
        lines.extend([
            f"    n32_ws_right_{panel} = pl.tile.extract(n32_score_{panel}, 0, 0, [384, 32], target_memory=pl.Mem.Right)",
            f"    n32_ws_{panel} = pl.tile.matmul(buf_coefficient_pair_inline8676__ssa_v0, n32_ws_right_{panel})",
            f"    pl.tile.store(pl.tile.set_validshape(n32_ws_{panel}, 6, 32), "
            f"[n32_transfer, {panel * 32}], buf_score_transfer_inline8845__ssa_v0)",
        ])
    lines.append("    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)")
    return ast.parse("\n".join(lines)).body


def use_n32_acc(schedule, catalog, output):
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("N32双QK先验证T96长历史")
    call = next(c for c in schedule["call_bindings"].values()
                if c["logical_function"] == "indexer_score_topk_native_pair")
    transfer = next(a["value"] for a in call["arguments"]
                    if a["parameter"] == "buf_score_transfer_inline8845__ssa_v0")
    if transfer["shape"] != [288, 1024]:
        raise ValueError("N32双QK要求原1024分数交接布局")
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
        raise ValueError("原Indexer叶计算结构改变")
    prefix, drain = [ast.dump(n) for n in leaf.body[:8]], ast.dump(leaf.body[-1])
    leaf.body = leaf.body[:8] + score_math() + leaf.body[-1:]
    assert [ast.dump(n) for n in leaf.body[:8]] == prefix
    assert ast.dump(leaf.body[-1]) == drain
    changed = [name for name, value in functions.items() if ast.dump(value) != before[name]]
    assert changed == [fn.name]
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "core_bodies_indexer_n32_acc.py"
    target.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      + ast.unparse(module) + "\n")
    audit = {"changed_functions": changed, "qk_mnk": [384, 32, 128],
             "ws_mnk": [16, 32, 384], "panels_per_1024": 32,
             "qk_acc_bytes_each": 49152, "next_qk_submitted_before_current_fix": True,
             "query_and_coefficient_left_bytes": 61440,
             "qk_dot_int32_max_abs_bound": 128 * 128 * 128,
             "leaf_plan_and_root_abi_unchanged": True,
             "original_1024_transfer_and_ffts_drain_unchanged": True,
             "aiv_topk_body_unchanged": True,
             "actual_allocation_checked": False, "device_validated": False}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(target)
    return result, audit
