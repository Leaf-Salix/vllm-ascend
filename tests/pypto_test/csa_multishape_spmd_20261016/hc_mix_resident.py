# SPDX-License-Identifier: Apache-2.0
"""HC mix/norm 整行驻留候选：省去 BF16 mixed 的 GM 回读。

每个成员处理相邻两行，T96 使用48个AIV。四路混合保持 (y0+y1)+(y2+y3)，
先转BF16再计算RMS。整行平方重排为16×256，仍做16个K256行归约，
按原0..15顺序做Vector累加；inverse继续使用原1×8 Vector rsqrt。
所有Scalar取值来自device GM/UB，不添加Host取值。
"""

import ast
import copy
import json
from pathlib import Path


def resident_math():
    lines = [
        "hc_mix_weight_1d = pl.tile.load(attn_norm_w__ssa_v0, [0], [4096], [4096], target_memory=pl.Mem.Vec)",
        "hc_mix_weight = pl.tile.reshape(hc_mix_weight_1d, [1, 4096])",
        "hc_mix_weight_f32 = pl.tile.cast(hc_mix_weight, pl.FP32)",
        "for hc_mix_row in pl.range(original_block_idx * 2, pl.min(original_block_idx * 2 + 2, t_dim_inline529__ssa_v0)):",
    ]
    for i in range(4):
        lines.extend([
            f"    hc_mix_gate{i} = pl.tensor.read(pre_val_store__rv_v2, [hc_mix_row, {i}])",
            f"    hc_mix_x{i} = pl.tile.load(x_flat_inline525__ssa_v0, [hc_mix_row, {i * 4096}], "
            "[1, 4096], [1, 4096], target_memory=pl.Mem.Vec)",
            f"    hc_mix_y{i} = pl.tile.muls(hc_mix_x{i}, hc_mix_gate{i})",
        ])
    lines.extend([
        "    hc_mix_y01 = pl.tile.add(hc_mix_y0, hc_mix_y1)",
        "    hc_mix_y23 = pl.tile.add(hc_mix_y2, hc_mix_y3)",
        "    hc_mix_y = pl.tile.add(hc_mix_y01, hc_mix_y23)",
        "    hc_mix_bf16 = pl.tile.cast(hc_mix_y, pl.BF16, mode=\"rint\")",
        # 原内部GM结果仍写出；真正删除的是后续加载，避免把两种变化混在一起。
        "    pl.tile.store(hc_mix_bf16, [hc_mix_row, 0], x_mixed_inline534__ssa_v0)",
        "    hc_mix_rounded = pl.tile.cast(hc_mix_bf16, pl.FP32)",
        "    hc_mix_sq = pl.tile.mul(hc_mix_rounded, hc_mix_rounded)",
        "    hc_mix_chunks = pl.tile.reshape(hc_mix_sq, [16, 256])",
        "    hc_mix_sum_tmp = pl.tile.create([16, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)",
        "    hc_mix_parts = pl.tile.row_sum(hc_mix_chunks, hc_mix_sum_tmp)",
        "    hc_mix_sum_zero = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)",
        "    hc_mix_sum = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)",
        "    for hc_mix_k in pl.range(16):",
        "        hc_mix_piece = pl.tile.read(hc_mix_parts, [hc_mix_k, 0])",
        "        hc_mix_piece_tile = pl.tile.adds(hc_mix_sum_zero, hc_mix_piece)",
        "        hc_mix_sum = pl.tile.add(hc_mix_sum, hc_mix_piece_tile)",
        "    hc_mix_mean = pl.tile.muls(hc_mix_sum, 0.000244140625)",
        "    hc_mix_variance = pl.tile.adds(hc_mix_mean, 1e-06)",
        "    hc_mix_inv_tmp = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)",
        "    hc_mix_inverse = pl.tile.rsqrt(hc_mix_variance, hc_mix_inv_tmp)",
        "    hc_mix_inv = pl.tile.read(hc_mix_inverse, [0, 0])",
        "    hc_mix_scaled = pl.tile.muls(hc_mix_rounded, hc_mix_inv)",
        "    hc_mix_normed = pl.tile.mul(hc_mix_scaled, hc_mix_weight_f32)",
        "    hc_mix_result = pl.tile.cast(hc_mix_normed, pl.BF16, mode=\"rint\")",
        "    pl.tile.store(hc_mix_result, [hc_mix_row, 0], x_normed_t__ssa_v0)",
    ])
    return ast.parse("\n".join(lines)).body


def use_resident_mix(schedule, catalog, output):
    tokens = schedule["root_arguments"]["x_hc"]["shape"][0]
    if tokens != 96:
        raise ValueError("整行驻留先限B16/S6；其他档单独选择成员数后验证")
    module = ast.parse(Path(catalog["body_bundle"]).read_text())
    fn = next(n for n in ast.walk(module)
              if isinstance(n, ast.FunctionDef) and n.name == "mix_x_rms_norm")
    text = ast.unparse(fn)
    for contract in ["mix_db_inline518__idx_v0", "norm_db_inline517__idx_v0",
                     "pl.pipeline(16, stage=2", "mix_db_inline518__idx_v0 * 256",
                     "pl.Tile[[8, 256], pl.FP32",
                     "pl.tile.rsqrt(", "0.000244140625", "pl.tile.adds(mean_inline3545__tile, 1e-06)"]:
        if contract not in text:
            raise ValueError("冻结mix/norm的算术或循环契约改变：" + contract)
    if not isinstance(fn.body[-1], ast.Return):
        raise ValueError("冻结mix/norm返回结构改变")
    fn.body = [fn.body[0], *resident_math(), fn.body[-1]]
    tasks = [t for t in schedule["tasks"]
             if {name for name in t["kernel_names"] if name} == {"mix_x_rms_norm"}]
    if len(tasks) != 1:
        raise ValueError("HC mix/norm逻辑任务不是唯一的")
    task = tasks[0]
    previous = task["block_num"]
    task.update(block_num=48, member_count=48)
    schedule["call_bindings"][task["task_id"]]["block_num"] = 48
    ast.fix_missing_locations(module)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "core_bodies_hc_mix_resident.py"
    source.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      + ast.unparse(module) + "\n")
    audit = {"tokens": tokens, "members_before": previous, "members_after": 48,
             "tokens_per_member": 2, "physical_mix_tile": [1, 4096],
             "rms_tiles": [16, 256], "rms_part_order": list(range(16)),
             "rms_part_accumulation": "Vector TADD，Scalar仅从UB取单个部分和",
             "bf16_boundary_retained": True, "original_mixed_gm_store_retained": True,
             "mixed_gm_read_removed_bytes": tokens * 4096 * 2,
             "root_and_gm_allocations_unchanged": True,
             "math_requires_device_exact_check": True,
             "numeric_equivalence_validated": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(source)
    return result, audit
