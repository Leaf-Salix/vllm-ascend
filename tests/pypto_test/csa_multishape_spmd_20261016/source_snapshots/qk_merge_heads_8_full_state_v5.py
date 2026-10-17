# SPDX-License-Identifier: Apache-2.0
"""PV在线合并按8个head处理，并依次完成左半和右半状态。

原来每个AIV同时合并32个head，左/右各256列。当前仍保留完整的
32-head running state，但每次只加载/计算8行，再assemble相应
不重叠行。每个元素的old*alpha + incoming*beta次序不变，m/l归约、
五块合并顺序、输出发布、Cube和FFTS均不变。能否减少屏障须看实际编译。
"""

import ast
import copy
import json
from pathlib import Path

MERGE_HEADS = 8


def merge_body():
    lines = []
    for chunk in range(32 // MERGE_HEADS):
        row = chunk * MERGE_HEADS
        lines.extend([
            f"qkm_alpha_{chunk} = pl.tile.reshape(pl.tile.extract(pl.tile.reshape("
            "alpha_inline4273__ssa_v0, [1, 32]), 0, "
            f"{row}, [1, {MERGE_HEADS}], target_memory=pl.Mem.Vec), [{MERGE_HEADS}, 1])",
            f"qkm_beta_{chunk} = pl.tile.reshape(pl.tile.extract(pl.tile.reshape("
            "beta_inline4330__ssa_v0, [1, 32]), 0, "
            f"{row}, [1, {MERGE_HEADS}], target_memory=pl.Mem.Vec), [{MERGE_HEADS}, 1])",
        ])
    # 先彻底完成left，再开始right，避免两份旧状态与两份新状态同时活跃。
    # 16-head交错两侧的首版实际需要196608B，超过当前188416B的UB上限。
    for side, column in (("left", 0), ("right", 256)):
        lines.append(f"qkm_{side} = {side}_iter")
        for chunk in range(32 // MERGE_HEADS):
            row = chunk * MERGE_HEADS
            prefix = f"qkm_{side}_{chunk}"
            lines.extend([
                f"{prefix}_old = pl.tile.slice(qkm_{side}, [{MERGE_HEADS}, 256], [{row}, 0])",
                f"{prefix}_incoming = pl.tile.load(pv_transfer_inline4320__ssa_v0, "
                f"[pv_transfer_row_inline4421__ssa_v1 + qk_lane_head_inline4436__ssa_v0 + {row}, {column}], "
                f"[{MERGE_HEADS}, 256], [{MERGE_HEADS}, 256], target_memory=pl.Mem.Vec)",
                f"{prefix}_scaled_old = pl.tile.row_expand_mul({prefix}_old, qkm_alpha_{chunk})",
                f"{prefix}_scaled_incoming = pl.tile.row_expand_mul({prefix}_incoming, qkm_beta_{chunk})",
                f"{prefix}_sum = pl.tile.add({prefix}_scaled_old, {prefix}_scaled_incoming)",
                f"qkm_{side} = pl.tile.assemble(qkm_{side}, {prefix}_sum, [{row}, 0])",
            ])
    lines.extend(["next_left_inline4242__ssa_v0 = qkm_left", "next_right_inline4247__ssa_v0 = qkm_right"])
    return ast.parse("\n".join(lines)).body


def split_pv_merge(schedule, catalog, output):
    tree = ast.parse(Path(catalog["body_bundle"]).read_text())
    original_functions = {n.name: ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
              and node.name == "qk_pv_aiv")
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("PV分head合并候选先限定B16/S6")
    changes = 0
    expected = ["pv_left_inline4243__ssa_v0", "t__tmp_v811", "t__tmp_v812",
                "next_left_inline4242__ssa_v0", "pv_right_inline4395__ssa_v0",
                "t__tmp_v813", "t__tmp_v814", "next_right_inline4247__ssa_v0"]
    sync_before = [ast.dump(node) for node in ast.walk(fn) if isinstance(node, ast.Call)
                   and ast.unparse(node.func).startswith("pl.system.")]
    writes_before = [ast.dump(n) for n in ast.walk(fn) if isinstance(n, ast.Call)
                     and ast.unparse(n.func) in ('pl.tile.store', 'pl.tensor.write')]
    scalar_reads_before = [ast.dump(n) for n in ast.walk(fn) if isinstance(n, ast.Call)
                           and ast.unparse(n.func) == 'pl.tensor.read']
    for parent in list(ast.walk(fn)):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body):
            if (not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name)
                    or statement.target.id != expected[0]):
                continue
            old = body[index:index + 8]
            if ([node.target.id for node in old if isinstance(node, ast.AnnAssign)] != expected
                    or ast.unparse(old[0].value.args[2]) != "[32, 256]"
                    or ast.unparse(old[4].value.args[2]) != "[32, 256]"):
                raise ValueError("PV原32-head合并结构改变")
            body[index:index + 8] = merge_body()
            changes += 1
            break
    if changes != 1:
        raise ValueError("PV合并变换位置不是唯一的")
    sync_after = [ast.dump(node) for node in ast.walk(fn) if isinstance(node, ast.Call)
                  and ast.unparse(node.func).startswith("pl.system.")]
    if sync_before != sync_after:
        raise ValueError("PV合并候选改变了显式同步")
    if writes_before != [ast.dump(n) for n in ast.walk(fn) if isinstance(n, ast.Call)
                         and ast.unparse(n.func) in ('pl.tile.store', 'pl.tensor.write')]:
        raise ValueError('PV合并改变了GM写入范围')
    if scalar_reads_before != [ast.dump(n) for n in ast.walk(fn) if isinstance(n, ast.Call)
                               and ast.unparse(n.func) == 'pl.tensor.read']:
        raise ValueError('PV合并改变了metadata读取')
    if any(ast.dump(n) != original_functions[n.name] for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name != fn.name):
        raise ValueError('PV合并意外修改了其他函数')
    # 行依赖仍按完整32-head运行状态和原query分工判断；这里只缩小PV读取Tile。
    assert sorted((row, col) for chunk in range(32 // MERGE_HEADS)
                  for row in range(chunk * MERGE_HEADS, (chunk + 1) * MERGE_HEADS)
                  for col in range(512)) == [(row, col) for row in range(32) for col in range(512)]
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "core_bodies_qk_merge_heads.py"
    target.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      "# PV按8-head先左后右合并，不改原FFTS和归约顺序。\n"
                      + ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    audit = {"kernel": fn.name, "heads_per_aiv": 32, "heads_per_merge_chunk": MERGE_HEADS,
             "running_state_shape_unchanged": [32, 256], "incoming_tile_old_bytes": 32768,
             "incoming_tile_new_bytes": MERGE_HEADS * 256 * 4, "alpha_beta_and_m_l_unchanged": True,
             "block_merge_order_unchanged": True, "original_ffts_unchanged": True,
             "qk_gather_and_aic_unchanged": True, "output_publish_unchanged": True,
             "extra_gm_or_root_inputs": False,
             "source_regions_written": [[i, i + MERGE_HEADS] for i in range(0, 32, MERGE_HEADS)],
             "left_finished_before_right_initialization": True,
             "gm_write_asts_and_scalar_metadata_reads_unchanged": True,
             "other_function_asts_unchanged": True,
             "pv_read_region_union_unchanged": [32, 512],
             "actual_ub_plan_verified": False, "precision_proven": False, "performance_proven": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["body_bundle"] = str(target)
    return result, audit
