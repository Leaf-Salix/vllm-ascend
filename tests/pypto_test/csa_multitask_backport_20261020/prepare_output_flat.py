# SPDX-License-Identifier: Apache-2.0
"""沿已验证单SPMD的跨组工作编号，减少输出投影的TMR分派而不引入软同步。"""

import argparse
import ast
import json
import re
import shutil

from prepare import HERE
from prepare_query_ready import replace_once


def edit_function(source, name, edit):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    before = "".join(lines[node.lineno - 1 : node.end_lineno])
    after = edit(before)
    if before == after:
        raise ValueError(f"{name}未产生预期修改")
    ast.parse(after)
    lines[node.lineno - 1 : node.end_lineno] = [after]
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="formal_4d5b2ed9")
    parser.add_argument("--candidate", default="output_flat_g8_v2")
    parser.add_argument("--nested-column-remainder", action="store_true", help="复现v1中NZ列偏移不能证明非负的CPU错误")
    parser.add_argument("--groups-per-task", type=int, choices=(1, 2, 4, 8), default=8)
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / name for name in (args.base, args.candidate))
    source = (base / "decode_o_proj.py").read_text()
    source = replace_once(
        source,
        'T_DYN = pl.dynamic("T_DYN")',
        f'OUTPUT_GROUP_BATCH = {args.groups_per_task}\n\nT_DYN = pl.dynamic("T_DYN")',
    )

    def oa(text):
        text = replace_once(
            text,
            "        proj_a_rows * (O_LORA // A_COL_TILE),",
            "        OUTPUT_GROUP_BATCH * proj_a_rows * (O_LORA // A_COL_TILE),",
        )
        text = replace_once(
            text,
            "        pa_unit = pl.tile.get_block_idx()\n",
            "        pa_global = pl.tile.get_block_idx()\n"
            "        pa_group = pl.max(g + pa_global // (proj_a_rows * (O_LORA // A_COL_TILE)), 0)\n"
            "        pa_unit = pa_global % (proj_a_rows * (O_LORA // A_COL_TILE))\n",
        )
        text = text.replace("pa_src0 = row_base_o + pa_r0", "pa_src0 = pa_group * T_PAD + pa_r0")
        if not args.nested_column_remainder:
            # 来源同样直接取全局编号对常量列块数的模；组步长是列块数的倍数。
            text = replace_once(
                text, "nf = pa_unit % (O_LORA // A_COL_TILE)", "nf = pa_global % (O_LORA // A_COL_TILE)"
            )
        text = text.replace("wo_a[g : g + 1,", "wo_a[pa_group : pa_group + 1,")
        text = text.replace("pl.load(wo_a, [g,", "pl.load(wo_a, [pa_group,")
        return text.replace("out_col_g + n0", "pa_group * O_LORA + n0")

    for name in ("_proj_a_mm_nz", "_proj_a_mm_nd"):
        source = edit_function(source, name, oa)

    def ob_nd(text):
        text = replace_once(
            text,
            "        proj_b_t_rows * (D // PROJ_B_D_TILE), name_hint=",
            "        OUTPUT_GROUP_BATCH * proj_b_t_rows * (D // PROJ_B_D_TILE), name_hint=",
        )
        text = replace_once(
            text,
            "        pb_unit = pl.tile.get_block_idx()\n",
            "        pb_global = pl.tile.get_block_idx()\n"
            "        pb_group = pl.max(g + pb_global // (proj_b_t_rows * (D // PROJ_B_D_TILE)), 0)\n"
            "        pb_unit = pb_global % (proj_b_t_rows * (D // PROJ_B_D_TILE))\n",
        )
        return text.replace("k0 = col_g +", "k0 = pb_group * O_LORA +").replace("g * D + n0", "pb_group * D + n0")

    source = edit_function(source, "_proj_b_mm_nd", ob_nd)

    def ob_nz(text):
        text = replace_once(
            text,
            "    d0 = pl.tile.get_block_idx() * PROJ_B_D_TILE\n",
            "    pb_global = pl.tile.get_block_idx()\n"
            "    pb_group = pl.max(g + pb_global // (D // PROJ_B_D_TILE), 0)\n"
            "    d0 = (pb_global % (D // PROJ_B_D_TILE)) * PROJ_B_D_TILE\n",
        )
        text = text.replace("[t0, col_g]", "[t0, pb_group * O_LORA]")
        text = text.replace("pl.max(g, 0) * O_LORA", "pb_group * O_LORA")
        return text.replace("g * D + n0", "pb_group * D + n0")

    source = edit_function(source, "_proj_b_mm_nz_kernel", ob_nz)
    source = edit_function(
        source,
        "_proj_b_mm_nz",
        lambda text: replace_once(
            text,
            "        D // PROJ_B_D_TILE, name_hint=",
            "        OUTPUT_GROUP_BATCH * (D // PROJ_B_D_TILE), name_hint=",
        ),
    )

    def output(text):
        text = replace_once(
            text, "pl.array.create(O_GROUPS, pl.TASK_ID)", "pl.array.create(O_GROUPS // OUTPUT_GROUP_BATCH, pl.TASK_ID)"
        )
        text = replace_once(
            text,
            "        for g in pl.parallel(O_GROUPS):\n",
            "        for group_task in pl.parallel(O_GROUPS // OUTPUT_GROUP_BATCH):\n"
            "            g = group_task * OUTPUT_GROUP_BATCH\n",
        )
        text = replace_once(
            text,
            "                (t_dim + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE,\n",
            "                OUTPUT_GROUP_BATCH * ((t_dim + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE),\n",
        )
        begin = text.index("                quant_start =")
        end = text.index("\n            partials, pb_tid =", begin)
        quant = text[begin:end]
        quant = re.sub(r"\bg\b", "quant_group", quant)
        quant = re.sub(r"\bcol_g\b", "quant_col", quant)
        quant = replace_once(
            quant,
            "                quant_start = pl.tile.get_block_idx() * QUANT_TASK_T_TILE\n",
            "                quant_global = pl.tile.get_block_idx()\n"
            "                quant_per_group = (t_dim + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE\n"
            "                quant_group = pl.max(g + quant_global // quant_per_group, 0)\n"
            "                quant_col = quant_group * O_LORA\n"
            "                quant_start = (quant_global % quant_per_group) * QUANT_TASK_T_TILE\n",
        )
        text = text[:begin] + quant + text[end:]
        text = replace_once(text, "proj_b_tids[g] = pb_tid", "proj_b_tids[group_task] = pb_tid")
        return replace_once(
            text,
            "deps=[proj_b_tids[i] for i in range(O_GROUPS)]",
            "deps=[proj_b_tids[i] for i in range(O_GROUPS // OUTPUT_GROUP_BATCH)]",
        )

    source = edit_function(source, "_decode_o_proj_tp1_tiled", output)
    ast.parse(source)
    shutil.copytree(base, target)
    (target / "decode_o_proj.py").write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                groups_per_task=args.groups_per_task,
                changed_files=["decode_o_proj.py"],
                donor=(
                    "13259f92 output_projection_a/b_stages：将(group,row,column)统一编号；"
                    "数学及有效行已在正式多task迁入"
                ),
                adaptation="把连续若干组作为一组TMR任务，保持每个原块独立分派；不用局部MIX或新软同步",
                preserved="OA K256顺序、分组amax/INT8舍入、OB行块/有效尾、cache属性、HCpost数学及sync_start不变",
                task_change=f"输出8组×3任务改为{8 // args.groups_per_task}组×3任务，再接原HCpost",
                nested_column_remainder=args.nested_column_remainder,
                tradeoff="合并减少分派，但同一组批次内OA→量化→OB使用整批依赖，可能损失原逐组重叠",
                status="private_candidate_no_device_evidence",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
