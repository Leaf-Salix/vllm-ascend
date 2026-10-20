# SPDX-License-Identifier: Apache-2.0
"""将已验证的逐token输出数学体按Native A3行数均衡策略映射到大T的48个AIV。"""

import argparse
import ast
import json
import shutil

from prepare import HERE
from prepare_query_ready import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="hc_post_workers_v1")
    parser.add_argument("--candidate", default="hc_post_balanced_v1")
    args = parser.parse_args()
    if not all(value.isidentifier() for value in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    source = (HERE / "sources" / args.base / "decode_o_proj.py").read_text()
    source = replace_once(
        source,
        "    act_t_blks = (t_dim + act_row_step - 1) // act_row_step\n",
        "    act_t_blks = (t_dim + act_row_step - 1) // act_row_step\n"
        "    act_base_rows = act_row_step\n"
        "    act_extra_rows = 0\n"
        "    # 大T参考A3 Native将余数分给前面的owner；最多48个非空同步任务。\n"
        "    if POST_SYNC and ROW_TILE == PROJ_B_MM_T_TILE:\n"
        "        act_t_blks = pl.min(t_dim, PROJ_B_ACT_WORKERS)\n"
        "        act_base_rows = t_dim // act_t_blks\n"
        "        act_extra_rows = t_dim - act_base_rows * act_t_blks\n",
    )
    source = replace_once(
        source,
        "        t0 = tblk * act_row_step\n",
        "        t0 = tblk * act_row_step\n"
        "        t_end = pl.min(t0 + act_row_step, t_dim)\n"
        "        if POST_SYNC and ROW_TILE == PROJ_B_MM_T_TILE:\n"
        "            t0 = tblk * act_base_rows + pl.min(tblk, act_extra_rows)\n"
        "            t_end = t0 + act_base_rows + pl.min(1, pl.max(act_extra_rows - tblk, 0))\n",
    )
    source = replace_once(
        source,
        "        for b_tb in pl.range(t0, pl.min(t0 + act_row_step, t_dim), PROJ_B_ACT_T_TILE):\n",
        "        for b_tb in pl.range(t0, t_end, PROJ_B_ACT_T_TILE):\n",
    )
    source = replace_once(
        source,
        "            PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE, False,\n",
        "            PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE, True,\n",
    )
    ast.parse(source)
    shutil.copytree(HERE / "sources" / args.base, target)
    (target / "decode_o_proj.py").write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                changed_files=["decode_o_proj.py"],
                donor="13259f92/output_projection_b_stages.py:o_dequant_hc_post_stage",
                native_reference="cann-recipes-infer@6d060311/hc_post/op_host/hc_post_tiling.cpp",
                policy="仅ROW_TILE=128对应的大T路径：min(T,48)非空owner、均衡连续行及sync_start",
                examples={"T144": "48x3", "T168": "24x4+24x3", "T192": "48x4"},
                preserved="小T/T96策略、N4096单行数学、stage2反量化、八组左折叠、BF16边界、任务依赖",
                historical_negative="旧8K/B24扩核回退，不当作当前128K/B24已经失败；保留同轮复核",
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
