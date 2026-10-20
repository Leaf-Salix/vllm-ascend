# SPDX-License-Identifier: Apache-2.0
"""按单SPMD和A3 HcPost的逐token owner，补齐小T的有效AIV与同步启动。"""

import argparse
import ast
import json
import shutil

from prepare import HERE
from prepare_query_ready import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="topk_prefix_v1")
    parser.add_argument("--candidate", default="hc_post_workers_v1")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    source = (HERE / "sources" / args.base / "decode_o_proj.py").read_text()
    original = source
    head = source.index("@pl.jit.inline\ndef decode_o_proj_tp1(")
    prefix, policy = source[:head], source[head:]
    policy = replace_once(
        policy,
        "            PROJ_B_SMALL_T_TILE, PROJ_A_MM_N_TILE, False,\n",
        "            PROJ_B_SMALL_T_TILE, PROJ_A_MM_N_TILE, True,\n",
    )
    policy = replace_once(
        policy,
        "    elif t_dim == PROJ_B_MEDIUM_T_TILE:\n",
        "    elif t_dim <= PROJ_B_ACT_WORKERS:\n"
        "        # 一个有效token一个AIV；最多48个任务，满足sync_start同时准入。\n"
        "        x_out = _decode_o_proj_tp1_tiled(\n"
        "            o_packed, wo_a, wo_b, wo_b_scale, residual, post, comb, x_out, heads_dep,\n"
        "            PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE, True,\n"
        "        )\n"
        "    elif t_dim == PROJ_B_MEDIUM_T_TILE:\n",
    )
    source = prefix + policy
    tree = ast.parse(source)
    before_tree = ast.parse(original)
    unchanged_before = [
        n for n in before_tree.body if not isinstance(n, ast.FunctionDef) or n.name != "decode_o_proj_tp1"
    ]
    unchanged_after = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name != "decode_o_proj_tp1"]
    if ast.dump(ast.Module(body=unchanged_before, type_ignores=[])) != ast.dump(
        ast.Module(body=unchanged_after, type_ignores=[])
    ):
        raise ValueError("修改超出外层资源策略，核内数学或大档实现被改变")
    shutil.copytree(HERE / "sources" / args.base, target)
    (target / "decode_o_proj.py").write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            {
                "candidate": args.candidate,
                "base": args.base,
                "changed_files": ["decode_o_proj.py"],
                "donor": "13259f92/output_projection_b_stages.py:o_dequant_hc_post_stage",
                "native_reference": "cann/cann-recipes-infer@6d060311ae75f6105fe4216e20566875da5a8838",
                "native_paths": [
                    "ops/ascendc/src/hc_post/op_host/hc_post_tiling.cpp",
                    "ops/ascendc/src/hc_post/op_kernel/hc_post_d_split.h",
                ],
                "changes": "仅T<=48按一token一AIV并sync_start，保留原T96策略和T144回退限制",
                "preserved": "N4096、1行计算体、group stage2、八组加法、BF16边界、全部依赖与大T策略",
                "ownership": "T24为24核各1行，T48为48核各1行；不把D切块再乘入同步启动块数",
                "status": "private_candidate_no_device_evidence",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
