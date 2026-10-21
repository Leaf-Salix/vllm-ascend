# SPDX-License-Identifier: Apache-2.0
"""按冻结单SPMD的Q→Score顺序，适配协同前段的直接任务依赖。"""

import argparse
import ast
import json
import shutil

from prepare import HERE


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"需要唯一的来源片段：{before[:100]}")
    return source.replace(before, after, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="cooperative_q_only_sync_v1")
    parser.add_argument("--candidate", default="cooperative_query_ready_v1")
    args = parser.parse_args()
    if not all(name.isidentifier() for name in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)

    path = target / "qkv_proj_rope.py"
    source = path.read_text()
    marker = "@pl.jit.inline(auto_scope=False)\ndef cooperative_q_dequant_after_score("
    begin = source.index(marker)
    helper = source[begin:]
    helper = replace_once(helper, "cooperative_q_dequant_after_score(", "cooperative_q_dequant_ready(")
    helper = replace_once(
        helper,
        "    score_dep: pl.Scalar[pl.TASK_ID],\n",
        "    qb_dep: pl.Scalar[pl.TASK_ID],\n    layout_dep: pl.Scalar[pl.TASK_ID],\n",
    )
    helper = replace_once(
        helper,
        "    tail_dependencies = pl.array.create(1, pl.TASK_ID)\n    tail_dependencies[0] = score_dep\n",
        "    # 真实QB输入和原scale布局任务都完成后发布Q就绪。\n"
        "    tail_dependencies = pl.array.create(2, pl.TASK_ID)\n"
        "    tail_dependencies[0] = qb_dep\n"
        "    tail_dependencies[1] = layout_dep\n",
    )
    helper = replace_once(helper, "    for dq_worker in pl.spmd(\n", "    with pl.spmd(\n")
    helper = replace_once(
        helper,
        "        deps=[tail_dependencies[0]],\n        allow_early_resolve=True,\n    ):\n",
        "        deps=[tail_dependencies[0], tail_dependencies[1]],\n"
        "        allow_early_resolve=True,\n"
        "    ) as query_ready_tid:\n"
        "        dq_worker = pl.tile.get_block_idx()\n",
    )
    helper = replace_once(helper, "    return q\n", "    return q, query_ready_tid\n")
    source = source[:begin] + helper
    ast.parse(source)
    path.write_text(source)

    path = target / "decode_csa.py"
    source = path.read_text()
    source = source.replace("cooperative_q_dequant_after_score", "cooperative_q_dequant_ready")
    source = replace_once(
        source,
        "            # 原KV在QB后具备派发资格；只有Q反量化通过Score TaskId后置。\n",
        "            # 原KV与Q反量化沿QB依赖；Q就绪后Score才能进入核内。\n",
    )
    source = replace_once(
        source,
        "            # 显式值数组保留跨scope的真实Score TaskId，不增加dummy任务。\n"
        "            score_ready = pl.array.create(1, pl.TASK_ID)\n"
        "            score_ready[0] = qb_tid\n",
        "",
    )
    source = replace_once(source, "                score_ready[0] = _leaf_tid\n", "")
    begin = source.index("            # 仅Q反量化依赖Score；KV沿QB依赖，保留allow_early_resolve。\n")
    end = source.index("            # sparse_attn_csa folds", begin)
    tail = source[begin:end]
    tail = replace_once(
        tail,
        "            # 仅Q反量化依赖Score；KV沿QB依赖，保留allow_early_resolve。\n"
        "            cooperative_q_dequant_ready(\n",
        "            # 13259f92的Vector顺序：Q反量化完成后进入Score。\n"
        "            # KV投影也提前提交；数学、写回保护及early resolve均保持。\n"
        "            _q_ready, query_ready_tid = cooperative_q_dequant_ready(\n",
    )
    tail = replace_once(tail, "0, t_dim, score_ready[0],", "0, t_dim, qb_tid, early_scale_deps[0],")
    source = source[:begin] + source[end:]
    source = replace_once(
        source,
        "            # Bound indexer scratch to its own runtime scope.\n",
        tail
        + "            # Q就绪是原layout完成的后继，作为Score的直接依赖保留二者。\n"
        + "            # 不新增dummy、GM信号或运行时等待实现。\n"
        + "            # Bound indexer scratch to its own runtime scope.\n",
    )
    source = replace_once(
        source,
        "                    max_seq_len, early_scale_layout, early_scale_deps[0],\n",
        "                    max_seq_len, early_scale_layout, query_ready_tid,\n",
    )
    ast.parse(source)
    path.write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            {
                "candidate": args.candidate,
                "base": args.base,
                "donor": "13259f92 vector_schedule.py：q_dequant_rope_stage先于indexer_score_vector_lane_stage",
                "changes": [
                    "去掉Q反量化对Score的后置顺序边，改为依赖QB和原scale-layout任务",
                    "Q/KV尾链提前到Score提交前；既有Score显式layout依赖改为其后继Q就绪任务",
                    "保留4核Hadamard/20核QB、非空Plan及QK sync_start，不改算术",
                ],
                "dependency_proof": "layout→Q→Score；原layout、QB及自动Tensor数据依赖均保留，没有新增dummy",
                "limitations": "多task等待整组Q，单SPMD为逐AIV顺序；不能预先宣称等价的调度收益",
                "preserved": "所有incore数学、尾行、cache/state保护、现有allow_early_resolve和私有工具链",
                "status": "source_prepared_no_device_evidence",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
