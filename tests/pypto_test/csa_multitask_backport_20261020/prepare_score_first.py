# SPDX-License-Identifier: Apache-2.0
"""协同前段的多task调度适配：Score优先，Q/KV尾段通过直接TaskId后置。"""

import argparse
import ast
import json
import shutil

from prepare import HERE, section
from prepare_fused_topk import add_parameter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="cooperative_prefix_v1")
    parser.add_argument("--candidate", default="cooperative_score_first_v1")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    path = target / "qkv_proj_rope.py"
    source = path.read_text()
    dequant = section(
        source,
        "@pl.jit.inline(auto_scope=False)\ndef q_proj_q_dequant(",
        "@pl.jit.inline(auto_scope=False)\ndef q_proj_q(",
    )
    dequant = dequant.replace("def q_proj_q_dequant(", "def cooperative_q_dequant_after_score(", 1)
    dequant = add_parameter(dequant, "cooperative_q_dequant_after_score", "    score_dep: pl.Scalar[pl.TASK_ID],\n")
    dequant = dequant.replace(
        '        name_hint="qproj_dequant_rms_nope_rope",\n',
        '        name_hint="qproj_dequant_rms_nope_rope",\n        deps=[tail_dependencies[0]],\n',
        1,
    )
    # TaskId仅出现在deps属性时，当前inline pass不会保留参数值的SSA替换。
    # 用实际值数组承载同一个依赖，不创建额外task或全局同步。
    dequant = dequant.replace(
        "    t_dim = pl.tensor.dim(q, 0)\n",
        "    tail_dependencies = pl.array.create(1, pl.TASK_ID)\n"
        "    tail_dependencies[0] = score_dep\n"
        "    t_dim = pl.tensor.dim(q, 0)\n",
        1,
    )
    kv = section(
        source,
        "@pl.jit.inline(auto_scope=False)\ndef _kv_project(",
        "@pl.jit.inline(auto_scope=False)\ndef kv_proj_rope(",
    )
    kv_with_value_dependency = kv.replace(
        "    t_dim = pl.tensor.dim(x, 0)\n",
        "    kv_dependencies = pl.array.create(1, pl.TASK_ID)\n"
        "    kv_dependencies[0] = late_dep\n"
        "    t_dim = pl.tensor.dim(x, 0)\n",
        1,
    ).replace("deps=[late_dep]", "deps=[kv_dependencies[0]]", 1)
    source = source.replace(kv, kv_with_value_dependency, 1)
    source += "\n\n" + dequant
    ast.parse(source)
    path.write_text(source)

    path = target / "decode_csa.py"
    source = path.read_text()
    source = source.replace(
        "    rope_prepare, q_proj_qr_policy, cooperative_q_proj_q, kv_proj_rope,\n",
        "    rope_prepare, q_proj_qr_policy, cooperative_q_dequant_after_score, kv_proj_rope,\n",
        1,
    )
    source = source.replace(
        "from .qkv_proj_rope import (\n",
        "from .q_projection import QPROJ_PIPE_M_TILE, q_proj_q_matmul\nfrom .qkv_proj_rope import (\n",
        1,
    )
    source = source.replace(
        "            # 原KV只等QB投影，不等其Vector反量化；从Score前段移到这条独立尾链。\n",
        "            # 原KV与Q反量化在下方通过直接Score TaskId后置。\n",
        1,
    )
    before = section(
        source,
        "            _q_result, qb_tid = cooperative_q_proj_q(",
        "            kv_proj_rope(",
    )
    source = source.replace(
        before,
        """            # 沿已有T<=dense tile合同提升同一QB缓冲；只拆开Cube和Vector派发。
            qproj_rows = ((t_dim + QPROJ_PIPE_M_TILE - 1) // QPROJ_PIPE_M_TILE) * QPROJ_PIPE_M_TILE
            q_proj_i32 = pl.create_tensor([qproj_rows, H * HEAD_DIM], dtype=pl.INT32)
            q_proj_i32, qb_tid = q_proj_q_matmul(
                wq_b, qr_i8_matmul, q_proj_i32, t_dim, cmp_kv_score_tid,
            )
""",
        1,
    )
    tail = section(
        source,
        "            kv_proj_rope(",
        "            # Bound indexer scratch to its own runtime scope.",
    )
    source = source.replace(tail, "", 1)
    source = source.replace(
        "            # Bound indexer scratch to its own runtime scope.\n",
        "            # 显式值数组保留跨scope的真实Score TaskId，不增加dummy任务。\n"
        "            score_ready = pl.array.create(1, pl.TASK_ID)\n"
        "            score_ready[0] = qb_tid\n"
        "            # Bound indexer scratch to its own runtime scope.\n",
        1,
    )
    marker = "\n            # sparse_attn_csa folds"
    if source.count(marker) != 1:
        raise ValueError("Attention入口标记不唯一")
    # 此处是显式调度顺序边，Score的结果不作为Q/KV数学输入。
    # Q反量化原本读取q_proj_i32的真实依赖仍由编译器保留。
    tail = tail.replace(
        "q_rope_sin_signed, q_rope_swap_idx, kv, qb_tid,",
        "q_rope_sin_signed, q_rope_swap_idx, kv, score_ready[0],",
    )
    deferred = (
        """                score_ready[0] = _leaf_tid

            # 优先Score，避免Q/KV尾段抢占其启动窗口；保留allow_early_resolve。
            cooperative_q_dequant_after_score(
                wq_b_scale, freqs_cos, q_rope_sin_signed, q_rope_swap_idx,
                q, qr_scale_pad_store, q_proj_i32, 0, t_dim, score_ready[0],
            )
"""
        + tail
    )
    source = source.replace(marker, "\n" + deferred + marker, 1)
    ast.parse(source)
    path.write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="e91e9f17协同前段的多task调度适配；基于v1五窗先后顺序翻转的证据",
                changes="QB Cube维持原位置；Q反量化和原KV投影通过直接Score TaskId后置",
                preserved="QB/反量化/KV数学、真实输入依赖、尾块、量化和cache/state规则",
                limitation="显式顺序边可能限制部分重叠，需配对mean/max判断；不能用v1最快窗认领收益",
                lifetime="沿原单dense-tile入口；QB缓冲提升至外层同一scope供后置反量化使用",
                status="source_prepared_no_device_evidence",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
