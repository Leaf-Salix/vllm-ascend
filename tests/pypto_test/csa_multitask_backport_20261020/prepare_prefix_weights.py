# SPDX-License-Identifier: Apache-2.0
"""迁移来源head权重先生产、再进入4/20投影分工的次序。"""

import argparse
import ast
import json
import shutil

from prepare import HERE, section
from prepare_query_ready import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="cooperative_query_groups_v1")
    parser.add_argument("--candidate", default="cooperative_prefix_weights_v1")
    args = parser.parse_args()
    if not all(name.isidentifier() for name in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)

    path = target / "decode_csa.py"
    source = path.read_text()
    source = replace_once(
        source,
        "# Dynamic shape variables.\n",
        "from .decode_indexer import MM_ROW_TILE, WEIGHTS_OK, indexer_weights_project, indexer_score_topk_forest\n\n"
        "PREFIX_WEIGHT_WORKERS = 24  # 冻结来源的AIC分工上限\n\n"
        "# Dynamic shape variables.\n",
    )
    source = replace_once(
        source,
        "            # 主压缩投影完成后，4核M256 Hadamard与已有20核QB同时获得派发资格。\n",
        "            # 来源先完成head权重，再进入4核Hadamard/20核QB分工。\n"
        "            # 原分K数学及归约保持；仅按来源24路分工去掉空worker。\n"
        "            weight_workers = pl.min(PREFIX_WEIGHT_WORKERS,"
        " WEIGHTS_OK * ((t_dim + MM_ROW_TILE - 1) // MM_ROW_TILE))\n"
        "            head_weights, head_weights_tid = indexer_weights_project(\n"
        "                x_normed_t, weights_proj, cmp_kv_score_tid, weight_workers,\n"
        "            )\n"
        "            # 多task等待完整权重发布，仍独立调度两个投影组。\n",
    )
    source = replace_once(
        source,
        "                idx_query_i8, idx_query_scale, cmp_kv_score_tid,\n",
        "                idx_query_i8, idx_query_scale, head_weights_tid,\n",
    )
    source = replace_once(
        source,
        "                wq_b, qr_i8_matmul, q_proj_i32, t_dim, cmp_kv_score_tid,\n",
        "                wq_b, qr_i8_matmul, q_proj_i32, t_dim, head_weights_tid,\n",
    )
    old = section(
        source,
        "                _index_scores, index_indices, _leaf_tid = indexer_weights_score(\n",
        "            # sparse_attn_csa folds",
    )
    source = source.replace(
        old,
        "                # head_weights沿原接口传给forest，不重复投影或改变系数量化。\n"
        "                _index_scores, index_indices, _leaf_tid = indexer_score_topk_forest(\n"
        "                    idx_query_i8, idx_query_scale, head_weights,\n"
        "                    idx_native_kv_cache, idx_block_table, position_ids, kv_seq_lens,\n"
        "                    idx_topk_scores, idx_topk, idx_quant_tid, head_weights_tid,\n"
        "                    idx_cache_write_tid, max_seq_len, early_scale_layout, query_ready_tid,\n"
        "                )\n\n",
        1,
    )
    ast.parse(source)
    path.write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            {
                "candidate": args.candidate,
                "base": args.base,
                "donor": "13259f92 projection_overlap_stages.py：主压缩→head权重→4/20分支",
                "changes": [
                    "head权重投影从Score封装中提前到Hadamard/QB之前，复用原数学helper",
                    "按来源最多24个AIC划分非空权重unit；分K次序与FP32归约保持",
                    "两分支依赖head_weights_tid，Score直接消费已经生成的权重",
                ],
                "preserved": "QB/Hadamard分工、Q就绪依赖、sync_start、原early resolve、所有数据/状态合同",
                "adaptation": "多task等整组head权重与归约；来源逐核发布，粒度不同须看完整CSA实测",
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
