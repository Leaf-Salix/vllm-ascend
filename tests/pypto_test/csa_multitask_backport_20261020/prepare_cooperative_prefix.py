# SPDX-License-Identifier: Apache-2.0
"""把冻结单SPMD协同前段的生产次序和4/20分工一起适配到多task。"""

import argparse
import ast
import json
import shutil

from prepare import HERE, section
from prepare_query_hadamard import migrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="qa_m16_v3")
    parser.add_argument("--candidate", default="cooperative_prefix_v1")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)

    # 数学直接复用冻结来源的M256 Hadamard；此次还联动投影时序，区别于孤立改4核。
    path = target / "decode_indexer.py"
    path.write_text(migrate(path.read_text(), (HERE / "sources/spmd/projection_overlap_stages.py").read_text()))
    path = target / "qkv_proj_rope.py"
    source = path.read_text()
    original = section(
        source,
        "@pl.jit.inline(auto_scope=False)\ndef q_proj_q(",
        "@pl.jit.inline(auto_scope=False)\ndef q_proj_rope_policy(",
    )
    extended = original.replace("def q_proj_q(", "def cooperative_q_proj_q(", 1)
    extended = extended.replace(
        "    t_dim = pl.tensor.dim(x, 0)\n",
        "    # Native入口保证T不超过当前dense tile，沿原QR的TaskId交接合同。\n"
        "    ready = pl.array.create(1, pl.TASK_ID)\n"
        "    ready[0] = qproj_dep\n"
        "    t_dim = pl.tensor.dim(x, 0)\n",
        1,
    )
    extended = extended.replace(
        "            q_proj_q_dequant(\n",
        "            ready[0] = _qproj_tid\n            q_proj_q_dequant(\n",
        1,
    )
    extended = extended.replace("    return q\n", "    return q, ready[0]\n", 1)
    source += "\n\n" + extended
    ast.parse(source)
    path.write_text(source)

    path = target / "decode_csa.py"
    source = path.read_text()
    source = source.replace(
        "from .decode_indexer import indexer, LONG_S6_MIN_QUERY_ROWS, TOPK_CANDIDATES_PER_LEAF\n",
        "from .decode_indexer import (\n"
        "    LONG_S6_MIN_QUERY_ROWS, TOPK_CANDIDATES_PER_LEAF, TP1_WEIGHTS_WORKERS,\n"
        "    indexer_qr_rope, indexer_qr_hadamard_mm, indexer_weights_score,\n"
        ")\n",
        1,
    )
    source = source.replace(
        "from .qkv_proj_rope import QR_M_TILE, qkv_proj_rope_policy\n",
        "from .qkv_proj_rope import (\n"
        "    QR_M_TILE, QPROJ_T_PAD, PREFILL_DENSE_TILE,\n"
        "    rope_prepare, q_proj_qr_policy, cooperative_q_proj_q, kv_proj_rope,\n"
        ")\n\n"
        "# 本候选与原q_proj_qr_policy使用同一个单dense-tile入口合同。\n"
        "if DECODE_BATCH * DECODE_SEQ > PREFILL_DENSE_TILE:\n"
        '    raise ValueError("协同前段需要完整QB投影的单tile TaskId合同")\n',
        1,
    )
    old = section(source, "            q, qa_tid = qkv_proj_rope_policy(", "            ori_block_num =")
    prefix = """            # QA/QR先独立发出；主QB推迟至Indexer query和主压缩投影完成。
            q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
            q_rope_swap_idx = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.INT32)
            rope_prepare(freqs_sin, q_rope_sin_signed, q_rope_swap_idx)
            qr_i8_matmul = pl.create_tensor([QPROJ_T_PAD, Q_LORA], dtype=pl.INT8)
            qr_scale_pad_store = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32, manual_dep=True)
            qa_tid = q_proj_qr_policy(
                x_normed_t, wq_a, gamma_cq, qr, qr_scale,
                qr_i8_matmul, qr_scale_pad_store, qa_m16_parallel,
            )
            idx_query_rope = pl.create_tensor([T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
            idx_query_i8 = pl.create_tensor([T_PAD * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
            idx_query_scale = pl.create_tensor([T_PAD * IDX_N_HEADS, 1], dtype=pl.FP32)
            idx_query_project_tid = indexer_qr_rope(
                x_normed_t, qr, qr_scale, idx_wq_b, idx_wq_b_scale,
                freqs_cos, idx_sin_signed, idx_query_rope,
            )

"""
    source = source.replace(old, prefix, 1)
    writeback = section(
        source,
        "            ori_block_num =",
        "            # Keep Q_A ahead of the heavier Attention Compressor projection.",
    )
    source = source.replace(writeback, "", 1)
    source = source.replace(
        "            # Keep Q_A ahead of the heavier Attention Compressor projection.\n"
        "            # The lighter Indexer projection overlaps Q_A; its Hadamard gate stays below.\n"
        "            compressor_dep = pl.system.task_dummy(deps=[late_dep, qa_tid])",
        "            # 来源COOPERATIVE_PREFIX：Indexer query先出，随后主压缩，再启动4/20分支。\n"
        "            compressor_dep = pl.system.task_dummy(deps=[late_dep, qa_tid, idx_query_project_tid])",
        1,
    )
    insertion = """            # 主压缩投影完成后，4核M256 Hadamard与已有20核QB同时获得派发资格。
            # 原KV只等QB投影，不等其Vector反量化；从Score前段移到这条独立尾链。
            _idx_had_tid, idx_quant_tid = indexer_qr_hadamard_mm(
                x_normed_t, idx_query_rope, hadamard_idx,
                idx_query_i8, idx_query_scale, cmp_kv_score_tid,
            )
            _q_result, qb_tid = cooperative_q_proj_q(
                x_normed_t, wq_b, wq_b_scale, freqs_cos, q_rope_sin_signed,
                q_rope_swap_idx, q, qr_i8_matmul, qr_scale_pad_store, cmp_kv_score_tid,
            )
            kv_proj_rope(
                x_normed_t, wkv, gamma_ckv, freqs_cos,
                q_rope_sin_signed, q_rope_swap_idx, kv, qb_tid,
            )

"""
    marker = "            # Bound indexer scratch to its own runtime scope.\n"
    source = source.replace(marker, insertion + writeback + marker, 1)
    old = section(
        source,
        "                _index_scores, index_indices = indexer(",
        "\n            # sparse_attn_csa folds",
    )
    source = source.replace(
        old,
        """                _index_scores, index_indices, _leaf_tid = indexer_weights_score(
                    x_normed_t, weights_proj, idx_query_i8, idx_query_scale,
                    idx_native_kv_cache, idx_block_table, idx_topk_scores, idx_topk,
                    position_ids, kv_seq_lens, idx_cache_write_tid,
                    cmp_kv_score_tid, idx_quant_tid, TP1_WEIGHTS_WORKERS,
                    max_seq_len, early_scale_layout, early_scale_deps[0],
                )
""",
        1,
    )
    ast.parse(source)
    path.write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="e91e9f17，冻结13259f92 COOPERATIVE_PREFIX",
                changes=[
                    "QA/QR与QB拆开编排；Indexer query投影→主Compressor投影→4核Hadamard/20核QB",
                    "原KV投影只等完整QB投影后启动；cache写回仍走原保护与状态逻辑",
                    "Indexer head weights沿来源等主压缩投影，不改变四K归约",
                ],
                preserved="各投影K序、量化、compact offsets、pool读后state提交、动态实际T及cache保护",
                adaptation="保留分task；KV使用完整QB的TaskId，不宣称复制了单SPMD逐核QB→KV的细粒度重叠",
                limits="已有Native入口T<=dense tile，TaskId只代表这一tile；完整收益待实测",
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
