"""Freeze the retained operator and fuse its Sparse final publication on a private copy."""

import difflib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
BASE = WORKSPACE / ".cache/csa-score-query-split-4ffccb7b-baseline"
PREFIX = WORKSPACE / ".cache/csa-sparse-final-publish-4ffccb7b"
OLD_PACKAGE = "dsv4_csa_score_query_split_4ffccb7b"
PACKAGE = "dsv4_csa_sparse_final_publish_4ffccb7b"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected one anchor: {before[:100]}")
    return text.replace(before, after)


def fuse_final_publication(text):
    # Preserve the public helper's signature and arithmetic. Only the inner
    # Sparse helper now publishes the final packed BF16 output itself.
    helper_end = text.index("\n\n@pl.jit.inline\ndef sparse_attn_csa_tp1(")
    inner, wrapper = text[:helper_end], text[helper_end:]
    inner = replace_once(inner, "H_TILE = 16\n", "H_TILE = 16\n\nFINAL_HEAD_TILE = H_TILE\n")
    inner = replace_once(inner,
        "    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],\n):",
        "    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],\n"
        "    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],\n"
        ") -> tuple[pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16], pl.Scalar[pl.TASK_ID]]:")
    inner = replace_once(inner,
        '    """Plan and run CSA QK/PV over sparse blocks, and build inverse-RoPE metadata."""',
        '    """Run CSA QK/PV and publish its final normalized, inverse-RoPE packed heads."""')
    rope_start = inner.index("    # Native cosine rows already have the consumer's interleaved layout.")
    return_start = inner.index("    return (", rope_start)
    rope_setup = inner[rope_start:return_start]
    inner = inner[:rope_start] + "    return o_packed_heads, qk_tid\n"
    inner = replace_once(inner, "    # QK/PV scratch tensors.\n", rope_setup + "    # QK/PV scratch tensors.\n")
    inner = replace_once(inner,
        "    attn_mi = pl.create_tensor([t_heads, 1], dtype=pl.FP32)\n"
        "    attn_li = pl.create_tensor([t_heads, 1], dtype=pl.FP32)\n"
        "    attn_oi = pl.create_tensor([t_heads, HEAD_DIM], dtype=pl.FP32)\n", "")
    inner = replace_once(inner, 'name_hint="qk_pv", deps=[qk_plan_tid],',
                         'name_hint="qk_pv", deps=[qk_plan_tid, rope_tid],')
    # Keep the exact existing INT32 lane map; do not introduce a new RoPE formula.
    index_start = wrapper.index("        m_idx = pl.tile.ci(")
    index_end = wrapper.index("        for m_idx in pl.range(", index_start)
    lane_map = wrapper[index_start:index_end].replace("H_TILE", "FINAL_HEAD_TILE")
    lane_map = lane_map.replace(
        "        m_gather_tmp = pl.create_tile([FINAL_HEAD_TILE, ROPE_DIM], dtype=pl.INT32)\n", "")
    lane_map = "\n".join("    " + line if line else line for line in lane_map.splitlines()) + "\n"
    inner = replace_once(inner,
        "            qk_lane_kv = qk_aiv * (ATTN_K_TILE // 2)\n",
        "            qk_lane_kv = qk_aiv * (ATTN_K_TILE // 2)\n" + lane_map)
    # The reduction scratch carries no state across softmax blocks. Its original
    # function-wide lifetime needlessly reserves 16 KiB across all PV updates.
    reduce_scratch = (
        "            qk_reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], "
        "dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)\n")
    inner = replace_once(inner, reduce_scratch, "")
    inner = replace_once(inner, "                        qk_scores_half = pl.load(\n",
                         "            " + reduce_scratch + "                        qk_scores_half = pl.load(\n")
    # Keep the original 16-head publication arithmetic and output grouping.
    publish = '''                        # Native SCFA normalizes inside its last PV update. Keep
                        # PTO arithmetic, but consume the live accumulators instead
                        # of publishing FP32 mi/li/oi for a separate merge task.
                        m_gather_tmp = pl.create_tile([FINAL_HEAD_TILE, ROPE_DIM], dtype=pl.INT32)
                        for pub_h in pl.unroll((H // 2) // H_TILE):
                            pub_local_head = pub_h * H_TILE
                            m_h0 = qk_lane_head + pub_local_head
                            m_mi = pl.slice(m_valid, [H_TILE, 1], [pub_local_head, 0])
                            m_li = pl.slice(l_valid, [H_TILE, 1], [pub_local_head, 0])
                            m_left = pl.slice(left_valid, [H_TILE, HEAD_DIM // 2], [pub_local_head, 0])
                            m_right = pl.slice(right_valid, [H_TILE, HEAD_DIM // 2], [pub_local_head, 0])
                            m_oi = pl.concat(m_left, m_right)
                            n_sink_bias = pl.load(attn_sink_col, [m_h0, 0], [H_TILE, 1])
                            n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
                            n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
                            n_full = pl.row_expand_div(m_oi, n_denom)
                            n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
                            m_rope = n_full[0:H_TILE, NOPE_DIM:HEAD_DIM]
                            m_cos_il = pl.load(freqs_cos, [pv_t, 0], [1, ROPE_DIM])
                            m_sin_signed = pl.load(rope_sin_signed, [pv_t, 0], [1, ROPE_DIM])
                            m_swapped = pl.tile.gather(n_full, m_swap_idx, m_gather_tmp)
                            m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il),
                                           pl.col_expand_mul(m_swapped, m_sin_signed))
                            n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")
                            n_full_bf16 = pl.concat(n_bf16[0:H_TILE, 0:NOPE_DIM], n_rope_bf16)
                            n_group_bf16 = pl.reshape(n_full_bf16, [PUBLISH_GROUPS, O_GROUP_IN])
                            n_pack_first = n_group_bf16[0:1, 0:O_GROUP_IN]
                            n_pack_second = n_group_bf16[1:2, 0:O_GROUP_IN]
                            n_pack_row = (m_h0 // HEADS_PER_GROUP) * T_PAD + pv_t
                            n_pack_row_second = n_pack_row + T_PAD
                            pl.store(n_pack_first, [n_pack_row, 0], o_packed_heads)
                            pl.store(n_pack_second, [n_pack_row_second, 0], o_packed_heads)
'''
    publish = publish.replace("H_TILE", "FINAL_HEAD_TILE")
    inner = replace_once(inner,
        "                        qk_output_row = pv_t * H + qk_lane_head\n"
        "                        pl.store(m_valid, [qk_output_row, 0], attn_mi)\n"
        "                        pl.store(l_valid, [qk_output_row, 0], attn_li)\n"
        "                        pl.store(left_valid, [qk_output_row, 0], attn_oi)\n"
        "                        pl.store(right_valid, [qk_output_row, HEAD_DIM // 2], attn_oi)\n", publish)
    signature_end = wrapper.index('    """Write CSA heads')
    wrapper = wrapper[:signature_end].replace(
        "tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]",
        "tuple[pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16], pl.Scalar[pl.TASK_ID]]",
    ) + wrapper[signature_end:]
    signature_end = wrapper.index('    """Write CSA heads')
    wrapper = wrapper[:signature_end] + '''    """Publish packed CSA heads from the final QK/PV vector update."""
    packed, ready = sparse_attn_csa(
        q, ori_kv, ori_block_table, cmp_kv, cmp_block_table, idx_topk,
        position_ids, seq_lens, attn_sink, freqs_cos, freqs_sin, o_packed_heads,
    )
    return packed, ready
'''
    return inner + wrapper


def main():
    if (ROOT / "task.txt").exists():
        raise RuntimeError("Never edit a queued experiment")
    for side in ("baseline", "candidate"):
        dest = Path(f"{PREFIX}-{side}")
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        package_root = dest / "vllm_ascend/ops/pypto"
        (package_root / OLD_PACKAGE).rename(package_root / PACKAGE)
    relative = Path("vllm_ascend/ops/pypto") / PACKAGE / "decode_sparse_attn_csa.py"
    baseline = Path(f"{PREFIX}-baseline") / relative
    candidate = Path(f"{PREFIX}-candidate") / relative
    before = baseline.read_text()
    after = fuse_final_publication(before)
    compile(after, str(candidate), "exec")
    candidate.write_text(after)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile=f"a/{relative}", tofile=f"b/{relative}")))
    (ROOT / "source.json").write_text(json.dumps({
        "baseline": "4ffccb7b", "source_prefix": str(PREFIX), "base_source": str(BASE),
        "variant": f"pkg:{PACKAGE}", "cases": [[131072, 16], [8192, 24]],
        "native_reference": "ops-transformer 28f40354 Sparse SCFA Vector last-S2 RowDivs/publication",
        "arithmetic": "Keep PTO softmax/rescaling/division/casts/inverse RoPE and original 16-head output tile",
    }, indent=2) + "\n")
    origin = ROOT.parent / "csa_score_query_split_20260929"
    for name in ("compile.py", "run.sh", "run_side.sh"):
        text = (origin / name).read_text().replace("csa_score_query_split", "csa_sparse_final_publish")
        text = text.replace("csa-score-query-split", "csa-sparse-final-publish")
        text = text.replace("query-partitioned AIV candidate", "fused final-publication candidate")
        if name == "compile.py":
            text = text.replace('platform="a2a3", save_kernels=True',
                                'platform="a2a3", dump_passes=True, save_kernels=True')
        (ROOT / name).write_text(text)
    print(PREFIX)


if __name__ == "__main__":
    main()
