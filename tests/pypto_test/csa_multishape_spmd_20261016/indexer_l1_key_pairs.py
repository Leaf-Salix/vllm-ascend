# SPDX-License-Identifier: Apache-2.0
"""N64循环版每次处理两面板，在第一块QK前提交第二块Key的GM→L1。

只让score双槽跨循环，Key/Right均在当前迭代创建和消费，避免前版Key
分支phi跨循环的别名问题。数学形状、FIX转换、1024交接、AIV及叶计划不变。
源顺序只是编译候选，实际预取重叠必须核对生成代码和完整L1测量。
"""

import ast
import json
from pathlib import Path

import indexer_l1_ring


def paired_math():
    lines = [
        "for ring_step in pl.range(buf_score_iters_inline8863__ssa_v0):",
        "    if ring_step >= 2:",
        "        pl.system.sync_wait(1, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)",
        "    ring_begin = buf_logical_begin_inline8767__ssa_v1 + ring_step * 1024",
        "    ring_transfer = buf_worker_inline8783__ssa_v0 * 12 + ring_step % 2 * 6",
        "    ring_score_pool = pl.tile.create([384, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)",
    ]

    def pages(indent, panel, tag):
        lines.append(" " * indent + f"{tag}_pool = pl.tile.create([64, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)")
        for page in range(2):
            key = f"{tag}_{page}"
            statements = [
                f"{key}_logical = pl.min((ring_begin + ({panel}) * 64 + {page * 32}) // 32, "
                "pl.max((buf_cache_len_inline8809__ssa_v0 - 1) // 32, 0))",
                f"{key}_physical = pl.max(pl.cast(pl.tensor.read(idx_block_table__ssa_v0, "
                f"[buf_batch_idx_inline8687__ssa_v0, {key}_logical]), pl.INDEX), 0)",
                f"{key}_byte = {key}_physical * native_page_bytes_inline8726__ssa_v0",
                f"if {key}_byte % 128 == 0:",
                f"    {tag}_pool = pl.tile.gather_row({tag}_pool, idx_kv_cache_inline8854__ssa_v0, "
                f"[{page * 32}, 0], [{key}_byte // 128, 0], [32, 128], transpose=False)",
                "else:",
                f"    {tag}_pool = pl.tile.gather_row({tag}_pool, idx_kv_cache_shift64_inline8790__ssa_v0, "
                f"[{page * 32}, 0], [{key}_byte // 128, 0], [32, 128], transpose=False)",
            ]
            lines.extend(" " * indent + line for line in statements)

    def key(indent, tag):
        lines.append(" " * indent + f"{tag}_right = pl.tile.extract(pl.tile.transpose_view({tag}_pool), "
                     "0, 0, [128, 64], target_memory=pl.Mem.Right)")

    def compute(indent, tag, panel, previous_slot, output_slot):
        # 先建WS Right，再提交QK，保持两者Right/Acc活跃区间相交，防止自动复用。
        statements = [
            f"{tag}_ws_right = pl.tile.extract(ring_score_pool, 0, {previous_slot * 64}, "
            "[384, 64], target_memory=pl.Mem.Right)",
            f"{tag}_qk = pl.tile.matmul(buf_query_left_inline8850__ssa_v0, {tag}_right)",
            f"{tag}_ws = pl.tile.matmul(buf_coefficient_pair_inline8676__ssa_v0, {tag}_ws_right)",
            f"ring_score_pool = pl.tile.assemble(ring_score_pool, {tag}_qk, [0, {output_slot * 64}], "
            "pre_quant=0.0009765625, pre_relu=True)",
            f"pl.tile.store(pl.tile.set_validshape({tag}_ws, 6, 64), "
            f"[ring_transfer, (({panel}) - 1) * 64], buf_score_transfer_inline8845__ssa_v0)",
        ]
        lines.extend(" " * indent + line for line in statements)

    pages(4, "0", "pair_prime")
    key(4, "pair_prime")
    lines.extend([
        "    pair_prime_qk = pl.tile.matmul(buf_query_left_inline8850__ssa_v0, pair_prime_right)",
        "    ring_score_pool = pl.tile.assemble(ring_score_pool, pair_prime_qk, [0, 0], pre_quant=0.0009765625, pre_relu=True)",
        "    for pair_panel in pl.range(1, 15, 2):",
    ])
    pages(8, "pair_panel", "pair_odd")
    key(8, "pair_odd")
    # 下一块Key只预取到L1，Right仍在当前块QK/WS之后提取。
    pages(8, "pair_panel + 1", "pair_even")
    compute(8, "pair_odd", "pair_panel", 0, 1)
    key(8, "pair_even")
    compute(8, "pair_even", "pair_panel + 1", 1, 0)
    pages(4, "15", "pair_last")
    key(4, "pair_last")
    compute(4, "pair_last", "15", 0, 1)
    lines.extend([
        "    pair_drain_right = pl.tile.extract(ring_score_pool, 0, 64, [384, 64], target_memory=pl.Mem.Right)",
        "    pair_drain_ws = pl.tile.matmul(buf_coefficient_pair_inline8676__ssa_v0, pair_drain_right)",
        "    pl.tile.store(pl.tile.set_validshape(pair_drain_ws, 6, 64), [ring_transfer, 960], buf_score_transfer_inline8845__ssa_v0)",
        "    pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)",
    ])
    return ast.parse("\n".join(lines)).body


def use_key_pairs(schedule, catalog, output):
    """复用原变换器的范围检查；局部替换生成器，函数退出即恢复。"""
    original = indexer_l1_ring.ring_math
    try:
        indexer_l1_ring.ring_math = paired_math
        result, audit = indexer_l1_ring.use_l1_ring(schedule, catalog, output)
    finally:
        indexer_l1_ring.ring_math = original
    audit.update(revision="v1_two_panels_local_key_prefetch", inner_loop_panels=2,
                 key_next_panel_prefetch=True, key_prefetch_target="L1",
                 full_key_panel_sequence=[0, *range(1, 15), 15],
                 score_store_panel_sequence=list(range(16)),
                 warning="两面板内提前提交GM到L1，不承诺生成代码已重叠；必须检查实际地址和等待")
    (Path(output) / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return result, audit
