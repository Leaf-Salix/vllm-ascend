# SPDX-License-Identifier: Apache-2.0
"""Freeze a steady-state branch issuing softmax before explicit row DMAs."""

import ast
import hashlib
import json
import textwrap
from pathlib import Path

from prepare_attention_prefetch import GATHER
from prepare_scale_gather import replace_once


def indent(code, spaces):
    return textwrap.indent(textwrap.dedent(code), " " * spaces)


def make_stream(original):
    begin = original.index("@pl.jit.inline(auto_scope=False)\ndef attention_qk_pv_vector_lane_stage(")
    fast = original[begin:]
    fast = replace_once(fast, "def attention_qk_pv_vector_lane_stage(", "def attention_qk_pv_vector_overlap_stage(")
    gather_begin = fast.index("        if qk_tick < qk_block_count:\n")
    merge_begin = fast.index("        if qk_tick >= QK_PRE_LAUNCH + 1:\n", gather_begin)
    fallback = fast[gather_begin:merge_begin]
    math_begin = fallback.index("                qk_reduce_tmp = pl.create_tile(")
    math_end = fallback.index("                pl.store(qk_probability,")
    math = fallback[math_begin:math_end]
    commit_end = fallback.index("                stats_cache_valid = pl.yield_(stats_cache_updated_path)")
    commit = fallback[math_end:commit_end]
    dma_begin = GATHER.index("                    for dma_half,")
    dma_end = GATHER.index("                    dma_path = pl.yield_(dma_finished)", dma_begin)
    dma = GATHER[dma_begin:dma_end]

    # All predicate loads happen before the independent Vector work. The fast
    # branch is one control-flow region through softmax, DMA and publication.
    branch = """        overlap_ready = pl.cast(0, pl.INT32)
        if qk_tick > 0 and qk_tick < qk_block_count and qk_sb > 0:
            if PLAN_PHYSICAL_ROWS and physical_mode != 0:
                overlap_previous = qk_tick - 1
                overlap_previous_t = qk_core + (overlap_previous // SPARSE_BLOCKS) * CORE_GROUPS
                if (pl.read(valid_block_mask, [qk_t, qk_sb]) > 0
                        and pl.read(valid_block_mask, [overlap_previous_t, overlap_previous % SPARSE_BLOCKS]) > 0):
                    overlap_ready = pl.cast(1, pl.INT32)
        if overlap_ready != 0:
            softmax_work = qk_tick - 1
            softmax_sb = softmax_work % SPARSE_BLOCKS
            softmax_t = qk_core + (softmax_work // SPARSE_BLOCKS) * CORE_GROUPS
            qk_slot = qk_core * QK_TRANSFER_SLOTS + softmax_work % QK_TRANSFER_SLOTS
            qk_transfer_row = qk_slot * H
            qk_s0 = softmax_sb * ATTN_K_TILE
            pl.system.sync_wait(
                EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
            )
            physical_column = (qk_sb - 1) * ATTN_K_TILE + qk_lane_kv
            physical_rows = pl.load(
                cmp_sparse_indices, [qk_t, physical_column], [1, ATTN_K_TILE // 2],
                target_memory=pl.MemorySpace.Vec,
            )
"""
    branch += indent(math, 12)
    branch += "            qk_kv_row = (qk_core * QK_TRANSFER_SLOTS + qk_tick % QK_TRANSFER_SLOTS) * ATTN_K_TILE\n"
    branch += indent(dma, 12)
    branch += """            if t_dim >= EARLY_KV_MIN_TOKENS:
                pl.system.sync_set(
                    EVENT_BASE + QK_KV_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                )
"""
    branch += indent(commit, 12)
    branch += """            if t_dim < EARLY_KV_MIN_TOKENS:
                pl.system.sync_set(
                    EVENT_BASE + QK_KV_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                )
            stats_cache_after, dma_after = pl.yield_(stats_cache_updated_path, dma_finished)
        else:
"""
    branch += indent(fallback, 12)
    branch += "            stats_cache_after, dma_after = pl.yield_(stats_cache_ready, dma_kv_iter)\n"
    suffix = fast[merge_begin:].replace("stats_cache_ready", "stats_cache_after")
    fast = fast[:gather_begin] + branch + suffix
    fast = replace_once(
        fast,
        "    # 多走三拍排空末块归并；循环携带的m/l/oi仅属于当前query，\n",
        "    dma_kv_state = pl.create_tile([KV_DMA_ROWS, HEAD_DIM], dtype=pl.BF16)\n"
        "    dma_zero_row = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)\n"
        "    # 多走三拍排空末块归并；循环携带的m/l/oi仅属于当前query，\n",
    )
    fast = replace_once(
        fast,
        "(m_iter, l_iter, left_iter, right_iter, stats_cache_iter)",
        "(m_iter, l_iter, left_iter, right_iter, stats_cache_iter, dma_kv_iter)",
    )
    fast = replace_once(
        fast,
        "init_values=(running_m, running_l, running_left, running_right, local_stats_ring)",
        "init_values=(running_m, running_l, running_left, running_right, local_stats_ring, dma_kv_state)",
    )
    fast = replace_once(
        fast,
        "        running_m, running_l, running_left, running_right, local_stats_ring = pl.yield_(\n"
        "            m_after, l_after, left_after, right_after, stats_cache_after\n",
        "        running_m, running_l, running_left, running_right, local_stats_ring, dma_kv_state = pl.yield_(\n"
        "            m_after, l_after, left_after, right_after, stats_cache_after, dma_after\n",
    )
    return original + "\n\nKV_DMA_ROWS = 32\n\n\n" + fast


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "merged_schedule_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    source["attention_stream_stages.py"] = make_stream(source["attention_stream_stages.py"])
    vector = source["vector_schedule.py"]
    vector = replace_once(
        vector,
        "from .attention_stream_stages import attention_qk_pv_vector_lane_stage\n",
        "from .attention_stream_stages import attention_qk_pv_vector_lane_stage, "
        "attention_qk_pv_vector_overlap_stage\n",
    )
    call_name = "    kv_ring, probability_ring, mi_ring, li_ring, packed = attention_qk_pv_vector_lane_stage(\n"
    begin = vector.index(call_name)
    end = vector.index("    signal = publish_phase(signal, PHASE_ATTN_VECTOR", begin)
    call = vector[begin:end]
    selected = "    if PLAN_PHYSICAL_ROWS:\n"
    selected += indent(call.replace("attention_qk_pv_vector_lane_stage", "attention_qk_pv_vector_overlap_stage"), 8)
    selected += "    else:\n" + indent(call, 8)
    source["vector_schedule.py"] = vector[:begin] + selected + vector[end:]
    for content in source.values():
        ast.parse(content)
    target = here / "runs/attention_overlap_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        (target / name).write_text(content)
    report = dict(
        source=str(target),
        parent="merged_schedule_v1.json",
        adopted=False,
        device_execution=False,
        mechanism="Issue the unchanged previous-block softmax before explicit current-block row DMAs "
        "in one steady-state branch",
        unchanged="Root ABI, mathematical bodies, slot count, masked event counts, per-size KV/Prob notification order",
        changed="SCORE_READY/load/math precede current gather; probability publication remains after gather",
        fallback="Original complete gather/softmax/publication body for SWA, masked, first and drain ticks",
        gate="Actual C05 resource, alias and emitted synchronization audit before any NPU submission",
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
    )
    with (here / "attention_overlap_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
