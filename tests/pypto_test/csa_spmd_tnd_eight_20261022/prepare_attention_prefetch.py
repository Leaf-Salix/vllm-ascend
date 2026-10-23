# SPDX-License-Identifier: Apache-2.0
# Exact source templates preserve the immutable compilation audited below.
# ruff: noqa: E501
"""Freeze early Score loads with a separate, compiler-visible KV DMA buffer."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once

PREFETCH = """        score_hold = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32)
        bias_hold = pl.create_tile([1, ATTN_K_TILE], dtype=pl.FP32)
        # Acquire the previous Score exactly once, before current-KV DMAs.
        # The math remains at its original position below the KV publication.
        if qk_tick > 0 and qk_tick <= qk_block_count:
            pf_work = qk_tick - 1
            pf_sb = pf_work % SPARSE_BLOCKS
            pf_t = qk_core + (pf_work // SPARSE_BLOCKS) * CORE_GROUPS
            if pl.read(valid_block_mask, [pf_t, pf_sb]) > 0:
                pf_slot = qk_core * QK_TRANSFER_SLOTS + pf_work % QK_TRANSFER_SLOTS
                pl.system.sync_wait(
                    EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                )
                pf_scores = pl.load(score_transfer, [pf_slot * H + qk_lane_head, 0], [H // 2, ATTN_K_TILE])
                pf_bias = pl.load(sparse_bias, [pf_t, pf_sb * ATTN_K_TILE], [1, ATTN_K_TILE])
                pf_score_valid, pf_bias_valid = pl.yield_(pf_scores, pf_bias)
            else:
                pf_score_valid, pf_bias_valid = pl.yield_(score_hold, bias_hold)
            prefetched_scores, prefetched_bias = pl.yield_(pf_score_valid, pf_bias_valid)
        else:
            # Uninitialized placeholders are never read when the same mask is false.
            prefetched_scores, prefetched_bias = pl.yield_(score_hold, bias_hold)
"""

GATHER = """        if qk_tick < qk_block_count:
            if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_tick % QK_TRANSFER_SLOTS
                qk_kv_row = qk_slot * ATTN_K_TILE
                if PLAN_PHYSICAL_ROWS and physical_mode != 0 and qk_sb > 0:
                    physical_column = (qk_sb - 1) * ATTN_K_TILE + qk_lane_kv
                    physical_rows = pl.load(
                        cmp_sparse_indices, [qk_t, physical_column], [1, ATTN_K_TILE // 2],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    for dma_half, (chunk_in,) in pl.range(2, init_values=(dma_kv_iter,)):
                        no_holes = pl.cast(0, pl.INT32)
                        # Keep DMA and optional Vector zero writes in separate
                        # loops; the old mixed-pipe row loop serialized each row.
                        for dma_row, (row_buf, hole_iter) in pl.range(
                            KV_DMA_ROWS, init_values=(chunk_in, no_holes),
                        ):
                            physical_row = pl.cast(pl.read(physical_rows, [0, dma_half * KV_DMA_ROWS + dma_row]), pl.INDEX)
                            if physical_row >= 0 and physical_row < pl.tensor.dim(cmp_kv_flat, 0):
                                dma_result = pl.gather_row(
                                    row_buf, cmp_kv_flat, [dma_row, 0], [physical_row, 0], [1, HEAD_DIM]
                                )
                                row_next, hole_next = pl.yield_(dma_result, hole_iter)
                            else:
                                row_next, hole_next = pl.yield_(row_buf, pl.cast(1, pl.INT32))
                            dma_loaded, needs_zero = pl.yield_(row_next, hole_next)
                        if needs_zero != 0:
                            for zero_row, (zero_buf,) in pl.range(KV_DMA_ROWS, init_values=(dma_loaded,)):
                                zero_index = pl.cast(pl.read(physical_rows, [0, dma_half * KV_DMA_ROWS + zero_row]), pl.INDEX)
                                if zero_index < 0 or zero_index >= pl.tensor.dim(cmp_kv_flat, 0):
                                    zero_result = pl.tile.assemble(zero_buf, dma_zero_row, [zero_row, 0])
                                    zero_next = pl.yield_(zero_result)
                                else:
                                    zero_next = pl.yield_(zero_buf)
                                dma_zeroed = pl.yield_(zero_next)
                            chunk_ready = pl.yield_(dma_zeroed)
                        else:
                            chunk_ready = pl.yield_(dma_loaded)
                        pl.store(
                            chunk_ready, [qk_kv_row + qk_lane_kv + dma_half * KV_DMA_ROWS, 0], kv_transfer
                        )
                        dma_finished = pl.yield_(chunk_ready)
                    dma_path = pl.yield_(dma_finished)
                else:
{fallback}
                    pl.store(qk_kv_half, [qk_kv_row + qk_lane_kv, 0], kv_transfer)
                    dma_path = pl.yield_(dma_kv_iter)
                dma_active = pl.yield_(dma_path)
            else:
                dma_active = pl.yield_(dma_kv_iter)
            dma_after = pl.yield_(dma_active)
        else:
            dma_after = pl.yield_(dma_kv_iter)
"""


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "merged_schedule_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    stream = source["attention_stream_stages.py"]
    begin = stream.index("@pl.jit.inline(auto_scope=False)\ndef attention_qk_pv_vector_lane_stage(")
    fast = stream[begin:]
    fast = replace_once(fast, "def attention_qk_pv_vector_lane_stage(", "def attention_qk_pv_vector_prefetch_stage(")
    fast = replace_once(
        fast,
        "    # 多走三拍排空末块归并；循环携带的m/l/oi仅属于当前query，\n",
        "    # A separate 32KiB DMA tile stays live through softmax and merge.\n"
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
    gather_begin = fast.index("        if qk_tick < qk_block_count:\n")
    gather_end = fast.index("        # 大档先消费上一块SCORE通知", gather_begin)
    original_gather = fast[gather_begin:gather_end]
    fallback_begin = original_gather.index("                else:\n") + len("                else:\n")
    fallback_end = original_gather.index("                pl.store(qk_kv_half,")
    fallback = original_gather[fallback_begin:fallback_end].rstrip()
    fast = fast[:gather_begin] + PREFETCH + GATHER.format(fallback=fallback) + fast[gather_end:]
    # Remove the original large-case Score wait; retain its KV_READY publication.
    large = fast.index("        if t_dim >= EARLY_KV_MIN_TOKENS:\n")
    wait_begin = fast.index("            if qk_tick > 0 and qk_tick <= qk_block_count:\n", large)
    wait_end = fast.index("            if qk_tick < qk_block_count:\n", wait_begin)
    fast = fast[:wait_begin] + fast[wait_end:]
    small_wait = """                if t_dim < EARLY_KV_MIN_TOKENS:
                    pl.system.sync_wait(
                        EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                    )
"""
    fast = replace_once(fast, small_wait, "")
    fast = replace_once(
        fast,
        "                qk_scores_half = pl.load(score_transfer, [qk_transfer_row + qk_lane_head, 0], [H // 2, ATTN_K_TILE])\n",
        "                qk_scores_half = prefetched_scores\n",
    )
    fast = replace_once(
        fast,
        "                qk_bias = pl.load(sparse_bias, [softmax_t, qk_s0], [1, ATTN_K_TILE])\n",
        "                qk_bias = prefetched_bias\n",
    )
    fast = replace_once(
        fast,
        "        running_m, running_l, running_left, running_right, local_stats_ring = pl.yield_(\n"
        "            m_after, l_after, left_after, right_after, stats_cache_ready\n",
        "        running_m, running_l, running_left, running_right, local_stats_ring, dma_kv_state = pl.yield_(\n"
        "            m_after, l_after, left_after, right_after, stats_cache_ready, dma_after\n",
    )
    source["attention_stream_stages.py"] = stream + "\n\nKV_DMA_ROWS = 32\n\n\n" + fast
    vector = source["vector_schedule.py"]
    vector = replace_once(
        vector,
        "from .attention_stream_stages import attention_qk_pv_vector_lane_stage\n",
        "from .attention_stream_stages import (\n"
        "    attention_qk_pv_vector_lane_stage, attention_qk_pv_vector_prefetch_stage,\n)\n",
    )
    begin = vector.index(
        "    kv_ring, probability_ring, mi_ring, li_ring, packed = attention_qk_pv_vector_lane_stage(\n"
    )
    end = vector.index("    signal = publish_phase(signal, PHASE_ATTN_VECTOR", begin)
    call = vector[begin:end]
    indent = lambda code: "".join("    " + line if line.strip() else line for line in code.splitlines(keepends=True))
    selected = "    if PLAN_PHYSICAL_ROWS:\n" + indent(
        call.replace("attention_qk_pv_vector_lane_stage", "attention_qk_pv_vector_prefetch_stage")
    )
    selected += "    else:\n" + indent(call)
    source["vector_schedule.py"] = vector[:begin] + selected + vector[end:]
    for content in source.values():
        ast.parse(content)
    target = here / "runs/attention_prefetch_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        (target / name).write_text(content)
    report = dict(
        source=str(target),
        parent="merged_schedule_v1.json",
        adopted=False,
        device_execution=False,
        mechanism="Load prior Score/bias before current KV; use a separate 32KiB two-pass DMA tile instead of a synchronizing MGATHER macro",
        unchanged="Math and output order, three GM slots, per-mask FIFO counts, native roots and the retained mechanisms",
        changed="SCORE_READY is consumed before current gather; existing size-dependent KV_READY/PROB_READY order remains",
        prior_negative="E44/E46 changed gather/buffers without this earlier Score load; those isolated experiments are not being repeated",
        gate="Full resource/ABI and actual sync audit; no NPU submission without an overlap path in emitted code",
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
    )
    with (here / "attention_prefetch_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
