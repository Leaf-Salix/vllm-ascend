# SPDX-License-Identifier: Apache-2.0
"""Prepare an isolated G6 Key-L1 carry across 1024-column Score steps."""

import ast
import difflib
import json
import shutil
import textwrap
from pathlib import Path

from prepare import replace_once


def transform(source):
    start = source.index("@pl.jit.inline(auto_scope=False)\ndef _indexer_score_cube_split_store(")
    end = source.index("@pl.jit.inline(auto_scope=False)\ndef indexer_score_cube_stage(", start)
    original = source[start:end]
    helper = replace_once(original, "def _indexer_score_cube_split_store(", "def _indexer_score_cube_cross_step(")
    begin = helper.index("            else:\n                buf_cache_len =")
    prefix, full = helper[:begin], helper[begin:]
    pool = """                    buf_key_l1_pool = pl.tile.create(
                        [2 * qk_panel_cols, IDX_HEAD_DIM], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat
                    )
"""
    full = replace_once(full, pool, "")
    carry_pool = textwrap.indent(textwrap.dedent(pool), "                ")
    full = replace_once(
        full,
        "                for buf_score_step in pl.range(buf_score_iters):\n",
        "                # Slot0 carries the next step's first N64 in L1 only.\n"
        "                # No Right tile or Score FIFO slot lives longer than before.\n"
        + carry_pool
        + "                for buf_score_step in pl.range(buf_score_iters):\n",
    )
    prime_begin = full.index("                        buf_prime_col = 0\n")
    prime_end = full.index("                        if key_prefetch_to_l0:\n", prime_begin)
    prime = full[prime_begin:prime_end]
    full = (
        full[:prime_begin]
        + "                        if buf_score_step == 0:\n"
        + textwrap.indent(prime, "    ")
        + full[prime_end:]
    )
    full = replace_once(
        full,
        "                        if buf_qk_panel + key_prefetch_panels < score_tile // qk_panel_cols:\n",
        "                        if (buf_qk_panel + key_prefetch_panels < score_tile // qk_panel_cols\n"
        "                                or buf_score_step + 1 < buf_score_iters):\n",
    )
    full = replace_once(
        full,
        "                            if key_prefetch_panels > 0:\n"
        "                                if key_prefetch_to_l0:\n",
        "                            # The final prefetch is L1-only; keep terminal Right lifetimes.\n"
        "                            if key_prefetch_panels > 0:\n"
        "                                if key_prefetch_to_l0 and buf_qk_panel + key_prefetch_panels "
        "< score_tile // qk_panel_cols:\n",
    )
    helper = prefix + full
    wrapper_start = source.index("def indexer_score_cube_stage(")
    call_start = source.index("        result = _indexer_score_cube_split_store(\n", wrapper_start)
    call_end = source.index("    else:\n", call_start)
    call = source[call_start:call_end]
    dispatch = (
        "        if key_prefetch_panels == 1 and key_prefetch_to_l0:\n"
        + textwrap.indent(call.replace("_indexer_score_cube_split_store(", "_indexer_score_cube_cross_step("), "    ")
        + "        else:\n"
        + textwrap.indent(call, "    ")
    )
    result = source[:call_start] + dispatch + source[call_end:]
    result = result[:end] + helper + result[end:]
    ast.parse(result)
    return result


def main():
    here = Path(__file__).resolve().parent
    parent = here / "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd"
    target = here / "candidates/score_cross_step_v1"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "indexer_score_stages.py"
    before = path.read_text()
    after = transform(before)
    path.write_text(after)
    (here / "score_cross_step_v1.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/indexer_score_stages.py",
                tofile="b/indexer_score_stages.py",
            )
        )
    )
    report = {
        "candidate": str(target),
        "parent": str(parent),
        "scope": "existing split-G6 balanced N64/1024, prefetch=1, TO_L0=True, live>3",
        "mechanism": (
            "carry the next Score step's first N64 in existing L1 slot0; prefetch on the previous step's last panel"
        ),
        "key_data_bytes_and_mma_count_unchanged": True,
        "right_lifetimes_unchanged_by_design": True,
        "source_fallback": "original split helper retained for other prefetch settings; live<=3 copied exactly",
        "slot_contract": [
            "Step0 primes logical panel0 into L1 slot0, then extracts to the original Right current slot.",
            "Panel p<15 prefetches p+1 into L1 slot (p+1)%2 and into the original Right prefetch slot.",
            "Panel15 prefetches next-step panel0 into L1 slot0 only, iff another step exists.",
            "L1 slot0's prior panel14 was already read into Right before the panel15 prefetch.",
            "The next step keeps its original ACK before FIFO reuse and extracts carried L1 slot0 into Right.",
            "Final step does not prefetch past the leaf; original tail clamp and logical/physical page guards apply.",
        ],
        "status": "CPU_gate_pending",
        "device_execution": False,
        "performance_eligible": False,
        "adopted": False,
    }
    (here / "score_cross_step_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
