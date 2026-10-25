# SPDX-License-Identifier: Apache-2.0
"""先取得Score/Bias，再发旧PV左半DMA，避免后续GM读取屏障提前排空预取。"""

import argparse
import ast
import difflib
import hashlib
import json
import shutil
import sys
import textwrap
from pathlib import Path

from prepare import replace_once

PRELOAD = """\
sm_scores_hold = pl.tile.create([H // 2, ATTN_K_TILE], dtype=pl.FP32)
sm_bias_hold = pl.tile.create([1, ATTN_K_TILE], dtype=pl.FP32)
if qk_tick > 0 and qk_tick <= qk_block_count:
    sm_work = qk_tick - 1
    sm_sb = sm_work % SPARSE_BLOCKS
    sm_t = qk_core + (sm_work // SPARSE_BLOCKS) * CORE_GROUPS
    sm_active_valid = pl.read(valid_block_mask, [sm_t, sm_sb])
    if sm_active_valid > 0:
        sm_row = (qk_core * QK_TRANSFER_SLOTS + sm_work % QK_TRANSFER_SLOTS) * H
        if t_dim < EARLY_KV_MIN_TOKENS:
            pl.system.sync_wait(
                EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
            )
        sm_scores = pl.load(score_transfer, [sm_row + qk_lane_head, 0], [H // 2, ATTN_K_TILE])
        sm_bias = pl.load(sparse_bias, [sm_t, sm_sb * ATTN_K_TILE], [1, ATTN_K_TILE])
        sm_scores_valid, sm_bias_valid = pl.yield_(sm_scores, sm_bias)
    else:
        sm_scores_valid, sm_bias_valid = pl.yield_(sm_scores_hold, sm_bias_hold)
    sm_active, prefetched_scores, prefetched_bias = pl.yield_(sm_active_valid, sm_scores_valid, sm_bias_valid)
else:
    sm_active, prefetched_scores, prefetched_bias = pl.yield_(pl.cast(0, pl.INT32), sm_scores_hold, sm_bias_hold)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", type=int, default=3)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    old = here.parent / "csa_spmd_tnd_eight_20261022"
    sys.path.insert(0, str(old))
    from prepare_attention_pv_prefetch import transform
    from pv_after_score_protocol import verify_all

    parent = here / "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd"
    name = f"pv_after_score_v{args.revision}"
    target = here / "candidates" / name
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "attention_stream_stages.py"
    original = path.read_text()
    source = transform(original, halves=1)
    source = replace_once(
        source,
        "            if pl.read(valid_block_mask, [softmax_t, softmax_sb]) > 0:\n",
        "            if sm_active > 0:\n",
    )
    wait = (
        "                if t_dim < EARLY_KV_MIN_TOKENS:\n"
        "                    pl.system.sync_wait(\n"
        "                        EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, "
        "core_type=pl.KernelType.AIV\n"
        "                    )\n"
    )
    source = replace_once(source, wait, "")
    source = replace_once(
        source,
        "                qk_scores_half = pl.load(score_transfer, [qk_transfer_row + qk_lane_head, 0], "
        "[H // 2, ATTN_K_TILE])\n",
        "                qk_scores_half = prefetched_scores\n",
    )
    source = replace_once(
        source,
        "                qk_bias = pl.load(sparse_bias, [softmax_t, qk_s0], [1, ATTN_K_TILE])\n",
        "                qk_bias = prefetched_bias\n",
    )
    source = replace_once(
        source,
        "        pv_left_hold = pl.tile.create([H // 2, HEAD_DIM // 2], dtype=pl.FP32)\n",
        textwrap.indent(PRELOAD, "        ")
        + "        pv_left_hold = pl.tile.create([H // 2, HEAD_DIM // 2], dtype=pl.FP32)\n",
    )
    # T>=144原本先等Score再发布新KV；预加载Score必须放在该等待之后。
    tree = ast.parse(source)
    vector = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "attention_qk_pv_vector_lane_stage"
    )
    loop = next(n for n in vector.body if isinstance(n, ast.For))
    early = next(
        n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == "t_dim >= EARLY_KV_MIN_TOKENS"
    )
    lines = source.splitlines(keepends=True)
    early_source = "".join(lines[early.lineno - 1 : early.end_lineno])
    del lines[early.lineno - 1 : early.end_lineno]
    source = "".join(lines)
    marker = "        sm_scores_hold = pl.tile.create([H // 2, ATTN_K_TILE], dtype=pl.FP32)\n"
    source = replace_once(source, marker, early_source + marker)
    ast.parse(source)
    path.write_text(source)
    before = ast.parse(original)
    after = ast.parse(source)
    assert ast.dump(before.body[-2]) == ast.dump(after.body[-2]), "Cube helper changed"

    def math_calls(tree):
        operations = {
            "mul",
            "add",
            "sub",
            "exp",
            "row_max",
            "row_sum",
            "cast",
            "maximum",
            "col_expand_add",
            "row_expand_sub",
            "row_expand_mul",
        }
        fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "attention_qk_pv_vector_lane_stage"
        )
        # Ignore the added INT32 predicate initialization; no arithmetic payload changes.
        return sorted(
            ast.dump(n)
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in operations
            and ast.unparse(n) != "pl.cast(0, pl.INT32)"
        )

    assert math_calls(before) == math_calls(after), "Numerical payload changed"
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(True),
            source.splitlines(True),
            fromfile="a/attention_stream_stages.py",
            tofile="b/attention_stream_stages.py",
        )
    )
    (here / (name + ".patch")).write_text(patch)
    report = {
        "source": str(target),
        "parent": "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd",
        "mechanism": "Score/Bias预加载及谓词读取在PV左半DMA前；随后softmax无新增GM读取",
        "numerical_math_calls_unchanged": True,
        "cube_helper_unchanged": True,
        "event_protocol": verify_all(),
        "historical_difference": "旧v2在PV预取后加载Score/Bias，PIPE_ALL排空预取；本候选先加载二者",
        "status": "prepared_not_adopted",
        "device_execution": False,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.glob("*.py"))},
    }
    (here / (name + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
