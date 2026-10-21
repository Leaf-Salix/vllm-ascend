# SPDX-License-Identifier: Apache-2.0
"""B1 full-half sorting overlap, retaining every partial-leaf fallback and FIFO edge."""

import argparse
import ast
import hashlib
import json
import shutil
import textwrap
from pathlib import Path

STREAM_INIT = """
legacy_stream_full = pl.cast(0, pl.INT32)
legacy_stream_root = pl.tile.create([1, TOPK_PAIR_WIDTH], dtype=pl.FP32)
if query_group_size == 2:
    if buf_score_lane < buf_live and single_steps == 4 and single_capacity == 4096:
        stream_visible, _stream_causal = indexer_query_visibility(
            bounds, kv_seq_lens, position_ids, single_batch, single_query + buf_score_lane
        )
        if stream_visible - single_begin >= 4096:
            legacy_stream_full = pl.cast(1, pl.INT32)
"""

STREAM_STEP = """
if query_group_size == 2 and legacy_stream_full > 0 and single_step % 2 == 1:
    # The old 4096 branch sorts two independent 2048 lists through block_len=1024.
    # Start the first list after its two FIFO reads, while AIC computes the second.
    stream_half = pl.tile.extract(
        legacy_half_scores, single_step - 1, 0, [2, BUFFERED_LONG_SCORE_TILE],
        target_memory=pl.MemorySpace.Vec,
    )
    stream_flat = pl.tile.reshape(stream_half, [1, 2048])
    stream_scores = pl.maximum(stream_flat, FP32_NEG_INF)
    stream_begin = single_begin + (single_step - 1) * BUFFERED_LONG_SCORE_TILE
    stream_indices = pl.add(pl.tile.arange(0, [1, 2048], dtype=pl.INT32), pl.cast(stream_begin, pl.INT32))
    stream_pairs = pl.sort32(stream_scores, pl.reinterpret_view(stream_indices, pl.UINT32))
    stream_pairs = pl.mrgsort(stream_pairs, block_len=64)
    stream_pairs = pl.mrgsort(stream_pairs, block_len=256)
    stream_pairs = pl.mrgsort(stream_pairs, block_len=1024)
    stream_new_root = pl.tile.extract(
        stream_pairs, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec,
    )
    if single_step == 1:
        legacy_stream_root = stream_new_root
    else:
        stream_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        # Preserve the original medium_right -> medium_left tie priority.
        stream_merged = pl.tile.mrgsort(stream_new_root, legacy_stream_root, tmp=stream_tmp)
        legacy_stream_root = pl.tile.extract(
            stream_merged, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec,
        )
"""


def stream_source(source):
    lines = source.splitlines(keepends=True)
    nodes = list(ast.walk(ast.parse(source)))
    loop = next(n for n in nodes if isinstance(n, ast.For) and ast.unparse(n.target) == "single_step")
    fallback = next(n for n in nodes if isinstance(n, ast.If) and ast.unparse(n.test) == "legacy_valid > 0")
    old_branch = "".join(lines[fallback.lineno - 1 : fallback.end_lineno])
    prefix = " " * fallback.col_offset
    replacement = prefix + "if legacy_stream_full > 0:\n"
    replacement += prefix + "    pl.store(legacy_stream_root, [legacy_slot, 0], pair_arena)\n"
    replacement += prefix + "else:\n" + textwrap.indent(old_branch, "    ")
    # Changes are applied in descending line order to retain AST positions.
    lines[fallback.lineno - 1 : fallback.end_lineno] = [replacement]
    lines[loop.end_lineno : loop.end_lineno] = [
        textwrap.indent(textwrap.dedent(STREAM_STEP).lstrip(), " " * (loop.col_offset + 4))
    ]
    lines[loop.lineno - 1 : loop.lineno - 1] = [
        textwrap.indent(textwrap.dedent(STREAM_INIT).lstrip(), " " * loop.col_offset)
    ]
    result = "".join(lines)
    ast.parse(result)
    assert textwrap.indent(old_branch, "    ") in result
    return result


def skip_unused_fills(source):
    lines = source.splitlines(keepends=True)
    nodes = list(ast.walk(ast.parse(source)))
    roots = next(
        n
        for n in nodes
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "single_roots"
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.tile.full"
    )
    segments = next(
        n
        for n in nodes
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "single_segments"
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.tile.full"
    )
    assert segments.lineno == roots.end_lineno + 1
    old = "".join(lines[roots.lineno - 1 : segments.end_lineno])
    replacement = """
single_roots = pl.tile.create([query_group_size // 2, TOPK_PAIR_WIDTH], dtype=pl.FP32)
single_segments = pl.tile.create([query_group_size, BUFFERED_LONG_SCORE_TILE], dtype=pl.FP32)
if query_group_size != 2:
"""
    lines[roots.lineno - 1 : segments.end_lineno] = [
        textwrap.indent(textwrap.dedent(replacement).lstrip(), " " * roots.col_offset) + textwrap.indent(old, "    ")
    ]
    result = "".join(lines)
    ast.parse(result)
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("stream", "unused_fills"), required=True)
    args = parser.parse_args()
    reference = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*.py")} == expected
    name = "b1_sort_" + args.kind
    target = here / "runs" / name
    source = (reference / "indexer_score_stages.py").read_text()
    revised = stream_source(source) if args.kind == "stream" else skip_unused_fills(source)
    shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "indexer_score_stages.py").write_text(revised)
    report = {
        "source": str(target),
        "reference_kind": "frozen_tnd_v122",
        "adopted": False,
        "device_execution": False,
        "kind": args.kind,
        "scope": "balanced G2; stream only fully visible 4096 half-leaf, keep all partial fallbacks",
        "preserved": ["logical leaf boundaries", "sort and merge tree", "right before left ties", "all FIFO edges"],
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    }
    with (here / f"{name}.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
