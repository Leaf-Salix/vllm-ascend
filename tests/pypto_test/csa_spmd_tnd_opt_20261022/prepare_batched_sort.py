# SPDX-License-Identifier: Apache-2.0
"""Batch three full 2048-score segments, keeping query boundaries and the merge tree."""

import argparse
import ast
import hashlib
import json
import shutil
import textwrap
from pathlib import Path

BATCH_PREFIX = """
batch_sort_valid = pl.cast(0, pl.INT32)
if query_group_size == 6 and not SHORT_TOPK_PREFIX:
    if (buf_score_lane + 1) * 3 <= buf_live and single_segment_span == 2048:
        batch_sort_visible = single_visible
        for batch_check_local in pl.unroll(3):
            batch_check_query = single_query + buf_score_lane * 3 + batch_check_local
            batch_check_visible, _batch_check_causal = indexer_query_visibility(
                bounds, kv_seq_lens, position_ids, single_batch, batch_check_query
            )
            batch_sort_visible = pl.min(batch_sort_visible, batch_check_visible)
        if single_segment_begin + 2048 <= batch_sort_visible:
            batch_sort_valid = pl.cast(1, pl.INT32)
if batch_sort_valid > 0:
    # Only three complete, individually valid segments enter this path.
    # Each 4096-FP32 pair row remains independent at all three merge levels:
    # block lengths 64/256/1024 use repeat counts 48/12/3 on the flat tile.
    batch_sort_flat = pl.tile.reshape(single_segment_rows, [1, (query_group_size // 2) * 2048])
    batch_sort_scores = pl.maximum(batch_sort_flat, FP32_NEG_INF)
    batch_sort_index_row = pl.add(
        pl.tile.arange(0, [1, 2048], dtype=pl.INT32), pl.cast(single_segment_begin, pl.INT32)
    )
    batch_sort_index_shape = pl.tile.create([query_group_size // 2, 2048], dtype=pl.INT32)
    batch_sort_indices = pl.col_expand(batch_sort_index_shape, batch_sort_index_row)
    batch_sort_flat_indices = pl.tile.reshape(batch_sort_indices, [1, (query_group_size // 2) * 2048])
    batch_sort_pairs = pl.sort32(batch_sort_scores, pl.reinterpret_view(batch_sort_flat_indices, pl.UINT32))
    batch_sort_pairs = pl.mrgsort(batch_sort_pairs, block_len=64)
    batch_sort_pairs = pl.mrgsort(batch_sort_pairs, block_len=256)
    batch_sort_pairs = pl.mrgsort(batch_sort_pairs, block_len=1024)
    batch_sort_rows = pl.tile.reshape(batch_sort_pairs, [query_group_size // 2, 4096])
    for batch_sort_local in pl.unroll(query_group_size // 2):
        batch_sort_new_root = pl.tile.extract(
            batch_sort_rows, batch_sort_local, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
        )
        if single_step < 2:
            single_roots = pl.tile.assemble(single_roots, batch_sort_new_root, [batch_sort_local, 0])
        else:
            batch_sort_old_root = pl.tile.extract(
                single_roots, batch_sort_local, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
            )
            batch_sort_merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
            batch_sort_merged = pl.tile.mrgsort(batch_sort_new_root, batch_sort_old_root, tmp=batch_sort_merge_tmp)
            batch_sort_next_root = pl.tile.extract(
                batch_sort_merged, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
            )
            single_roots = pl.tile.assemble(single_roots, batch_sort_next_root, [batch_sort_local, 0])
else:
"""


def transform(source):
    candidates = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.If)
        and ast.unparse(n.test)
        == ("query_group_size != 2 and (single_step % 2 == 1 or single_step + 1 == single_steps)")
    ]
    assert len(candidates) == 1
    loop = candidates[0].body[-1]
    assert isinstance(loop, ast.For) and ast.unparse(loop.target) == "single_local"
    lines = source.splitlines(keepends=True)
    original = "".join(lines[loop.lineno - 1 : loop.end_lineno])
    replacement = textwrap.indent(textwrap.dedent(BATCH_PREFIX).lstrip(), " " * loop.col_offset)
    replacement += textwrap.indent(original, "    ")
    lines[loop.lineno - 1 : loop.end_lineno] = [replacement]
    result = "".join(lines)
    ast.parse(result)
    # The complete old per-query path is retained verbatim as the fallback.
    assert textwrap.indent(original, "    ") in result
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="batched_sort3_v2")
    args = parser.parse_args()
    reference = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*.py")} == expected
    target = here / "runs" / args.name
    shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = transform((reference / "indexer_score_stages.py").read_text())
    (target / "indexer_score_stages.py").write_text(source)
    report = {
        "source": str(target),
        "reference_kind": "frozen_tnd_v122",
        "adopted": False,
        "device_execution": False,
        "scope": "G6 without short-prefix; three fully valid 2048 segments only",
        "preserved": [
            "logical query identities",
            "sort32 input index order",
            "64/256/1024 merge tree",
            "new-root-before-old-root merge",
            "all partial/tail/stale-query fallback",
            "all FIFO protocol",
        ],
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    }
    with (here / f"{args.name}.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
