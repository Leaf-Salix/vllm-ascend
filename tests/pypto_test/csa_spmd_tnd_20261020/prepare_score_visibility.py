# SPDX-License-Identifier: Apache-2.0
"""Hoist immutable per-query visibility out of long G6 segment iterations."""

import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    source = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    target = here / "runs/score_visibility_native_candidate_v1"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "indexer_score_stages.py"
    before = path.read_text()
    anchor = "                single_roots = pl.tile.full("
    assert before.count(anchor) == 1
    declarations = """                # Metadata is immutable during this invocation. Compute each
                # live query's visibility once per leaf, including every original
                # request/position/causal check. Empty lanes must not read positions.
                cached_visible0 = pl.cast(0, pl.INDEX)
                cached_visible1 = pl.cast(0, pl.INDEX)
                cached_visible2 = pl.cast(0, pl.INDEX)
                if query_group_size == 6:
                    if buf_score_lane * 3 < buf_live:
                        cached_visible0, _causal0 = indexer_query_visibility(
                            bounds, kv_seq_lens, position_ids, single_batch,
                            single_query + buf_score_lane * 3,
                        )
                    if buf_score_lane * 3 + 1 < buf_live:
                        cached_visible1, _causal1 = indexer_query_visibility(
                            bounds, kv_seq_lens, position_ids, single_batch,
                            single_query + buf_score_lane * 3 + 1,
                        )
                    if buf_score_lane * 3 + 2 < buf_live:
                        cached_visible2, _causal2 = indexer_query_visibility(
                            bounds, kv_seq_lens, position_ids, single_batch,
                            single_query + buf_score_lane * 3 + 2,
                        )
"""
    source_text = before.replace(anchor, declarations + anchor)
    original_call = """                                (single_query_visible, _causal) = indexer_query_visibility(
                                    bounds, kv_seq_lens, position_ids, single_batch, single_query + single_query_lane
                                )
"""
    assert source_text.count(original_call) == 1
    replacement = """                                if query_group_size == 6:
                                    single_query_visible = cached_visible0
                                    if single_local == 1:
                                        single_query_visible = cached_visible1
                                    elif single_local == 2:
                                        single_query_visible = cached_visible2
                                else:
                                    (single_query_visible, _causal) = indexer_query_visibility(
                                        bounds, kv_seq_lens, position_ids, single_batch,
                                        single_query + single_query_lane,
                                    )
"""
    source_text = source_text.replace(original_call, replacement)
    ast.parse(source_text)
    path.write_text(source_text)
    patch = "".join(difflib.unified_diff(
        before.splitlines(keepends=True), source_text.splitlines(keepends=True),
        fromfile="a/indexer_score_stages.py", tofile="b/indexer_score_stages.py",
    ))
    (here / "score_visibility_v1.patch").write_text(patch)
    report = {
        "source": str(source), "candidate": str(target),
        "changed_files": ["indexer_score_stages.py"],
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "scope": "Only balanced G6 AIV; no math, tile, leaf, store, notify, wait or ABI change",
        "device_execution": False,
    }
    (here / "score_visibility_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
