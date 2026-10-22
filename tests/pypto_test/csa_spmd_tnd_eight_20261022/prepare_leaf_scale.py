# SPDX-License-Identifier: Apache-2.0
"""Port the MPMD leaf-local proof into the frozen single-SPMD implementation."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from prepare_scale_gather import replace_once


def replace_node(source, node, replacement):
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1 : node.end_lineno] = [" " * node.col_offset + s + "\n" for s in replacement.splitlines()]
    return "".join(lines)


def main():
    here = Path(__file__).resolve().parent
    baseline = json.loads((here / "baseline.json").read_text())
    source = {p.name: p.read_text() for p in (here / "runs/baseline").glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == baseline["files"]
    upstream = "d263bc3f20aaf62d7b1f1f0175cf10c167785e20"
    filename = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd/indexer_scale_layout_stages.py"
    helper = subprocess.check_output(["git", "show", upstream + ":" + filename], text=True)
    source["leaf_scale_layout_stages.py"] = helper
    filename = "indexer_score_stages.py"
    source[filename] = replace_once(
        source[filename],
        "from .indexer_scale_layout_stages import LAYOUT_WORDS_PER_REQUEST, indexer_scale_layout_base",
        "from .leaf_scale_layout_stages import indexer_leaf_scale_layout",
    )
    nodes = [
        n
        for n in ast.walk(ast.parse(source[filename]))
        if isinstance(n, ast.If) and ast.unparse(n.test) == "scale_page_coalescing"
    ]
    assert len(nodes) == 1
    source[filename] = replace_node(
        source[filename],
        nodes[0],
        """if scale_page_coalescing:
    single_first_page, single_contiguous = indexer_leaf_scale_layout(
        idx_block_table, single_batch, single_page_begin, single_page_count,
        pl.tensor.dim(idx_native_kv_cache, 0),
    )""",
    )
    filename = "vector_schedule.py"
    source[filename] = replace_once(
        source[filename], "from .indexer_scale_layout_stages import indexer_scale_layout_plan_stage\n", ""
    )
    nodes = [
        n
        for n in ast.walk(ast.parse(source[filename]))
        if isinstance(n, ast.If) and ast.unparse(n.test) == "BALANCED and SCALE_PAGE_COALESCING"
    ]
    assert len(nodes) == 1
    source[filename] = replace_node(
        source[filename], nodes[0], "# Layout proof is performed by each actual Score leaf consumer."
    )
    source["policy.py"] = replace_once(
        source["policy.py"],
        "scale_page_coalescing=(long_history and group == 6 "
        "and (parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS)),",
        "scale_page_coalescing=(long_history and group == 6),",
    )
    target = here / "runs/leaf_scale_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    for cid in ("C04", "C05"):
        original = json.loads(Path(baseline["cases"][cid]["source_report"]).read_text())
        description = dict(buffers=original["descriptors"]["spmd"], policy=original["policies"]["spmd"])
        if not description["policy"]["SCALE_PAGE_COALESCING"]:
            description["buffers"]["group_offsets"]["shape"][0] += 2 * original["case"]["request_capacity"]
        description["policy"]["SCALE_PAGE_COALESCING"] = True
        (here / f"descriptors/{cid}_leaf_scale_v1.json").write_text(json.dumps(description, indent=2) + "\n")
    report = dict(
        source=str(target),
        baseline="baseline.json",
        adopted=False,
        helper_source_commit=upstream,
        helper_source_path="vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd/indexer_scale_layout_stages.py",
        helper_copied_verbatim=True,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        changes=["policy.py", "vector_schedule.py", "indexer_score_stages.py", "leaf_scale_layout_stages.py"],
        mechanism="Each Score AIV proves its leaf pages once; remove serial META layout proof",
        unchanged="Score/TopK arithmetic and FIFO; preserve original continuous and fallback readers",
        workspace="Keep existing optional 2*B suffix allocated but unused; no new cross-core proof ownership",
        performance_evidence_reused=False,
    )
    with (here / "leaf_scale_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
