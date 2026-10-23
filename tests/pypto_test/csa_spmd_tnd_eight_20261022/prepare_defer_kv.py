# SPDX-License-Identifier: Apache-2.0
"""Move independent original-KV projection behind Score in the existing large-T G6 tier."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    baseline = json.loads((here / "baseline.json").read_text())
    source = {p.name: p.read_text() for p in (here / "runs/baseline").glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == baseline["files"]
    filename = "projection_overlap_stages.py"
    source[filename] = replace_once(
        source[filename],
        "    QB_WIDE_N: pl.constexpr = False,\n",
        "    QB_WIDE_N: pl.constexpr = False,\n    DEFER_KV: pl.constexpr = False,\n",
    )
    original = (
        "    kv_worker = (worker + GROUPS - BRANCH_GROUPS) % GROUPS\n"
        "    kv_fp32 = kv_project_stage(normed, wkv, kv_fp32, kv_worker, GROUPS)\n"
        "    signal = publish_phase(signal, PHASE_KV_PROJECT, worker, CORES, generation)\n"
    )
    source[filename] = replace_once(
        source[filename],
        original,
        "    if not DEFER_KV:\n" + "".join("    " + line for line in original.splitlines(True)),
    )
    filename = "cube_schedule.py"
    source[filename] = replace_once(
        source[filename],
        "            QB_WIDE_N,\n",
        "            QB_WIDE_N,\n            SPLIT_SCORE_ROWS and OA_COLS == 256,\n",
    )
    marker = "    signal = publish_phase(signal, PHASE_SCORE_CUBE, worker, CORES, generation)\n"
    source[filename] = replace_once(
        source[filename],
        marker,
        marker + "\n    if COOPERATIVE_PREFIX and SPLIT_SCORE_ROWS and OA_COLS == 256:\n"
        "        # The local Score FIFO has drained; AIVs may now finish TopK while\n"
        "        # original-KV projection runs. Keep the exact logical owner and\n"
        "        # publish only after the original complete-K projection finishes.\n"
        + "".join("    " + line.replace("BRANCH_GROUPS", "PREFIX_BRANCH_GROUPS") for line in original.splitlines(True)),
    )
    target = here / "runs/defer_kv_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    original_report = json.loads(Path(baseline["cases"]["C05"]["source_report"]).read_text())
    descriptor = dict(buffers=original_report["descriptors"]["spmd"], policy=original_report["policies"]["spmd"])
    (here / "descriptors/C05_defer_kv_v1.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    report = dict(
        source=str(target),
        baseline="baseline.json",
        adopted=False,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        changes=["projection_overlap_stages.py", "cube_schedule.py"],
        evidence="baseline_C05_phase_sdk_v1.json: AIC4..7 are consistently late in KV_PROJECT and SCORE_CUBE",
        guard="COOPERATIVE_PREFIX and SPLIT_SCORE_ROWS and OA_COLS==256 (existing large-T long-G6 resource tier)",
        mechanism="Original KV follows local Score FIFO drain, before Attention; overlap it with the Vector tail",
        unchanged="KV/Score math, buffers, producer ownership, publish/wait counts, FIFO drain and cache writes",
        risk="Could delay KV normalization/TopK instead of shortening full critical path; full-L1 measurement required",
    )
    with (here / "defer_kv_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
