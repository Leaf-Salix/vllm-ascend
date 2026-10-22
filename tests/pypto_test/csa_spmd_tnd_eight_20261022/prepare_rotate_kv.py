# SPDX-License-Identifier: Apache-2.0
"""Keep KV in the prefix, but transfer its logical ranks to earlier-finishing AICs."""

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
        "    QB_WIDE_N: pl.constexpr = False,\n    KV_TAIL_OWNERS: pl.constexpr = False,\n",
    )
    marker = "    kv_worker = (worker + GROUPS - BRANCH_GROUPS) % GROUPS\n"
    source[filename] = replace_once(
        source[filename],
        marker,
        marker + "    if KV_TAIL_OWNERS:\n        kv_worker = (worker + GROUPS - (2 * GROUPS // 3)) % GROUPS\n",
    )
    source["cube_schedule.py"] = replace_once(
        source["cube_schedule.py"],
        "            QB_WIDE_N,\n",
        "            QB_WIDE_N,\n            SPLIT_SCORE_ROWS and OA_COLS == 256,\n",
    )
    target = here / "runs/rotate_kv_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    report = dict(
        source=str(target),
        baseline="baseline.json",
        adopted=False,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        changes=["projection_overlap_stages.py", "cube_schedule.py"],
        guard="COOPERATIVE_PREFIX and SPLIT_SCORE_ROWS and OA_COLS==256",
        mechanism="Move eight active KV ranks in T144 from physical AIC4..11 to AIC16..23; retain prefix position",
        invariant="All 24 logical ranks remain a permutation; AIV still waits for all 24 actual physical KV publishers",
        unchanged="Mathematics, buffers, phase order, consumers and publish/wait counts",
        hypothesis="Earlier prefix cores may absorb KV work and reduce Score start skew; no performance claim yet",
    )
    with (here / "rotate_kv_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
