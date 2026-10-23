# SPDX-License-Identifier: Apache-2.0
"""Replace large-T serial META layout proof with the validated local block reader."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/local_scale_v1"
    expected = json.loads((here / "local_scale_v1.json").read_text())["files"]
    source = {p.name: p.read_text() for p in parent.glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == expected
    source["policy.py"] = replace_once(
        source["policy.py"],
        "scale_page_coalescing=(long_history and group == 6 "
        "and (parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS)),",
        "scale_page_coalescing=(long_history and group == 6 and parallel_small_qa),",
    )
    target = here / "runs/distributed_scale_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    baseline = json.loads((here / "baseline.json").read_text())
    original = json.loads(Path(baseline["cases"]["C05"]["source_report"]).read_text())
    description = dict(buffers=original["descriptors"]["spmd"], policy=original["policies"]["spmd"])
    assert description["policy"]["SCALE_PAGE_COALESCING"]
    description["policy"]["SCALE_PAGE_COALESCING"] = False
    # Binding allocates the layout proof suffix only when its strategy is active.
    description["buffers"]["group_offsets"]["shape"][0] -= 2 * original["case"]["request_capacity"]
    (here / "descriptors/C05_distributed_scale_v1.json").write_text(json.dumps(description, indent=2) + "\n")
    report = dict(
        source=str(target),
        parent="local_scale_v1",
        baseline="baseline.json",
        adopted=False,
        changes=["policy.py"],
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        mechanism=("Large-T uses per-score-block local page proof and exact legacy fallback; remove serial META proof"),
        unchanged="Score arithmetic, FIFO events, physical page validation and repeated-tail fallback",
        hypothesis="Reduced META critical path may offset repeated block proof; requires complete L1 evidence",
    )
    with (here / "distributed_scale_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
