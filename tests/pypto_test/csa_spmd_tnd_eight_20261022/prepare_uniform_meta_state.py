# SPDX-License-Identifier: Apache-2.0
"""Validate actual proof writes on every equal/mixed/unequal-KV graph replay."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/uniform_meta_state_v1"
    shutil.copytree(here / "runs/uniform_state_harness_v1", target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "state_replay.py"
    source = path.read_text()
    marker = "        return (\n            values,\n            caches,\n"
    source = replace_once(source, marker,
        '        if side == "candidate":\n'
        '            entry = entries[side]\n'
        '            assert entry.policy.plan_physical_rows and entry.policy.query_group == 6\n'
        '            q = metadata["idx_query_start_loc"].tolist()\n'
        '            lens = metadata["kv_seq_lens"].tolist()\n'
        '            base = ((requests + 1 + 15) // 16) * 16\n'
        '            assert base > requests + 1\n'
        '            expected = int(\n'
        '                requests > 1 and capacity == requests * 6\n'
        '                and q[:requests + 1] == [r * 6 for r in range(requests + 1)]\n'
        '                and len(lens) == requests and lens[0] >= 6 and len(set(lens)) == 1\n'
        '            )\n'
        '            actual = int(entry.buffers["group_offsets"][base - 1].cpu())\n'
        '            geometry["uniform_meta_proof"] = actual == expected\n'
        + marker)
    ast.parse(source)
    path.write_text(source)
    launcher = replace_once((here / "launch_uniform_state.sh").read_text(),
                            'harness="$here/runs/uniform_state_harness_v1"',
                            'harness="$here/runs/uniform_meta_state_v1"')
    (here / "launch_uniform_meta_state.sh").write_text(launcher)
    report = dict(
        parent="uniform_state_harness_v1", source=str(target), metadata_cases=16, continuous_steps=4,
        performance_eligible=False,
        added_check="Actual candidate META proof equals an independent predicate from this replay's CPU metadata copy",
        transitions=["full equal -> unequal KV -> equal", "full equal -> inactive padding -> equal",
                     "reverse page owners with proof enabled", "mixed and fragmented metadata with proof disabled"],
        unchanged="All exact output/score/TopK/cache/state, guards, addresses, generation and prior geometry checks",
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "uniform_meta_state_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
