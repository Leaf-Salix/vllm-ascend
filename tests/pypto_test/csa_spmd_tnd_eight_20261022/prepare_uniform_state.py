# SPDX-License-Identifier: Apache-2.0
"""Add equal/mixed/equal and unequal-KV replay coverage without timing changes."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/uniform_state_harness_v1"
    shutil.copytree(here / "runs/priority_state_harness_v1", target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "state_replay.py"
    source = path.read_text()
    marker = '        ("priority_all_s6", [6] * requests, [history] * requests, None),\n'
    source = replace_once(source, marker, marker +
        '        ("full_unequal_kv", [6] * requests, [history + (r == 0) for r in range(requests)], None),\n'
        '        ("full_equal_kv_again", [6] * requests, [history] * requests, None),\n'
        '        ("one_inactive_request", [6] * (requests - 1), [history] * (requests - 1), None),\n'
        '        ("full_equal_after_padding", [6] * requests, [history] * requests, None),\n')
    marker = '        expected, next_caches = None, {}\n'
    source = replace_once(source, marker,
        '        # CPU copies belong only to this validator; production never reads them back.\n'
        '        q = metadata["idx_query_start_loc"].tolist()\n'
        '        lengths = metadata["kv_seq_lens"].tolist()\n'
        '        step = q[1] if len(q) > 1 else 0\n'
        '        uniform = (\n'
        '            requests > 1 and len(q) >= requests + 1 and len(lengths) == requests\n'
        '            and step > 0 and requests * step == capacity\n'
        '            and q[:requests + 1] == [r * step for r in range(requests + 1)]\n'
        '            and lengths[0] >= step and len(set(lengths)) == 1\n'
        '        )\n'
        '        row["expected_uniform_width"] = step if uniform else 0\n'
        '        if label in ("priority_all_s6", "full_equal_kv_again", "full_equal_after_padding",\n'
        '                     "priority_reverse_owners"):\n'
        '            assert row["expected_uniform_width"] == 6\n'
        '        elif label in ("full_unequal_kv", "one_inactive_request"):\n'
        '            assert row["expected_uniform_width"] == 0\n' + marker)
    ast.parse(source)
    path.write_text(source)
    launcher = replace_once(
        (here / "launch_priority_state.sh").read_text(),
        'harness="$here/runs/priority_state_harness_v1"',
        'harness="$here/runs/uniform_state_harness_v1"',
    )
    (here / "launch_uniform_state.sh").write_text(launcher)
    report = dict(
        parent="priority_state_harness_v1", metadata_cases=16, continuous_steps=4,
        performance_eligible=False,
        additions=["full unequal KV", "equal again", "one inactive request", "equal after padding"],
        observed_predicate="Expected width computed from actual CPU metadata copies each replay; not a device trace",
        unchanged="All exact output/score/TopK/cache/state, generation, address and guard validation",
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "uniform_state_harness_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
