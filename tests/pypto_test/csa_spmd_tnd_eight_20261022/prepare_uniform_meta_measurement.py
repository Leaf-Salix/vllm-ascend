# SPDX-License-Identifier: Apache-2.0
"""Read back the actual META proof outside all complete-L1 timing windows."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/uniform_meta_measurement_v1"
    shutil.copytree(here / "runs/sdk_measurement_harness_v1", target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "run_pto.py"
    source = path.read_text()
    marker = "                return state, checks\n"
    source = replace_once(source, marker,
        '                if side == "candidate":\n'
        '                    entry = entries[side]\n'
        '                    assert entry.policy.plan_physical_rows and entry.policy.query_group == 6\n'
        '                    requests = entry.buffers["idx_lengths"].shape[0]\n'
        '                    base = ((requests + 1 + 15) // 16) * 16\n'
        '                    assert base > requests + 1\n'
        '                    assert entry.buffers["x"].shape[0] == 6 * requests\n'
        '                    marker = int(entry.buffers["group_offsets"][base - 1].cpu())\n'
        '                    checks["metadata/uniform_six_proof_enabled"] = marker == 1\n'
        + marker)
    ast.parse(source)
    path.write_text(source)
    launcher = replace_once((here / "launch_sdk_measure.sh").read_text(),
                            'harness="$here/runs/sdk_measurement_harness_v1"',
                            'harness="$here/runs/uniform_meta_measurement_v1"')
    (here / "launch_uniform_meta_measure.sh").write_text(launcher)
    report = dict(
        parent="sdk_measurement_harness_v1", source=str(target),
        added_check="META uniform_six proof is actually INT32 1 after candidate eager, graph and post-L1",
        timing_loop_unchanged=True, full_chain_boundary_unchanged=True,
        scope="Only full-capacity equal-S6 priority performance cases; mixed functionality uses a separate validator",
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "uniform_meta_measurement_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
