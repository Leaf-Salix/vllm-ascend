# SPDX-License-Identifier: Apache-2.0
"""Freeze the existing ten-window harness with an observation after sampling."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/sdk_measurement_harness_v2"
    target = here / "runs/capacity_measurement_harness_v1"
    runner = (parent / "run_pto.py").read_text()
    marker = '            report["passed"] = True\n            save("complete")\n'
    runner = replace_once(
        runner,
        marker,
        "            from capacity_observation import observe_capacity\n\n"
        '            report["physical_plan_observation"] = observe_capacity(\n'
        '                entries["candidate"], native, case, args.output\n'
        "            )\n" + marker,
    )
    ast.parse(runner)
    assert runner.index('report["physical_plan_observation"]') > runner.index('report["checks"][side + "_post_l1"]')
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "run_pto.py").write_text(runner)
    for name in (
        "capacity_observation.py",
        "analyze_kv_access.py",
        "analyze_priority_phase.py",
        "measurement_provenance.py",
    ):
        shutil.copy2(here / name, target / name)
    launcher = replace_once(
        (here / "launch_sdk_measure_v2.sh").read_text(),
        'harness="$here/runs/sdk_measurement_harness_v2"',
        'harness="$here/runs/capacity_measurement_harness_v1"',
    )
    with (here / "launch_capacity_measure.sh").open("x") as stream:
        stream.write(launcher)
    report = dict(
        source=str(target),
        parent="sdk_measurement_harness_v2",
        sampling_unchanged=True,
        operator_source_unchanged=True,
        observation="One readback only after all L1 windows and existing post-L1 strict checks",
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "capacity_measurement_harness_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
