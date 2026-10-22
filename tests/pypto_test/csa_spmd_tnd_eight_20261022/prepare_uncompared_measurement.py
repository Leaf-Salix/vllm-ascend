# SPDX-License-Identifier: Apache-2.0
"""Measure a new candidate while qualifying an external reference in a separate certificate."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/sdk_measurement_harness_v2"
    previous = here / "runs/sdk_measurement_harness_v1"
    shutil.copytree(previous, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = (target / "run_pto.py").read_text()
    source = replace_once(
        source,
        '    comparison.add_argument("--calibrate-baseline", action="store_true")\n',
        '    comparison.add_argument("--calibrate-baseline", action="store_true")\n'
        '    comparison.add_argument("--uncompared-candidate", action="store_true")\n',
    )
    source = replace_once(
        source,
        '        "calibration": args.calibrate_baseline,\n',
        '        "calibration": args.calibrate_baseline,\n        "uncompared_candidate": args.uncompared_candidate,\n',
    )
    before, after = ast.parse((previous / "run_pto.py").read_text()), ast.parse(source)
    for span in ("range(3)", "range(10)"):
        loops = [
            next(n for n in ast.walk(t) if isinstance(n, ast.For) and ast.unparse(n.iter) == span)
            for t in (before, after)
        ]
        assert ast.dump(loops[0]) == ast.dump(loops[1])
    (target / "run_pto.py").write_text(source)
    launcher = replace_once(
        (here / "launch_sdk_measure.sh").read_text(),
        'if [[ "$mode" == candidate ]]; then\n',
        'if [[ "$mode" == uncompared ]]; then\n'
        "    comparison=(--uncompared-candidate)\n"
        'elif [[ "$mode" == candidate ]]; then\n',
    )
    launcher = replace_once(
        launcher, 'harness="$here/runs/sdk_measurement_harness_v1"', 'harness="$here/runs/sdk_measurement_harness_v2"'
    )
    (here / "launch_sdk_measure_v2.sh").write_text(launcher)
    report = dict(
        parent="sdk_measurement_harness_v1.json",
        target=str(target),
        reason="External same-SDK reference has exact equivalent compiled code but different source-module hashes",
        mode="New candidate gets ten complete L1 windows; no ratio or fictitious baseline hashes in the raw report",
        validation="Original frozen source remains the correctness reference; SDK/source checks are unchanged",
        shared_toolchain=json.loads((target / "expected_sdk.json").read_text()),
        measurement_loops_unchanged=True,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "sdk_measurement_harness_v2.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
