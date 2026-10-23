# SPDX-License-Identifier: Apache-2.0
"""Bind the established phase diagnostic to the same SDK as priority measurements."""

import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    target = here / "runs/phase_sdk_harness_v1"
    shutil.copytree(here / "runs/phase_baseline_harness", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("measurement_provenance.py", "expected_sdk.json"):
        shutil.copy2(here / "runs/sdk_measurement_harness_v1" / name, target / name)
    source = (target / "run_pto.py").read_text()
    source = replace_once(
        source, "import types\n", "import types\n\nfrom measurement_provenance import capture_toolchain\n"
    )
    source = replace_once(
        source,
        '        "diagnostic_phase_clock": True,\n',
        '        "diagnostic_phase_clock": True,\n'
        '        "shared_toolchain": capture_toolchain(\n'
        '            json.loads(Path(__file__).with_name("expected_sdk.json").read_text())),\n',
    )
    (target / "run_pto.py").write_text(source)
    launcher = replace_once(
        (here / "launch_phase.sh").read_text(),
        'harness="$here/runs/phase_baseline_harness"',
        'harness="$here/runs/phase_sdk_harness_v1"',
    )
    (here / "launch_priority_phase.sh").write_text(launcher)
    report = dict(
        performance_eligible=False,
        purpose="Three diagnostic phase-clock replays of priority C05; no L1 performance windows",
        parent="phase_baseline_harness.json",
        shared_toolchain=json.loads((target / "expected_sdk.json").read_text()),
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
        instrumentation_unchanged=True,
    )
    with (here / "phase_sdk_harness_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
