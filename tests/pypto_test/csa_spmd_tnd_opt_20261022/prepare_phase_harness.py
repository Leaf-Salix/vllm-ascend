# SPDX-License-Identifier: Apache-2.0
"""Freeze a diagnostic-only paired harness; the formal L1 harness is untouched."""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from prepare_control_candidates import replace_once


def main():
    here = Path(__file__).resolve().parent
    original = here / "runs/frozen_v122_harness"
    target = here / "runs/phase_harness"
    shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    launcher = (target / "run_probe.sh").read_text()
    launcher = launcher.replace(str(original), str(target))
    launcher = replace_once(launcher, "for path in (\n", 'for path in (\n    harness / "phase_clock.py",\n')
    (target / "run_probe.sh").write_text(launcher)
    shutil.copyfile(here / "phase_clock.py", target / "phase_clock.py")
    reference = json.loads((target / "comparison_reference.json").read_text())
    reference["frozen_source"] = str(target / "reference")
    reference["diagnostic_phase_clock"] = True
    (target / "comparison_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    source = (target / "bsh_probe.py").read_text()
    source = replace_once(
        source,
        "            return source, dict(\n",
        "            from phase_clock import instrument\n"
        "            source, phase_audit = instrument(source)\n"
        "            return source, dict(\n"
        "                phase_clock=phase_audit,\n",
    )
    source = replace_once(source, "source_rewritten=bool(args.wait_first_fast_polls)", "source_rewritten=True")
    (target / "bsh_probe.py").write_text(source)
    source = (target / "run_probe.py").read_text()
    source = replace_once(
        source,
        "    args = parser.parse_args()\n",
        "    args = parser.parse_args()\n"
        "    if args.l1_windows or args.service_route or args.metadata_matrix:\n"
        '        raise ValueError("Phase probe is diagnostic only; no L1/service/matrix mixing")\n',
    )
    source = replace_once(
        source, "(signal[:, :, 1:] == 0).all()", "(signal[:, :, 1:16] == 0).all() and (signal[:, :, 20:] == 0).all()"
    )
    source = replace_once(
        source,
        '            report["stage"] = "complete"\n',
        "            from phase_clock import export_entries\n"
        '            report["phase_clock"] = export_entries(candidate, baseline, args.output)\n'
        '            report["stage"] = "complete"\n',
    )
    (target / "run_probe.py").write_text(source)
    for file in target.glob("*.py"):
        ast.parse(file.read_text())
    subprocess.run(["bash", "-n", str(target / "run_probe.sh")], check=True)
    report = {
        "harness": str(target),
        "eligible_for_performance": False,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*") if p.is_file()},
    }
    with (here / "phase_harness.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
