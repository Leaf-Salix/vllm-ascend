# SPDX-License-Identifier: Apache-2.0
"""Freeze a one-execution observation harness from the shared-SDK fixture."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/sdk_measurement_harness_v2"
    target = here / "runs/kv_access_harness_v1"
    source = (parent / "run_pto.py").read_text()
    begin = source.index('            variants = [("reference", baseline_name)]\n')
    end = source.index("            for side, package in variants:\n", begin)
    source = source[:begin] + '            variants = [("candidate", candidate_name)]\n' + source[end:]
    begin = source.index("                state, check = snapshot(side)\n")
    end = source.index("            if args.service_validation:\n", begin)
    source = (
        source[:begin]
        + """                # The kernel is the already-validated retained implementation.
                # This execution only observes metadata; never compare a result to itself.
                report["checks"][side] = {"mathematics_revalidated": False}
            from kv_access_probe import export_access

            report["kv_access"] = export_access(entries["candidate"], native, case, args.output)
            report["passed"] = True
            save("metadata_export_complete")
            return

"""
        + source[end:]
    )
    marker = "    def save(stage=None):\n"
    source = replace_once(
        source,
        marker,
        """    report.update(
        scope="Retained TND-SPMD selected-KV metadata observation only",
        performance_eligible=False,
        performance_measured=False,
        sampling={"eager_runs": 1, "valid_windows": 0},
        executed_sides=["candidate"],
    )

"""
        + marker,
    )
    ast.parse(source)
    # Everything after the observation return is unreachable; no timer/probe
    # compilation hook is installed, and no NPU operator source is changed.
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "run_pto.py").write_text(source)
    shutil.copy2(here / "kv_access_probe.py", target / "kv_access_probe.py")
    launcher = replace_once(
        (here / "launch_sdk_measure_v2.sh").read_text(),
        'harness="$here/runs/sdk_measurement_harness_v2"',
        'harness="$here/runs/kv_access_harness_v1"',
    )
    with (here / "launch_kv_access_probe.sh").open("x") as stream:
        stream.write(launcher)
    report = dict(
        source=str(target),
        parent="sdk_measurement_harness_v2",
        purpose="One retained-kernel execution, export TopK and physical plans; no performance or math revalidation",
        operator_source_unchanged=True,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "kv_access_harness_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
