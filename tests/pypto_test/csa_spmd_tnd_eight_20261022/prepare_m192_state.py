# SPDX-License-Identifier: Apache-2.0
"""Allow B28 in the unchanged 16-metadata plus four-step state validator."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/uniform_state_harness_v1"
    target = here / "runs/m192_state_harness_v1"
    source = (parent / "run_pto.py").read_text()
    source = replace_once(
        source,
        'args.case not in ("h131072_b16_s6", "h131072_b24_s6")',
        'args.case not in ("h131072_b16_s6", "h131072_b24_s6", "h131072_b28_s6")',
    )
    source = replace_once(source, "priority C04/C05 SPMD pair", "C04/C05/C06 SPMD cases")
    ast.parse(source)
    # Execute only the original schedule-building AST, with no torch/NPU work.
    tree = ast.parse((parent / "state_replay.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate")
    begin = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "histories" for t in n.targets)
    )
    end = next(
        i for i, n in enumerate(fn.body[begin:], begin) if isinstance(n, ast.For) and ast.unparse(n.iter) == "schedules"
    )
    ns = dict(history=131072, requests=28)
    exec(compile(ast.Module(body=fn.body[begin:end], type_ignores=[]), "unchanged_state_schedules", "exec"), ns)
    schedules = ns["schedules"]
    assert len(schedules) == 16
    for name, queries, histories, owners, _ in schedules:
        assert len(queries) == len(histories) <= 28 and sum(queries) <= 168
        assert all(1 <= q <= 6 for q in queries) and all(0 <= h < 131072 + 128 for h in histories)
        if owners is not None:
            assert len(set(owners)) == len(queries) and all(0 <= owner < 28 for owner in owners)
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "run_pto.py").write_text(source)
    assert (target / "state_replay.py").read_bytes() == (parent / "state_replay.py").read_bytes()
    launcher = replace_once(
        (here / "launch_uniform_state.sh").read_text(),
        'harness="$here/runs/uniform_state_harness_v1"',
        'harness="$here/runs/m192_state_harness_v1"',
    )
    with (here / "launch_m192_state.sh").open("x") as out:
        out.write(launcher)
    report = dict(
        source=str(target),
        parent="uniform_state_harness_v1.json",
        performance_eligible=False,
        changed="Add C06 to the pre-initialization allowlist; all state math/checks unchanged",
        cpu_schedule_checks=16,
        continuous_steps_unchanged=4,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    )
    with (here / "m192_state_harness_v1.json").open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    service_parent = here / "runs/priority_service_harness_v1"
    service_target = here / "runs/m192_service_harness_v1"
    service = replace_once(
        (service_parent / "run_pto.py").read_text(),
        'args.case not in ("h131072_b16_s6", "h131072_b24_s6")',
        'args.case not in ("h131072_b16_s6", "h131072_b24_s6", "h131072_b28_s6")',
    )
    service = replace_once(service, "Service-only C04/C05 validation", "Service-only C04/C05/C06 validation")
    shutil.copytree(service_parent, service_target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (service_target / "run_pto.py").write_text(service)
    service_launcher = replace_once(
        (here / "launch_priority_service.sh").read_text(),
        'harness="$here/runs/priority_service_harness_v1"',
        'harness="$here/runs/m192_service_harness_v1"',
    )
    with (here / "launch_m192_service.sh").open("x") as out:
        out.write(service_launcher)
    service_report = dict(
        source=str(service_target),
        parent="priority_service_harness_v1",
        changed="Only add C06 to service allowlist; canonical service/graph/math checks unchanged",
        performance_eligible=False,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in service_target.glob("*.py")},
    )
    with (here / "m192_service_harness_v1.json").open("x") as out:
        out.write(json.dumps(service_report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
