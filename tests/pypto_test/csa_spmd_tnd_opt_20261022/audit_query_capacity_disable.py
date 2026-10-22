# SPDX-License-Identifier: Apache-2.0
"""Bind the narrowed T80 disabled program to its measured v122 reference."""

import hashlib
import json
from pathlib import Path

from audit_query_head_guard import normalized


def main():
    here = Path(__file__).resolve().parent
    job = here / "runs/query_head_guard_t80_g6"
    snapshot = job.with_name(job.name + "_source") / "original_spmd"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob("*.py")} == expected
    measured = json.loads((job / "result.json").read_text())
    assert measured["stage"] == "complete" and measured["passed"]
    assert measured["comparison_reference"]["files"] == expected
    paths = [
        p for p in (job / "build_output").glob("*/ptoas/dynamic_csa_full.pto") if str(snapshot) + "/" in p.read_text()
    ]
    assert len(paths) == 1
    old = paths[0].parent.parent
    new_root = here / "runs/compile_query_head_capacity_guard_t80"
    assert json.loads((new_root / "result.json").read_text())["full_codegen_passed"]
    new = new_root / "kernel"
    artifacts = []
    for relative in (
        "ptoas/dynamic_csa_full.cpp",
        "kernels/aic/cube_schedule.cpp",
        "kernels/aiv/vector_schedule.cpp",
        "orchestration/decode_csa_spmd.cpp",
        "compiled_meta.json",
    ):
        a, b = old / relative, new / relative
        artifacts.append(
            dict(
                relative=relative,
                identical=normalized(a) == normalized(b),
                old=str(a),
                new=str(b),
                old_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),
                new_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),
            )
        )
    report = dict(
        passed=all(a["identical"] for a in artifacts),
        artifacts=artifacts,
        scope="Disabled T80 G6 generated program, packed ABI and host launch; not ELF identity or final cohort",
        device_execution=False,
        formal_nine_case_gate_passed=False,
    )
    with (here / "query_head_capacity_guard_disabled_audit.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"]}))
    assert report["passed"]


if __name__ == "__main__":
    main()
