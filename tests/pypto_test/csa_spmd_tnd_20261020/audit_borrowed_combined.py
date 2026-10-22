# SPDX-License-Identifier: Apache-2.0
"""Prove the combined candidate keeps each previously validated executable."""

import hashlib
import json
from pathlib import Path

from audit_borrowed_score_stream import normalized_cpp


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    plan = json.loads((here / "borrowed_combined_compile_plan_v105.json").read_text())
    source = Path(plan["source"])
    assert all(digest(source / name) == value for name, value in plan["source_manifest"].items())
    jobs = {row["task_id"]: row for row in json.loads((here / "jobs.json").read_text())}
    proofs = []
    for row in plan["cases"]:
        directory = Path(row["compile_output"])
        assert json.loads((directory / "result.json").read_text())["passed"]
        new = directory / "kernel/ptoas/dynamic_csa_full.cpp"
        if row["topk_prefix"]:
            old = here / "runs/topk_prefix_guard_b8_cpu_v102/kernel/ptoas/dynamic_csa_full.cpp"
            origin = "Guarded TopK candidate, equivalent to measured v101 B8"
        elif row["short_half_ub"]:
            old = here / "runs/short_half_ub_short_b16_cpu_v95/kernel/ptoas/dynamic_csa_full.cpp"
            origin = "Short-half UB candidate, measured v95 short B16"
        else:
            measured = Path(jobs[row["task_id"]]["output"])
            files = [
                p for p in (measured / "build_output").glob("*/ptoas/dynamic_csa_full.cpp")
                if "PTOAS__WAIT_GE_BACKOFF<150>" in p.read_text()
            ]
            assert len(files) == 1
            old = files[0]
            origin = "Previously adopted output-balance TND; both new switches disabled"
        old_host = old.parents[1] / "orchestration/decode_csa_spmd.cpp"
        new_host = new.parents[1] / "orchestration/decode_csa_spmd.cpp"
        proof = dict(
            history=row["history"], batch=row["batch"], origin=origin,
            whole_cpp_equal=normalized_cpp(old) == normalized_cpp(new),
            host_byte_equal=old_host.read_bytes() == new_host.read_bytes(),
            old_cpp=str(old), new_cpp=str(new),
            old_cpp_sha256=digest(old), new_cpp_sha256=digest(new),
            orchestration=dict(old_path=str(old_host), new_path=str(new_host),
                               old_sha256=digest(old_host), new_sha256=digest(new_host)),
            topk_prefix=row["topk_prefix"], short_half_ub=row["short_half_ub"],
        )
        proofs.append(proof)
    report = dict(
        passed=all(p["whole_cpp_equal"] and p["host_byte_equal"] for p in proofs),
        normalization="Only full // pto: debug-name comment lines removed",
        proofs=proofs, source_manifest=plan["source_manifest"],
        device_execution=False, performance_measured=False,
        scope="Eight full CPU builds; composition preserves tested enabled and disabled paths",
    )
    (here / "borrowed_combined_codegen_audit_v105.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": [
        {k: p[k] for k in ("history", "batch", "whole_cpp_equal", "host_byte_equal")} for p in proofs
    ]}, indent=2))
    assert report["passed"], "Do not reuse old measurements or adopt a changed unvalidated executable"


if __name__ == "__main__":
    main()
