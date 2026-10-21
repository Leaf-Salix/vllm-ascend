# SPDX-License-Identifier: Apache-2.0
"""Collect terminal trial evidence, preserving all paired windows and known Native differences."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from evaluate_nine_cases import REFERENCE_KIND, case_measurement

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "csa_spmd_tnd_20261020"))
from evaluate_s6_cohort import shared_toolchain_fingerprint, source_hashes  # noqa: E402


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.task_ids) != len(set(args.task_ids)):
        raise ValueError("Duplicate task ID")
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    jobs_path = here.parent / "csa_spmd_tnd_20261020/jobs.json"
    jobs = json.loads(jobs_path.read_text())
    by_id = {job["task_id"]: job for job in jobs}
    rows = []
    for task_id in args.task_ids:
        job = by_id[task_id]
        status = subprocess.check_output(["task-submit", "--status", task_id], text=True).strip()
        job["last_observed_status"] = status
        if status != "completed (exit=0)":
            raise ValueError(f"{task_id}: {status}; retain the original handle")
        directory = Path(job["output"])
        snapshot = directory.with_name(directory.name + "_source")
        path = directory / "result.json"
        result = json.loads(path.read_text())
        reference = result.get("comparison_reference", {})
        if not (
            result.get("passed")
            and result.get("functional_passed")
            and result.get("stage") == "complete"
            and reference.get("kind") == REFERENCE_KIND
            and reference.get("files") == expected
        ):
            raise ValueError(f"{task_id}: incomplete frozen-v122 correctness evidence")
        if source_hashes(snapshot, "original_spmd/") != expected:
            raise ValueError("Reference bytes differ from v122")
        fingerprint = source_hashes(snapshot, "spmd/")
        sdk = shared_toolchain_fingerprint(snapshot, result)
        if sdk is None or not result.get("bsh_wait_policy_audit", {}).get("passed"):
            raise ValueError("Missing SDK/wait audit")
        row = dict(
            task_id=task_id,
            output=str(directory),
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            passed=True,
            functional_passed=True,
            candidate_source_hashes=fingerprint,
            shared_toolchain=sdk,
            performance_measured=bool(result.get("l1")),
        )
        if result.get("l1"):
            row.update(case_measurement(result, expected))
            row["paired_faster"] = sum(a < b for a, b in zip(row["candidate_samples_us"], row["reference_samples_us"]))
        else:
            row.update(
                history=result["history"],
                batch=result["batch"],
                query_lengths=result["query_lengths"],
                token_capacity=result["token_capacity"],
                request_capacity=result["request_capacity"],
                constant_indexer_history=bool(result.get("constant_indexer_history")),
            )
            matrix = directory / "metadata_matrix.json"
            if matrix.exists():
                data = json.loads(matrix.read_text())
                if not data["passed"]:
                    raise ValueError("Metadata matrix failed")
                row["metadata_matrix"] = {
                    "path": str(matrix),
                    "sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
                    "passed": True,
                    "metadata_cases": [{"name": v["name"], "passed": v["passed"]} for v in data["metadata_cases"]],
                    "continuous_steps": len(data["continuous_steps"]),
                    "same_graph_replays": data.get("same_graph_replays"),
                }
        row["native_existing_difference"] = {
            name: {key: value.get(key) for key in ("status", "mismatches", "max_abs", "atol", "rtol")}
            for name, value in (
                ("candidate_output", result.get("native_frozen_output", {})),
                (
                    "reference_output",
                    result.get("original_spmd_attribution", {}).get("baseline_native_frozen_output", {}),
                ),
            )
        }
        rows.append(row)
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")
    report = dict(
        reference_kind=REFERENCE_KIND,
        formal_nine_case_gate_passed=False,
        scope="Selected trial evidence; independent gains cannot be added; adoption remains a separate decision",
        cases=rows,
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(
            json.dumps(
                {
                    key: row.get(key)
                    for key in ("task_id", "passed", "performance_measured", "latency_change_pct", "paired_faster")
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
