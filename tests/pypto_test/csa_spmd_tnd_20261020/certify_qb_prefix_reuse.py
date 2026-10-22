# SPDX-License-Identifier: Apache-2.0
"""Certify the six unchanged full-chain paths after Q_B/prefix specialization."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_borrowed_score_stream import normalized_cpp
from certify_borrowed_reuse import check_optional_guard
from certify_output_balance_reuse import checked_manifest, verify_post_l1_harness
from evaluate_s6_cohort import source_hashes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    canonical = args.canonical_snapshot.resolve()
    current = checked_manifest(canonical)
    native = canonical / "native_toolchain_manifest.json"
    assert all(digest(Path(p)) == value for p, value in json.loads(native.read_text())["files"].items())
    plan = json.loads((here / "qb_prefix_compile_plan_v114.json").read_text())
    for name, value in plan["source_manifest"].items():
        assert digest(canonical / "spmd" / name) == value
        assert digest(Path(plan["source"]) / name) == value
    jobs = {j["task_id"]: j for j in json.loads((here / "jobs.json").read_text())}
    harness = {k: v for k, v in current.items() if not k.startswith(("spmd/", "original_spmd/"))}
    reused = {}
    for case in plan["cases"]:
        if case["active"]:
            continue
        job = jobs[case["task_id"]]
        assert job["last_observed_status"] == "completed (exit=0)"
        directory = Path(job["output"])
        result = json.loads((directory / "result.json").read_text())
        assert result["passed"] and not result["policy"]["HC_CONSUMER_REDUCE"]
        old = directory.with_name(directory.name + "_source")
        old_manifest = checked_manifest(old)
        assert source_hashes(old, "original_spmd/") == source_hashes(canonical, "original_spmd/")
        assert digest(old / "native_toolchain_manifest.json") == digest(native)
        old_harness = {k: v for k, v in old_manifest.items() if not k.startswith(("spmd/", "original_spmd/"))}
        assert old_harness.keys() == harness.keys()
        changes = {k for k in old_harness if old_harness[k] != harness[k]}
        assert changes <= {"run_probe.py", "metadata_matrix.py"}
        if "run_probe.py" in changes:
            check_optional_guard(old / "run_probe.py", canonical / "run_probe.py")
        if "metadata_matrix.py" in changes:
            verify_post_l1_harness(
                (old / "metadata_matrix.py").read_text(), (canonical / "metadata_matrix.py").read_text()
            )
        compiled = Path(case["compile_output"])
        assert case["fresh_compile"] and json.loads((compiled / "result.json").read_text())["passed"]
        existing = list((directory / "build_output").glob("*/ptoas/dynamic_csa_full.cpp"))
        proofs = []
        for side in ("candidate", "baseline"):
            files = [p for p in existing if ("PTOAS__WAIT_GE_BACKOFF<150>" in p.read_text()) == (side == "candidate")]
            assert len(files) == 1
            previous = files[0]
            # The frozen BSH source/toolchain is byte-identical. Reuse its
            # existing executable; only the changed candidate needs rebuilding.
            new = compiled / "kernel/ptoas/dynamic_csa_full.cpp" if side == "candidate" else previous
            assert normalized_cpp(previous) == normalized_cpp(new), (case["history"], case["batch"], side)
            old_host = previous.parents[1] / "orchestration/decode_csa_spmd.cpp"
            new_host = new.parents[1] / "orchestration/decode_csa_spmd.cpp"
            assert old_host.read_bytes() == new_host.read_bytes()
            proofs.append(dict(side=side, old_cpp=str(previous), new_cpp=str(new),
                               old_cpp_sha256=digest(previous), new_cpp_sha256=digest(new),
                               orchestration=dict(old_path=str(old_host), new_path=str(new_host),
                                                  old_sha256=digest(old_host), new_sha256=digest(new_host))))
        reused[case["task_id"]] = dict(
            history=case["history"], batch=case["batch"],
            source_manifest_sha256=digest(old / "source_manifest.json"), proofs=proofs,
            harness_changes=sorted(changes), harness_changes_outside_timing_and_disabled=True,
        )
    assert len(reused) == 6
    report = dict(passed=True, reused=reused, canonical_snapshot=str(canonical),
                  native_toolchain_sha256=digest(native),
                  normalization="Only full // pto: debug-name comment lines removed",
                  canonical_fields=dict(candidate=source_hashes(canonical, "spmd/"),
                                        baseline=source_hashes(canonical, "original_spmd/"), harness=harness),
                  scope="Six fresh unchanged-candidate C++/Host proofs; preserve every original paired L1 window")
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "reused_cases": len(reused)}))


if __name__ == "__main__":
    main()
