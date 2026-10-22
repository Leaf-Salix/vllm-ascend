# SPDX-License-Identifier: Apache-2.0
"""Certify existing same-SDK C02 windows without pretending their source hashes match."""

import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from audit_score_group_guard import content, fingerprint
from measurement_provenance import package_hashes


def main():
    here = Path(__file__).resolve().parent
    workspace = next(p for p in here.parents if (p / "env-dsv4-0251rc1.sh").is_file())
    primary = workspace / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test"
    path = primary / "csa_performance_20261022_aicpu/runs/h262144_b16_s6_tnd_v1/result.json"
    data = json.loads(path.read_text())
    original = json.loads((here / "baseline.json").read_text())
    sdk = json.loads((here / "runs/sdk_measurement_harness_v1/expected_sdk.json").read_text())
    assert data["case"] == original["cases"]["C02"]["case"] and data["passed"]
    assert data["environment"]["shared_release"] == sdk["release"] and sdk["status"] == "validated"
    task = "task_20261022_200748_181884127036"
    log = subprocess.check_output(["task-submit", "--log", task], text=True)
    header = json.loads(
        next(line for line in log.splitlines() if line.startswith("CSA_SHARED_TOOLCHAIN ")).split(" ", 1)[1]
    )
    assert header["release"] == sdk["release"] and header["status"] == "validated"
    for name in ("pypto", "ptoas", "simpler"):
        assert Path(header[name]).resolve() == (workspace / sdk[name].get("build", sdk[name]["source"])).resolve()
    # C02 and C04 have the same dynamic compiled kernel class, independently
    # established before these measurements. Compare against the freshly measured
    # original implementation on the CURRENT SDK, not old-SDK timing data.
    current = here / "runs/h131072_b16_s6_baseline_sdk_v1/result.json"
    current_report = json.loads(current.read_text())
    assert current_report["passed"] and current_report["calibration"]
    assert current_report["source_hashes"]["candidate"] == original["files"]
    current_artifacts = list(current.parent.glob("build_output/decode_csa_spmd_*"))
    expected_code = content(current_artifacts[0])
    matches = [p for p in path.parent.glob("build_output/decode_csa_spmd_*") if content(p) == expected_code]
    assert len(matches) == 1
    windows = [w for w in data["windows"] if w["side"] == "spmd"]
    assert len(windows) >= 10 and all(w["records"] == w["cores"] == 72 and w["task_groups"] == 1 for w in windows)
    samples = []
    for window in windows:
        raw = json.loads(Path(window["raw"]).read_text())
        rows = raw["aicore_tasks"]
        assert raw["chip_swimlane_level"] == 1 and len(rows) == len({r[0] for r in rows}) == 72
        assert len({r[1] for r in rows}) == 1 and all(r[4] >= r[3] > 0 for r in rows)
        span = (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / raw["metadata"]["clock_freq_hz"]
        assert span == window["incore_span_us"]
        samples.append(span)
    assert samples == data["statistics_us"]["spmd"]["samples"]
    assert statistics.median(samples) == data["statistics_us"]["spmd"]["median"]
    for name in ("spmd_eager", "spmd_graph", "spmd_post_l1"):
        check = data["checks"][name]
        assert all(check["guards"].values()) and all(v["status"] == "PASS" for v in check["numerical"].values())
    report = dict(
        passed=True,
        case_id="C02",
        case=data["case"],
        reference_kind=original["reference_kind"],
        source_report=str(path),
        task_id=task,
        source_side="spmd",
        device_id=data["device_id"],
        original_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        actual_source_package=data["sources"]["spmd"],
        actual_source_hashes=package_hashes(Path(data["sources"]["spmd"])),
        source_files_are_identical=False,
        shared_toolchain=sdk,
        recorded_bootstrap=header,
        environment=data["environment"],
        same_sdk_baseline_artifact=str(current_artifacts[0]),
        same_sdk_baseline_report=str(current),
        matching_artifact=str(matches[0]),
        compiled_contract_sha256=fingerprint(expected_code),
        equivalence_rule="Full CPP, packed wrappers, orchestration and ABI; remove only pto locations",
        baseline_algorithm_unchanged_for_this_case=True,
        source_difference="Mixed-case group ordering and Host fixes; inactive in C02, compiled code identical",
        samples_us=samples,
        median_us=statistics.median(samples),
        windows=windows,
        calibration_remeasured=False,
    )
    with (here / "C02_reused_reference.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(dict(passed=True, device=report["device_id"], windows=len(samples), median_us=report["median_us"]))
    )


if __name__ == "__main__":
    main()
