# SPDX-License-Identifier: Apache-2.0
"""Reuse four existing original-algorithm, same-SDK measurements without device work."""

import json
import statistics
from pathlib import Path

from audit_score_group_guard import content, fingerprint


def main():
    here = Path(__file__).resolve().parent
    workspace = next(p for p in here.parents if (p / "env-dsv4-0251rc1.sh").is_file())
    archive = workspace / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_performance_20261022_aicpu"
    sdk = json.loads((here / "runs/sdk_measurement_harness_v1/expected_sdk.json").read_text())
    baseline = json.loads((here / "baseline.json").read_text())
    classes = json.loads((here / "leaf_scale_guard_disabled_classes.json").read_text())
    jobs = json.loads((archive / "jobs.json").read_text())
    fixture = json.loads((here / "C02_reused_fixture_audit.json").read_text())
    assert fixture["fixture_source_equal"] and all(fixture["input_construction_call_ast_equal"].values())
    assert Path(fixture["source_harness"]).resolve() == (archive / "sources").resolve()
    report = dict(
        device_remeasured=False,
        baseline_kind=baseline["reference_kind"],
        shared_toolchain=sdk,
        cases={},
        fixture_audit="C02_reused_fixture_audit.json (same archive and common fixture construction)",
        exclusions={
            "C02": "Already qualified separately; do not repeat",
            "C04/C05": "Keep current explicitly calibrated priority references; do not choose favorable alternatives",
            "C08": "Archive has no SPMD L1 windows and uses a different group-order implementation",
        },
    )
    for case_id in ("C01", "C03", "C06", "C07"):
        case = baseline["cases"][case_id]["case"]
        root = archive / "runs" / f"{case['name']}_tnd_v1"
        data = json.loads((root / "result.json").read_text())
        assert data["passed"] and data["case"] == case
        assert data["environment"]["shared_release"] == sdk["release"]
        header_line = next(
            s for s in (root / "task.log").read_text().splitlines() if s.startswith("CSA_SHARED_TOOLCHAIN ")
        )
        header = json.loads(header_line.split(" ", 1)[1])
        assert header["release"] == sdk["release"] and header["status"] == sdk["status"] == "validated"
        for name in ("pypto", "ptoas", "simpler"):
            assert Path(header[name]).resolve() == (workspace / sdk[name].get("build", sdk[name]["source"])).resolve()
        expected = content(Path(classes[case_id]["artifact"]))
        assert fingerprint(expected) == classes[case_id]["code_hash"]
        matches = [p for p in (root / "build_output").glob("decode_csa_spmd_*") if content(p) == expected]
        assert len(matches) == 1
        windows = [w for w in data["windows"] if w["side"] == "spmd"]
        assert len(windows) >= 10
        samples = []
        for w in windows:
            assert w["records"] == w["cores"] == 72 and w["task_groups"] == 1
            raw = json.loads(Path(w["raw"]).read_text())
            rows = raw["aicore_tasks"]
            assert raw["chip_swimlane_level"] == 1 and len(rows) == len({r[0] for r in rows}) == 72
            assert len({r[1] for r in rows}) == 1 and all(r[4] >= r[3] > 0 for r in rows)
            span = (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / raw["metadata"]["clock_freq_hz"]
            assert span == w["incore_span_us"]
            samples.append(span)
        assert samples == data["statistics_us"]["spmd"]["samples"]
        assert statistics.median(samples) == data["statistics_us"]["spmd"]["median"]
        for name in ("spmd_eager", "spmd_graph", "spmd_post_l1"):
            checks = data["checks"][name]
            assert all(checks["guards"].values()) and all(v["status"] == "PASS" for v in checks["numerical"].values())
        (job,) = [j for j in jobs if Path(j["output"]).resolve() == root.resolve()]
        report["cases"][case_id] = dict(
            passed=True,
            task_id=job["task_id"],
            case=case,
            source_report=str(root / "result.json"),
            actual_compiled=str(matches[0]),
            original_equivalence=classes[case_id],
            recorded_bootstrap=header,
            device_id=data["device_id"],
            all_samples_us=samples,
            median_us=statistics.median(samples),
            raw_windows=[w["raw"] for w in windows],
            precision_basis="Original archive eager/graph/post-L1 checks pass; original executable/ABI equivalent",
        )
    with (here / "same_sdk_reused_other_references.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print({k: v["median_us"] for k, v in report["cases"].items()})


if __name__ == "__main__":
    main()
