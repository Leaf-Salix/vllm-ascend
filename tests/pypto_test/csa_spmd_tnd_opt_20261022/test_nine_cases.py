# SPDX-License-Identifier: Apache-2.0
"""Protect the new denominator, fixed mixed distribution and L1 measurement contract."""

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from evaluate_nine_cases import EXPECTED_CASES, MIXED_QUERIES, case_measurement, evaluate, summarize
from materialize_reference import materialize


def result_for(history, batch, queries):
    return {
        "comparison_reference": {"kind": "frozen_tnd_v122", "files": {"entry.py": "test-digest"}},
        "stage": "complete",
        "passed": True,
        "functional_passed": True,
        "device_execution": True,
        "history": history,
        "batch": batch,
        "query_lengths": list(queries),
        "history_lengths": [history] * batch,
        "actual_tokens": sum(queries),
        "token_capacity": sum(queries),
        "request_capacity": batch,
        "bsh_wait_policy_audit": {
            "passed": True,
            "ticks": 150,
            "actual_ccec_sources": [
                {"passed": True, "native_lowering": True, "legacy_rewrite": False} for _ in range(4)
            ],
        },
        "l1": {
            "passed": True,
            "shared_workspace": False,
            "protocol": {"spmd_original": True, "spmd_tnd": True},
            "windows": [
                {
                    "side": side,
                    "core_records": 72,
                    "cores": 72,
                    "task_count": 1,
                    "clock_freq_hz": 50000000,
                    "incore_span_us": 89.0 if side == "spmd_tnd" else 100.0,
                }
                for i in range(5)
                for side in (("spmd_tnd", "spmd_original") if i % 2 == 0 else ("spmd_original", "spmd_tnd"))
            ],
        },
    }


@pytest.mark.parametrize("case", sorted(EXPECTED_CASES))
def test_each_fixed_case_is_accepted(case):
    result = result_for(*case)
    row = case_measurement(result, {"entry.py": "test-digest"})
    assert row["ratio"] == 0.89
    assert row["reference_samples_us"] == [100.0] * 5


@pytest.mark.parametrize(
    "fault",
    [
        "bsh",
        "source",
        "history",
        "mixed_order",
        "padding",
        "missing_window",
        "partial_core",
        "nonalternate",
        "clock",
        "nan",
        "precision",
        "legacy",
        "generation",
        "shared_workspace",
    ],
)
def test_invalid_comparisons_are_rejected(fault):
    result = result_for(131072, 16, MIXED_QUERIES)
    window = result["l1"]["windows"][0]
    if fault == "bsh":
        result["comparison_reference"]["kind"] = "bsh"
    elif fault == "source":
        result["comparison_reference"]["files"]["entry.py"] = "different"
    elif fault == "history":
        result["history_lengths"][0] -= 1
    elif fault == "mixed_order":
        result["query_lengths"] = list(reversed(MIXED_QUERIES))
    elif fault == "padding":
        result["token_capacity"] = 96
    elif fault == "missing_window":
        result["l1"]["windows"].pop()
    elif fault == "partial_core":
        window["cores"] = 71
    elif fault == "nonalternate":
        window["side"] = "spmd_original"
    elif fault == "clock":
        window["clock_freq_hz"] = 100000000
    elif fault == "nan":
        window["incore_span_us"] = math.nan
    elif fault == "precision":
        result["functional_passed"] = False
    elif fault == "legacy":
        result["bsh_wait_policy_audit"]["actual_ccec_sources"][0]["legacy_rewrite"] = True
    elif fault == "generation":
        result["l1"]["protocol"]["spmd_tnd"] = False
    elif fault == "shared_workspace":
        result["l1"]["shared_workspace"] = True
    with pytest.raises(ValueError):
        case_measurement(result, {"entry.py": "test-digest"})


def test_eight_case_or_duplicate_task_cannot_pass():
    with pytest.raises(ValueError, match="nine distinct"):
        evaluate([], [f"task_{i}" for i in range(8)])
    with pytest.raises(ValueError, match="nine distinct"):
        evaluate([], ["one_task"] * 9)


def test_mixed_case_has_equal_weight_and_cannot_be_dropped():
    rows = [case_measurement(result_for(*case), {"entry.py": "test-digest"}) for case in sorted(EXPECTED_CASES)]
    mixed = next(row for row in rows if tuple(row["query_lengths"]) == MIXED_QUERIES)
    mixed["ratio"] = 1.2
    report = summarize(rows, [{"same": "fingerprint"}] * 9)
    assert report["geometric_mean_ratio"] == pytest.approx((0.89**8 * 1.2) ** (1 / 9))
    assert report["passed"] is False
    with pytest.raises(ValueError, match="all eight S6"):
        summarize([r for r in rows if r is not mixed], [{}] * 8)
    with pytest.raises(ValueError, match="differs across"):
        summarize(rows, [{}] * 8 + [{"candidate": "another candidate"}])


def test_reference_can_be_rebuilt_without_runs_directory(tmp_path):
    output = tmp_path / "restored"
    manifest = materialize(output)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.py")} == manifest["files"]
    with pytest.raises(FileExistsError):
        materialize(output)


def test_actual_new_launcher_freezes_v122_and_preserves_timing(tmp_path):
    here = Path(__file__).resolve().parent
    harness = here / "runs/frozen_v122_harness"
    if not harness.exists():
        pytest.skip("Generate frozen harness first; this skip is not device evidence")
    launcher = (harness / "run_probe.sh").read_text()
    marker = '"$baseline_source" "${manifest:-}" <<\'PY\'\n'
    script = launcher.split(marker, 1)[1].split("\nPY\n", 1)[0]
    snapshot = tmp_path / "snapshot"
    reference = harness / "reference"
    subprocess.run(
        [
            sys.executable,
            "-",
            str(here.parents[2]),
            str(snapshot),
            str(harness / "run_probe.sh"),
            "probe",
            str(reference),
            str(reference),
            "unused_bsh_manifest",
        ],
        input=script,
        text=True,
        check=True,
    )
    saved = json.loads((snapshot / "comparison_reference.json").read_text())
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert saved["kind"] == "frozen_tnd_v122" and saved["files"] == expected
    assert not (snapshot / "bsh_baseline_manifest.json").exists()
    for name, digest in expected.items():
        assert hashlib.sha256((snapshot / "original_spmd" / name).read_bytes()).hexdigest() == digest
    old = here.parent / "csa_spmd_tnd_20261020"
    for name in ("tnd_performance.py", "metadata_matrix.py", "service_probe.py", "service_metadata_replay.py"):
        assert (snapshot / name).read_bytes() == (old / name).read_bytes()
    mutated = copy.deepcopy(saved)
    mutated["kind"] = "validated_qb_wide_v108b"
    result = result_for(131072, 16, MIXED_QUERIES)
    result["comparison_reference"] = mutated
    with pytest.raises(ValueError, match="exact frozen_tnd_v122"):
        case_measurement(result, expected)
