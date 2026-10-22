# SPDX-License-Identifier: Apache-2.0
"""Archive the same-address B16 Query diagnostic, explicitly outside acceptance."""

import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import median

from collect_trials import shared_toolchain_fingerprint, source_hashes
from evaluate_nine_cases import REFERENCE_KIND, case_measurement


def main():
    here = Path(__file__).resolve().parent
    task = "task_20261022_070839_165489119202"
    jobs_path = here.parent / "csa_spmd_tnd_20261020/jobs.json"
    jobs = json.loads(jobs_path.read_text())
    job = next(j for j in jobs if j["task_id"] == task)
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    assert status == "completed (exit=0)"
    output = Path(job["output"])
    snapshot = output.with_name(output.name + "_source")
    path = output / "result.json"
    result = json.loads(path.read_text())
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    candidate = json.loads((here / "query_contiguous_heads_v4.json").read_text())["files"]
    assert result["stage"] == "complete" and result["passed"] and result["functional_passed"]
    assert result["device_execution"] and result["comparison_reference"]["kind"] == REFERENCE_KIND
    assert result["comparison_reference"]["files"] == expected
    assert source_hashes(snapshot, "original_spmd/") == expected
    assert source_hashes(snapshot, "spmd/") == candidate
    sdk = shared_toolchain_fingerprint(snapshot, result)
    assert sdk is not None
    assert result["history"] == 131072 and result["history_lengths"] == [131072] * 16
    assert result["query_lengths"] == [6] * 16
    assert result["actual_tokens"] == result["token_capacity"] == 96
    assert result["batch"] == result["request_capacity"] == 16
    assert not result.get("phase_clock") and not result.get("wait_policy_experiment")
    audit = result["bsh_wait_policy_audit"]
    assert audit["passed"] and audit["ticks"] == 150
    assert len(audit["actual_ccec_sources"]) >= 4
    assert all(s["passed"] and s["native_lowering"] and not s["legacy_rewrite"] for s in audit["actual_ccec_sources"])
    timing = result["l1"]
    assert timing["passed"] and timing["shared_workspace"] is True
    assert all(timing["protocol"].get(s) for s in ("spmd_tnd", "spmd_original"))
    windows = timing["windows"]
    assert len(windows) >= 10 and len(windows) % 2 == 0
    for i, window in enumerate(windows):
        order = ("spmd_tnd", "spmd_original") if (i // 2) % 2 == 0 else ("spmd_original", "spmd_tnd")
        assert window["side"] == order[i % 2] and window["task_count"] == 1
        assert window["core_records"] == window["cores"] == 72 and window["clock_freq_hz"] == 50000000
        assert math.isfinite(window["incore_span_us"]) and window["incore_span_us"] > 0
    try:
        case_measurement(result, expected)
    except ValueError as error:
        assert str(error) == "Invalid full-chain L1 protocol"
    else:
        raise AssertionError("The formal gate must reject shared workspaces")
    samples = {
        side: [w["incore_span_us"] for w in windows if w["side"] == side] for side in ("spmd_original", "spmd_tnd")
    }
    baseline, measured = (median(samples[s]) for s in ("spmd_original", "spmd_tnd"))
    report = dict(
        task_id=task,
        passed=True,
        formal_gate_eligible=False,
        formal_nine_case_gate_passed=False,
        scope="Same-address full-L1 attribution only; not independent-workspace acceptance",
        output=str(output),
        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        shared_toolchain=sdk,
        reference_v122_us=baseline,
        candidate_us=measured,
        latency_change_pct=(measured / baseline - 1) * 100,
        reference_samples_us=samples["spmd_original"],
        candidate_samples_us=samples["spmd_tnd"],
        paired_faster=sum(a < b for a, b in zip(samples["spmd_tnd"], samples["spmd_original"])),
        native_precision_exact=result["precision_exact"],
        protocol=timing["protocol"],
        final_gate_rejection_verified=True,
        shared_address_mechanism=(
            "Existing harness assigns baseline.buffers=candidate.buffers; "
            "binding reads that dictionary on each call"
        ),
    )
    with (here / "query_contiguous_heads_v4_shared_diagnostic_results.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    job["last_observed_status"] = status
    jobs_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "formal_gate_eligible", "latency_change_pct", "paired_faster")}))


if __name__ == "__main__":
    main()
