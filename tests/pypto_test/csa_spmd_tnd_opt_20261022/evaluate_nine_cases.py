# SPDX-License-Identifier: Apache-2.0
"""Evaluate nine equal-weight paired cases against immutable TND v122."""

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "csa_spmd_tnd_20261020"))
from evaluate_s6_cohort import shared_toolchain_fingerprint, source_hashes  # noqa: E402

MIXED_QUERIES = (1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 3, 4, 6)
EXPECTED_CASES = frozenset(
    [(131072, b, (6,) * b) for b in (1, 4, 8, 16, 24, 28)]
    + [(8192, b, (6,) * b) for b in (16, 32)]
    + [(131072, 16, MIXED_QUERIES)]
)
REFERENCE_KIND = "frozen_tnd_v122"


def case_measurement(result, reference_files):
    reference = result.get("comparison_reference", {})
    if result.get("phase_clock") or reference.get("diagnostic_phase_clock"):
        raise ValueError("Instrumented phase clocks cannot be performance-gate evidence")
    if reference.get("kind") != REFERENCE_KIND or reference.get("files") != reference_files:
        raise ValueError("Expected the exact frozen_tnd_v122 reference, not BSH or another TND")
    audit = result.get("bsh_wait_policy_audit", {})  # Historical harness field name, both sides now TND.
    sources = audit.get("actual_ccec_sources", [])
    if not (
        result.get("stage") == "complete"
        and result.get("passed")
        and result.get("functional_passed")
        and result.get("device_execution")
        and audit.get("passed")
        and audit.get("ticks") == 150
        and len(sources) >= 4
        and all(s.get("passed") and s.get("native_lowering") and not s.get("legacy_rewrite") for s in sources)
    ):
        raise ValueError("Missing complete correctness or native150 evidence for both TND sides")
    key = (result["history"], result["batch"], tuple(result["query_lengths"]))
    if key not in EXPECTED_CASES or not (
        result["history_lengths"] == [key[0]] * key[1]
        and result["actual_tokens"] == result["token_capacity"] == sum(key[2])
        and result["request_capacity"] == key[1]
        and not result.get("constant_indexer_history")
        and not result.get("wait_policy_experiment")
    ):
        raise ValueError("Input distribution/capacity differs from the fixed nine-case goal")
    timing = result["l1"]
    if (
        not timing.get("passed")
        or timing.get("shared_workspace")
        or not all(timing.get("protocol", {}).get(s) for s in ("spmd_tnd", "spmd_original"))
    ):
        raise ValueError("Invalid full-chain L1 protocol")
    windows = timing["windows"]
    if len(windows) < 10 or len(windows) % 2:
        raise ValueError("Need at least five complete paired windows")
    for index, window in enumerate(windows):
        order = ("spmd_tnd", "spmd_original") if (index // 2) % 2 == 0 else ("spmd_original", "spmd_tnd")
        if not (
            window["side"] == order[index % 2]
            and window["core_records"] == window["cores"] == 72
            and window["task_count"] == 1
            and window["clock_freq_hz"] == 50000000
            and math.isfinite(window["incore_span_us"])
            and window["incore_span_us"] > 0
        ):
            raise ValueError("L1 windows must alternate in pairs and cover one complete 72-core MIX task")
    samples = {
        side: [w["incore_span_us"] for w in windows if w["side"] == side] for side in ("spmd_original", "spmd_tnd")
    }
    baseline, candidate = (median(samples[s]) for s in ("spmd_original", "spmd_tnd"))
    return {
        "history": key[0],
        "batch": key[1],
        "query_lengths": list(key[2]),
        "actual_tokens": sum(key[2]),
        "reference_v122_us": baseline,
        "candidate_us": candidate,
        "ratio": candidate / baseline,
        "latency_change_pct": (candidate / baseline - 1) * 100,
        "reference_samples_us": samples["spmd_original"],
        "candidate_samples_us": samples["spmd_tnd"],
        "native_precision_exact": result.get("precision_exact"),
    }


def evaluate(jobs, selected):
    if len(selected) != 9 or len(set(selected)) != 9:
        raise ValueError("Select exactly nine distinct task IDs")
    here = Path(__file__).resolve().parent
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    records = {job["task_id"]: job for job in jobs}
    rows, fingerprints = [], []
    for task_id in selected:
        job = records[task_id]
        if job.get("last_observed_status") != "completed (exit=0)":
            raise ValueError(f"{task_id}: task not successfully completed")
        directory = Path(job["output"])
        result = json.loads((directory / "result.json").read_text())
        row = case_measurement(result, expected)
        snapshot = directory.with_name(directory.name + "_source")
        saved_reference = json.loads((snapshot / "comparison_reference.json").read_text())
        if saved_reference != result["comparison_reference"]:
            raise ValueError(f"{task_id}: result/snapshot reference mismatch")
        candidate = source_hashes(snapshot, "spmd/")
        baseline = source_hashes(snapshot, "original_spmd/")
        if not candidate or baseline != expected:
            raise ValueError(f"{task_id}: missing candidate or baseline source differs from v122")
        sdk = shared_toolchain_fingerprint(snapshot, result)
        if sdk is None:
            raise ValueError(f"{task_id}: shared validated SDK required")
        manifest = json.loads((snapshot / "source_manifest.json").read_text())
        fingerprints.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "shared_toolchain": sdk,
                "runtime_artifacts": result["runtime_artifacts"],
                "harness": {k: v for k, v in manifest.items() if not k.startswith(("spmd/", "original_spmd/"))},
            }
        )
        rows.append(dict(row, task_id=task_id, output=str(directory)))
    return summarize(rows, fingerprints)


def summarize(rows, fingerprints):
    if len(rows) != 9 or {(r["history"], r["batch"], tuple(r["query_lengths"])) for r in rows} != EXPECTED_CASES:
        raise ValueError("Must cover all eight S6 cases and the fixed 128K/B16 mixed case once")
    if len(fingerprints) != 9:
        raise ValueError("Each case requires a verified source/SDK fingerprint")
    if any(f != fingerprints[0] for f in fingerprints[1:]):
        raise ValueError("Candidate, baseline, harness or SDK differs across the cohort")
    geometric_mean = math.exp(sum(math.log(r["ratio"]) for r in rows) / 9)
    return {
        "reference_kind": REFERENCE_KIND,
        "scope": "Performance gate only; affected dynamic state and actual service acceptance remain required",
        "metric": "Nine equal-weight candidate/v122 full-chain L1 median latency ratios",
        "threshold": 0.90,
        "geometric_mean_ratio": geometric_mean,
        "latency_reduction_pct": (1 - geometric_mean) * 100,
        "passed": geometric_mean <= 0.90,
        "cases": rows,
        "fingerprint": fingerprints[0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs=9)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.jobs.read_text()), args.task_ids)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "geometric_mean_ratio", "reference_kind")}))
    raise SystemExit(0 if report["passed"] else 1)
