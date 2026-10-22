# SPDX-License-Identifier: Apache-2.0
"""Bind the two query-local measurements to raw L1 and the CPU-gated code."""

import argparse
import json
import math
import statistics
from pathlib import Path

from analyze_priority_phase import executable
from measurement_provenance import require_same_sdk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", action="store_true", help="Compare the diagnostic milestones, not L1 gains")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    if args.phase:
        compare_phases(here)
        return
    source = json.loads((here / "query_local_v5.json").read_text())
    retained = json.loads((here / "retained_merged_schedule.json").read_text())
    results = {}
    for case, name, task in (
        ("C04", "h131072_b16_s6", "task_20261023_055838_147095425686"),
        ("C05", "h131072_b24_s6", "task_20261023_055621_141049523995"),
    ):
        run = here / "runs" / f"{name}_query_local_v5"
        data = json.loads((run / "result.json").read_text())
        baseline = json.loads((here / "runs" / f"{name}_baseline_sdk_v1/result.json").read_text())
        require_same_sdk(data["shared_toolchain"], baseline["shared_toolchain"])
        assert data["passed"] and data["device_execution"] and data["stage"] == "complete"
        assert data["case"] == baseline["case"] and data["source_hashes"]["candidate"] == source["files"]
        for check in data["checks"].values():
            assert all(check["guards"].values())
            assert all(row["bit_equal"] for row in check["numerical"].values())
        samples = []
        for window in data["windows"]:
            raw = json.loads(Path(window["raw"]).read_text())
            rows, frequency = raw["aicore_tasks"], raw["metadata"]["clock_freq_hz"]
            assert raw["chip_swimlane_level"] == 1 and frequency > 0
            assert window["side"] == "candidate" and len(rows) == len({r[0] for r in rows}) == 72
            assert len({r[1] for r in rows}) == 1
            assert all(len(r) == 6 and r[4] >= r[3] > 0 for r in rows)
            value = (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / frequency
            assert math.isclose(value, window["incore_span_us"], rel_tol=0, abs_tol=1e-8)
            samples.append(value)
        assert len(samples) >= 10 and samples == data["statistics_us"]["candidate"]["samples"]
        median = statistics.median(samples)
        compiled = here / "runs" / f"compile_query_local_{case}_v5/kernel/ptoas/dynamic_csa_full.cpp"
        canonical = executable(compiled.read_text())
        matching = [
            str(p)
            for p in (run / "build_output").glob("decode_csa_spmd_*/ptoas/dynamic_csa_full.cpp")
            if executable(p.read_text()) == canonical
        ]
        assert matching, "Measured C++ differs from CPU-gated candidate"
        original_us = retained["cases"][case]["baseline_median_us"]
        retained_us = retained["cases"][case]["current_median_us"]
        results[case] = dict(
            task_id=task,
            report=str(run / "result.json"),
            passed=True,
            median_us=median,
            all_samples_us=samples,
            original_median_us=original_us,
            retained_median_us=retained_us,
            ratio_to_original=median / original_us,
            ratio_to_retained=median / retained_us,
            regression_vs_retained_pct=(median / retained_us - 1) * 100,
            strict_numerical_and_guards=True,
            raw_72_role_windows=len(samples),
            measured_cpp_matches_cpu_gate=matching,
        )
    assert all(row["ratio_to_retained"] > 1 for row in results.values())
    report = dict(
        candidate="query_local_v5",
        adopted=False,
        cases=results,
        decision="Reject on both priority cases; do not extend the resource domain or run adoption-only service checks",
        formal_retained_geomean_unchanged=retained["equal_weight_geometric_mean_ratio"],
    )
    if (here / "query_local_v5_C05_phase_comparison.json").exists():
        report["diagnostic"] = "query_local_v5_C05_phase_comparison.json"
    else:
        report["diagnostic_pending"] = "C05 phase-clock probe to locate where the changed schedule adds cost"
    with (here / "query_local_v5_results.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                case: {k: row[k] for k in ("median_us", "retained_median_us", "regression_vs_retained_pct")}
                for case, row in results.items()
            },
            indent=2,
        )
    )


def compare_phases(here):
    before = json.loads((here / "retained_C05_phase_sdk_v1.json").read_text())
    after = json.loads((here / "query_local_v5_C05_phase.json").read_text())
    require_same_sdk(before["shared_toolchain"], after["shared_toolchain"])
    results = {}
    for name in (
        "IDX_SCALE",
        "SCORE_CUBE",
        "SCORE_VECTOR",
        "TOPK",
        "ATTN_PLAN",
        "ATTN_CUBE",
        "ATTN_VECTOR",
        "OB",
        "OUT",
    ):
        old = statistics.median(window["phases"][name]["latest_end_us"] for window in before["windows"])
        new = statistics.median(window["phases"][name]["latest_end_us"] for window in after["windows"])
        results[name] = dict(retained_end_us=old, candidate_end_us=new, diagnostic_shift_us=new - old)
    report = dict(
        performance_eligible=False,
        retained="retained_C05_phase_sdk_v1.json",
        candidate="query_local_v5_C05_phase.json",
        milestones=results,
        interpretation=(
            "Score finishes slightly earlier, while Attention and the final output finish later. "
            "The formal full-L1 regressions reject this candidate."
        ),
        limitations=[
            "These instrumented endpoint differences are not formal performance gains or causal effect sizes",
            "QUERY now publishes after Attention, so neither QUERY nor ATTN_VECTOR interval labels are comparable",
            "Query retile, extra scale reads, local handoff and interleaving changed together; "
            "their costs are not isolated",
        ],
    )
    with (here / "query_local_v5_C05_phase_comparison.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
