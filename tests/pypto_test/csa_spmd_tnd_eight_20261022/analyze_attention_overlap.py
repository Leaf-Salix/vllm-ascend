# SPDX-License-Identifier: Apache-2.0
"""Validate complete raw L1 windows and source identity before judging overlap."""

import argparse
import json
import math
import statistics
from pathlib import Path

from analyze_priority_phase import executable
from measurement_provenance import require_same_sdk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", choices=("attention_overlap_v1", "attention_valid_overlap_v1"), default="attention_overlap_v1"
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    stem = args.candidate.removesuffix("_v1")
    source = json.loads((here / f"{args.candidate}.json").read_text())
    retained = json.loads((here / "retained_merged_schedule.json").read_text())
    tasks = json.loads((here / f"{args.candidate}_tasks.json").read_text())
    cases = {}
    for case, name in (("C04", "h131072_b16_s6"), ("C05", "h131072_b24_s6")):
        run = here / tasks[case]["output"]
        data = json.loads((run / "result.json").read_text())
        original = json.loads((here / "runs" / f"{name}_baseline_sdk_v1/result.json").read_text())
        require_same_sdk(data["shared_toolchain"], original["shared_toolchain"])
        assert data["passed"] and data["device_execution"] and data["stage"] == "complete"
        assert data["case"] == original["case"] and data["source_hashes"]["candidate"] == source["files"]
        assert len(data["checks"]) >= 3
        for check in data["checks"].values():
            assert check["guards"] and all(check["guards"].values())
            assert check["numerical"] and all(row["bit_equal"] for row in check["numerical"].values())
        samples = []
        raw_windows = []
        for window in data["windows"]:
            raw = json.loads(Path(window["raw"]).read_text())
            rows, hz = raw["aicore_tasks"], raw["metadata"]["clock_freq_hz"]
            assert raw["chip_swimlane_level"] == 1 and hz > 0 and window["side"] == "candidate"
            assert len(rows) == len({r[0] for r in rows}) == 72 and len({r[1] for r in rows}) == 1
            assert all(len(r) == 6 and r[4] >= r[3] > 0 for r in rows)
            span = (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / hz
            assert math.isclose(span, window["incore_span_us"], rel_tol=0, abs_tol=1e-8)
            samples.append(span)
            raw_windows.append(window["raw"])
        assert len(samples) >= 10 and samples == data["statistics_us"]["candidate"]["samples"]
        cpu = here / f"runs/compile_{stem}_{case}_v1/kernel/ptoas/dynamic_csa_full.cpp"
        matching = [
            str(p)
            for p in (run / "build_output").glob("decode_csa_spmd_*/ptoas/dynamic_csa_full.cpp")
            if executable(p.read_text()) == executable(cpu.read_text())
        ]
        assert matching
        median = statistics.median(samples)
        current = retained["cases"][case]["current_median_us"]
        baseline = retained["cases"][case]["baseline_median_us"]
        cases[case] = dict(
            task_id=tasks[case]["task_id"],
            report=str(run / "result.json"),
            median_us=median,
            all_samples_us=samples,
            raw_72_role_windows=raw_windows,
            original_median_us=baseline,
            retained_median_us=current,
            ratio_to_original=median / baseline,
            ratio_to_retained=median / current,
            change_vs_retained_pct=100 * (median / current - 1),
            strict_numerical_and_guards=True,
            measured_cpp_matches_cpu_gate=matching,
        )
    report = dict(
        candidate=args.candidate,
        adopted=False,
        cases=cases,
        any_priority_case_improved=any(row["ratio_to_retained"] < 1 for row in cases.values()),
        retained_geometric_mean_ratio_unchanged=retained["equal_weight_geometric_mean_ratio"],
        limitations="Observed complete-L1 comparison; CPU overlap potential does not measure actual overlap duration",
    )
    with (here / f"{args.candidate}_results.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                case: {k: row[k] for k in ("median_us", "retained_median_us", "change_vs_retained_pct")}
                for case, row in cases.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
