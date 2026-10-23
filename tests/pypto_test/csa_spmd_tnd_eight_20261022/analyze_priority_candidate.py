# SPDX-License-Identifier: Apache-2.0
"""Compare a frozen priority candidate to qualified original and retained runs.

All medians are reconstructed from complete 72-role raw L1 windows. Historical
baselines are read only. Matching CPU artifacts validate the executed code;
CPU compilation itself is never treated as a performance observation.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from analyze_physical_capacity import samples_from_windows
from analyze_priority_phase import executable
from measurement_provenance import require_same_sdk


def complete_checks(data):
    assert data["passed"] and data["device_execution"] and data["stage"] == "complete"
    expected = {f"{side}_{stage}" for side in ("reference", "candidate") for stage in ("eager", "graph", "post_l1")}
    assert set(data["checks"]) == expected
    for check in data["checks"].values():
        assert check["guards"] and all(check["guards"].values())
        assert check["numerical"]
        assert all(row["bit_equal"] and row["status"] == "PASS" for row in check["numerical"].values())


def samples(data):
    values = samples_from_windows(data["windows"], "candidate")
    assert values == data["statistics_us"]["candidate"]["samples"]
    assert len(values) >= 10
    return values


def match_build(run, cpu):
    files = [path.relative_to(cpu) for path in cpu.rglob("*.cpp")]
    assert len(files) == 4
    matches = [
        str(build)
        for build in (run / "build_output").glob("decode_csa_spmd_*")
        if all(
            (build / path).is_file() and executable((build / path).read_text()) == executable((cpu / path).read_text())
            for path in files
        )
    ]
    assert matches, f"No four-file executed build matches {cpu}"
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = json.loads((here / f"{args.candidate}.json").read_text())
    tasks = json.loads((here / f"{args.candidate}_tasks.json").read_text())
    retained = json.loads((here / "retained_physical_capacity.json").read_text())
    assert set(tasks) == {"C04", "C05"}
    cases = {}
    for case, task in tasks.items():
        run = here / task["output"]
        result = run / "result.json"
        current_result = here / task["retained_result"]
        original_result = here / task["original_result"]
        data, current, original = [json.loads(path.read_text()) for path in (result, current_result, original_result)]
        complete_checks(data)
        complete_checks(current)
        assert original["passed"] and original["stage"] == "complete"
        assert data["case"] == current["case"] == original["case"]
        assert data["device_id"] == current["device_id"] == original["device_id"] == task["device"]
        require_same_sdk(data["shared_toolchain"], current["shared_toolchain"])
        require_same_sdk(data["shared_toolchain"], original["shared_toolchain"])
        assert data["source_hashes"]["candidate"] == source["files"]
        candidate_values, current_values, original_values = map(samples, (data, current, original))
        median, retained_median, original_median = map(
            statistics.median, (candidate_values, current_values, original_values)
        )
        assert math.isclose(retained_median, retained["cases"][case]["current_median_us"], rel_tol=0, abs_tol=1e-9)
        frozen_denominator = retained["cases"][case]["baseline_median_us"]
        # An even median can be 778.3900000000001 although the frozen JSON is
        # 778.39. Accept binary rounding only, keep all raw samples and use the
        # unchanged official denominator for the goal ratio.
        assert math.isclose(original_median, frozen_denominator, rel_tol=0, abs_tol=1e-9)
        candidate_match = match_build(run, here / task["cpu_kernel"])
        retained_match = match_build(current_result.parent, here / task["retained_cpu_kernel"])
        cases[case] = dict(
            task_id=task["task_id"],
            device=task["device"],
            report=str(result),
            source_manifest=f"{args.candidate}.json",
            strict_checks_and_guards=True,
            measured_four_cpp_matches_cpu_gate=candidate_match,
            retained_four_cpp_matches_current_cpu_gate=retained_match,
            median_us=median,
            retained_median_us=retained_median,
            original_median_us=original_median,
            change_vs_retained_pct=100 * (median / retained_median - 1),
            frozen_denominator_us=frozen_denominator,
            ratio_to_original=median / frozen_denominator,
            all_samples_us=candidate_values,
            retained_samples_us=current_values,
            original_samples_us=original_values,
            raw_72_role_windows=[window["raw"] for window in data["windows"]],
            retained_result=str(current_result),
            original_result=str(original_result),
            baseline_remeasured=False,
            pilot_positive=median < retained_median,
        )
    report = dict(
        candidate=args.candidate,
        adopted=False,
        cases=cases,
        retained_geometric_mean_ratio_unchanged=retained["equal_weight_geometric_mean_ratio"],
        limitation="Same-card same-SDK complete-L1 pilot; adoption still needs applicable state/service/domain gates",
    )
    with (here / f"{args.candidate}_results.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                case: {key: row[key] for key in ("median_us", "retained_median_us", "change_vs_retained_pct")}
                for case, row in cases.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
