# SPDX-License-Identifier: Apache-2.0
"""Qualify a candidate against a certified historical original, without remeasurement."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_physical_capacity import samples_from_windows
from analyze_priority_phase import executable
from measurement_provenance import require_same_sdk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    certificate = json.loads((here / "same_sdk_reused_other_references.json").read_text())
    reference = certificate["cases"][args.case]
    source = json.loads(args.source.read_text())
    data = json.loads((args.run / "result.json").read_text())
    require_same_sdk(data["shared_toolchain"], certificate["shared_toolchain"])
    assert data["passed"] and data["device_execution"] and data["stage"] == "complete"
    assert data["case"] == reference["case"] and data["device_id"] == reference["device_id"]
    assert data["source_hashes"]["candidate"] == source["files"]
    assert set(data["checks"]) == {
        f"{side}_{phase}" for side in ("reference", "candidate") for phase in ("eager", "graph", "post_l1")
    }
    for check in data["checks"].values():
        assert all(check["guards"].values())
        assert all(row["status"] == "PASS" and row["bit_equal"] for row in check["numerical"].values())
    samples = samples_from_windows(data["windows"], "candidate")
    original_windows = [
        dict(raw=raw, side="spmd", incore_span_us=value)
        for raw, value in zip(reference["raw_windows"], reference["all_samples_us"])
    ]
    original = samples_from_windows(original_windows, "spmd")
    assert samples == data["statistics_us"]["candidate"]["samples"]
    assert original == reference["all_samples_us"]
    baseline, median = statistics.median(original), statistics.median(samples)
    assert baseline == reference["median_us"]
    paths = [p.relative_to(args.cpu) for p in args.cpu.rglob("*.cpp")]
    assert len(paths) == 4
    matching = [
        str(build)
        for build in (args.run / "build_output").glob("decode_csa_spmd_*")
        if all(
            (build / p).is_file() and executable((build / p).read_text()) == executable((args.cpu / p).read_text())
            for p in paths
        )
    ]
    assert matching
    retained = json.loads((here / "retained_physical_capacity.json").read_text())["cases"][args.case]
    report = dict(
        case_id=args.case,
        candidate=args.source.stem,
        task_id=args.task,
        adopted=False,
        source_report=str(args.run / "result.json"),
        source_manifest=str(args.source),
        reference_certificate="same_sdk_reused_other_references.json",
        baseline_remeasured=False,
        median_us=median,
        baseline_median_us=baseline,
        retained_median_us=retained["current_median_us"],
        ratio_to_frozen=median / baseline,
        change_vs_retained_pct=100 * (median / retained["current_median_us"] - 1),
        samples_us=samples,
        baseline_samples_us=original,
        raw_windows=[w["raw"] for w in data["windows"]],
        strict_checks_and_guards=True,
        measured_cpp_matches_cpu_gate=matching,
        pilot_positive=median < retained["current_median_us"],
        limitation="Same-device same-SDK qualified historical comparison; no summed phase savings",
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("median_us", "change_vs_retained_pct", "pilot_positive")}))


if __name__ == "__main__":
    main()
