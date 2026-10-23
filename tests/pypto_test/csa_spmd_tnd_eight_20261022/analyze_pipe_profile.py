# SPDX-License-Identifier: Apache-2.0
"""Bind complete hardware-PMU evidence to a previously measured implementation."""

import argparse
import json
from pathlib import Path

from analyze_priority_phase import executable
from measurement_provenance import require_same_sdk
from profile_current import summarize_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--reference-cpp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "result.json").read_text())
    reference = json.loads(args.reference_result.read_text())
    require_same_sdk(report["shared_toolchain"], reference["shared_toolchain"])
    assert report["passed"] and report["diagnostic_hardware_profile"] and not report["performance_eligible"]
    assert reference["passed"] and report["case"] == reference["case"]
    assert not any(name in report for name in ("windows", "statistics_us", "ratio_to_frozen"))
    for check in report["checks"].values():
        assert all(check["guards"].values())
        assert all(value["status"] == "PASS" and value["bit_equal"] for value in check["numerical"].values())
    assert {"candidate_post_pmu", "reference_post_pmu"} <= report["checks"].keys()
    profile = report["hardware_profile"]
    assert profile["passed"] and profile["capture_passed"] and profile["signal_padding_passed"]
    assert profile["signal_generation_after"] - profile["signal_generation_before"] == 3
    reconstructed = summarize_csv(profile["kernel_details"])
    assert reconstructed["raw_rows"] == profile["raw_rows"] and reconstructed["summary"] == profile["summary"]
    actual_cpp = Path(report["kernel_configs"]["candidate"]).parent / "ptoas/dynamic_csa_full.cpp"
    assert args.reference_cpp.resolve().is_relative_to((args.reference_result.parent / "build_output").resolve())
    assert executable(actual_cpp.read_text()) == executable(args.reference_cpp.read_text())
    result = dict(
        case_id=report["case"]["id"],
        passed=True,
        performance_eligible=False,
        source_report=str(args.run / "result.json"),
        reference_report=str(args.reference_result),
        selected_candidate_cpp=str(actual_cpp),
        reference_cpp=str(args.reference_cpp),
        complete_cpp_match=True,
        comparison_rule="Remove only full-line // pto: location comments",
        shared_toolchain=report["shared_toolchain"],
        device_id=report["device_id"],
        strict_pre_post_checks_passed=True,
        hardware_profile=profile,
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        report["case"]["id"],
        {
            key: value["median"]
            for key, value in profile["summary"].items()
            if key.endswith("time(us)") or "icache" in key
        },
    )


if __name__ == "__main__":
    main()
