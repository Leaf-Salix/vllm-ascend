# SPDX-License-Identifier: Apache-2.0
"""Summarize diagnostic endpoints against the exact same-SDK calibrated baseline."""

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

from measurement_provenance import require_same_sdk


def executable(source):
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("// pto:"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnostic = json.loads((args.run / "result.json").read_text())
    reference = json.loads(args.reference.read_text())
    require_same_sdk(diagnostic["shared_toolchain"], reference["shared_toolchain"])
    assert diagnostic["case"] == reference["case"] and reference["passed"] and reference["calibration"]
    assert diagnostic["passed"] and not diagnostic["performance_eligible"] and diagnostic["diagnostic_phase_clock"]
    assert "statistics_us" not in diagnostic and "ratio_to_frozen" not in diagnostic
    for check in diagnostic["checks"].values():
        assert all(check["guards"].values()) and all(v["bit_equal"] for v in check["numerical"].values())
    original = (args.run / "instrumentation/original.cpp").read_text()
    matching = [
        str(p)
        for p in args.reference.parent.glob("build_output/decode_csa_spmd_*/ptoas/dynamic_csa_full.cpp")
        if executable(original) == executable(p.read_text())
    ]
    assert matching, "Diagnostic compiler input differs from the measured baseline"
    windows = []
    for window in diagnostic["phase_windows"]:
        rows = json.loads(Path(window["raw"]).read_text())
        grouped = collections.defaultdict(list)
        for row in rows:
            if "interval_us" in row:
                grouped[row["name"]].append(row)
        summary = {}
        for name, selected in sorted(grouped.items(), key=lambda item: max(r["notify_end_us"] for r in item[1])):
            summary[name] = dict(
                roles=len(selected),
                earliest_end_us=min(r["notify_end_us"] for r in selected),
                latest_end_us=max(r["notify_end_us"] for r in selected),
                completion_spread_us=max(r["notify_end_us"] for r in selected)
                - min(r["notify_end_us"] for r in selected),
                median_interval_us=statistics.median(r["interval_us"] for r in selected),
                median_gm_wait_us=statistics.median(r["wait_us"] for r in selected),
                latest_cores=[r["core"] for r in sorted(selected, key=lambda r: -r["notify_end_us"])[:4]],
            )
        windows.append(dict(raw=window["raw"], phases=summary))
    assert len(windows) == 3
    result = dict(
        case_id=reference["case"]["id"],
        performance_eligible=False,
        source_report=str(args.run / "result.json"),
        shared_toolchain=diagnostic["shared_toolchain"],
        preinstrumentation_matches_calibrated_cpp=matching,
        original_cpp_sha256=hashlib.sha256(original.encode()).hexdigest(),
        comparison_rule="Remove only full-line // pto: location comments",
        windows=windows,
        limitations=[
            "Instrumentation changes scheduling; these values are never performance-gain samples",
            "Intervals start at the preceding published phase and may differ across roles",
            "GM wait excludes FIFO/FFTS waits; subtraction does not yield pure arithmetic time",
        ],
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    for name in ("META", "QUERY", "SCORE_CUBE", "SCORE_VECTOR", "ATTN_CUBE", "ATTN_VECTOR", "OB", "OUT"):
        selected = [w["phases"][name] for w in windows]
        print(
            name,
            {
                key: round(statistics.median(p[key] for p in selected), 2)
                for key in ("latest_end_us", "completion_spread_us", "median_interval_us", "median_gm_wait_us")
            },
        )


if __name__ == "__main__":
    main()
