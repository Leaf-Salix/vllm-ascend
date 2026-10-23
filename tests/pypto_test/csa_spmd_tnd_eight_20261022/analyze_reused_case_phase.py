# SPDX-License-Identifier: Apache-2.0
"""Bind a phase diagnostic to a qualified historical eight-case measurement."""

import argparse
import collections
import json
import statistics
from pathlib import Path

from analyze_priority_phase import executable
from measurement_provenance import require_same_sdk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    certificate = json.loads((here / "same_sdk_reused_other_references.json").read_text())
    reference = certificate["cases"][args.case]
    diagnostic = json.loads((args.run / "result.json").read_text())
    require_same_sdk(diagnostic["shared_toolchain"], certificate["shared_toolchain"])
    assert reference["passed"] and reference["original_equivalence"]["passed"]
    assert diagnostic["case"] == reference["case"] and diagnostic["passed"]
    assert diagnostic["device_execution"] and diagnostic["stage"] == "complete"
    assert not diagnostic["performance_eligible"] and diagnostic["diagnostic_phase_clock"]
    assert diagnostic["device_id"] == reference["device_id"]
    assert "statistics_us" not in diagnostic and "ratio_to_frozen" not in diagnostic
    assert len(reference["all_samples_us"]) >= 10
    for check in diagnostic["checks"].values():
        assert all(check["guards"].values())
        assert all(v["status"] == "PASS" and v["bit_equal"] for v in check["numerical"].values())
    original = args.run / "instrumentation/original.cpp"
    measured = Path(reference["actual_compiled"]) / "ptoas/dynamic_csa_full.cpp"
    assert executable(original.read_text()) == executable(measured.read_text())
    windows = []
    for window in diagnostic["phase_windows"]:
        rows = json.loads(Path(window["raw"]).read_text())
        grouped = collections.defaultdict(list)
        for row in rows:
            if "interval_us" in row:
                grouped[row["name"]].append(row)
        summary = {}
        for name, selected in grouped.items():
            summary[name] = dict(
                roles=len(selected),
                earliest_end_us=min(r["notify_end_us"] for r in selected),
                latest_end_us=max(r["notify_end_us"] for r in selected),
                completion_spread_us=max(r["notify_end_us"] for r in selected)
                - min(r["notify_end_us"] for r in selected),
                median_interval_us=statistics.median(r["interval_us"] for r in selected),
                median_gm_wait_us=statistics.median(r["wait_us"] for r in selected),
            )
        if args.case == "C01":
            # Phase publications include idle roles. Separate the six actual
            # query owners using the verified q%24 geometry of this source.
            for name, actors in (("ATTN_CUBE", range(6)), ("ATTN_VECTOR", range(24, 36))):
                active = [r for r in grouped[name] if r["core"] in actors]
                assert len(active) == len(actors)
                summary[name]["active_query_owners"] = dict(
                    roles=len(active),
                    latest_end_us=max(r["notify_end_us"] for r in active),
                    median_interval_us=statistics.median(r["interval_us"] for r in active),
                    median_gm_wait_us=statistics.median(r["wait_us"] for r in active),
                )
        windows.append(dict(raw=window["raw"], phases=summary))
    assert len(windows) == 3
    report = dict(
        case_id=args.case,
        source_report=str(args.run / "result.json"),
        performance_eligible=False,
        reference_kind="qualified_historical_original",
        reference_certificate="same_sdk_reused_other_references.json",
        reference_report=reference["source_report"],
        preinstrumentation_matches_measured_cpp=str(measured),
        shared_toolchain=diagnostic["shared_toolchain"],
        windows=windows,
        limitations=[
            "Instrumented endpoints and per-role intervals are not formal performance samples",
            "Subtracting GM wait leaves FIFO, DMA, arithmetic and instrumentation costs",
            "Completion spreads and inter-phase endpoint gaps are not removable-latency estimates",
        ],
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    names = ("QUERY", "SCORE_VECTOR", "ATTN_PLAN", "ATTN_VECTOR", "OA_GROUP_7", "OB", "OUT")
    print(json.dumps({name: statistics.median(w["phases"][name]["latest_end_us"] for w in windows) for name in names}))


if __name__ == "__main__":
    main()
