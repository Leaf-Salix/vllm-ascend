# SPDX-License-Identifier: Apache-2.0
"""Use same-window cuts and the fixed eight-case denominator to rank work."""

import json
import math
import statistics
from pathlib import Path


def partition(rows):
    names = ("IDX_SCALE", "SCORE_VECTOR", "ATTN_VECTOR", "OUT")
    first = min(row["notify_start_us"] for row in rows if row["name"] == "BEGIN")
    cuts = [first] + [max(row["notify_end_us"] for row in rows if row["name"] == name) for name in names]
    assert cuts == sorted(cuts)
    widths = [right - left for left, right in zip(cuts, cuts[1:])]
    span = cuts[-1] - cuts[0]
    assert math.isclose(sum(widths), span, abs_tol=1e-9)
    labels = ("front_through_idx_scale", "score_related", "topk_plan_attention", "output_chain")
    return dict(
        diagnostic_span_us=span,
        cuts_us=cuts,
        regions={label: dict(us=width, share=width / span) for label, width in zip(labels, widths)},
    )


def main():
    here = Path(__file__).resolve().parent
    retained = json.loads((here / "retained_output_rectangle.json").read_text())
    ratio = retained["equal_weight_geometric_mean_ratio"]
    required = retained["target_ratio"] / ratio
    cases = {}
    for case in ("C04", "C05"):
        source = json.loads((here / f"retained_{case}_phase_sdk_v1.json").read_text())
        windows = []
        for window in source["windows"]:
            raw = Path(window["raw"])
            windows.append(dict(raw=str(raw), **partition(json.loads(raw.read_text()))))
        cases[case] = dict(
            evidence=f"retained_{case}_phase_sdk_v1.json",
            windows=windows,
            median_region_shares={
                name: statistics.median(w["regions"][name]["share"] for w in windows) for name in windows[0]["regions"]
            },
            interpretation="Descriptive global cuts within each replay; overlap and waits remain inside regions",
        )
    report = dict(
        original_denominator=retained["baseline_kind"],
        formal_gmean_ratio=ratio,
        goal_ratio=retained["target_ratio"],
        performance_eligible=False,
        additional_equal_reduction_needed={str(k): 1 - required ** (8 / k) for k in (2, 4, 6, 8)},
        cases=cases,
        source_findings=[
            "G6 Score already has live<=3 Cube fast path and combined three-row 2048-candidate Vector sorting",
            "Native indexer keys remain separate 4096-byte page gathers, two per N64 panel",
            "Each retained Score step has 16 N64 panels and a paired AIC/2AIV READY/ACK protocol",
            "AIV acknowledges the score load before sorting, but next-step consumption still shares its resources",
            "Scale page coalescing already exists and cannot be counted as a new mechanism",
        ],
        priority_key_cost_model={
            "C04": dict(groups=16, score_steps=528, page_load_commands=16896, logical_key_bytes=69206016),
            "C05": dict(groups=24, score_steps=792, page_load_commands=25344, logical_key_bytes=103809024),
            "interpretation": (
                "Source-derived issued page payloads including clamped tail repeats; not measured HBM traffic"
            ),
        },
        decisions=[
            "Prioritize a mechanism applicable across several cases; two-only improvements would each need about 30%",
            "Measure current READY/ACK pressure before deciding whether AIV-assisted key packing is viable",
            "Bulk key packing must budget an extra GM write/read round trip, UB lifetime and additional FIFO events",
            "Keep quant/coefficient fusion at CPU-only prototype stage until its critical-path leverage is justified",
        ],
        limits=[
            "No subtraction of independently sampled medians, no claimed recoverable microseconds",
            "C05 historical diagnostic precedes the output-only rectangle adoption; front source remains matched",
            "Whole-kernel PMU engine times overlap and do not establish a phase-specific bottleneck",
        ],
    )
    with (here / "macro_budget_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                required=report["additional_equal_reduction_needed"],
                shares={case: row["median_region_shares"] for case, row in cases.items()},
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
