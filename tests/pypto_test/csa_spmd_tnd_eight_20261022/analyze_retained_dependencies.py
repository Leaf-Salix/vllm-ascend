# SPDX-License-Identifier: Apache-2.0
"""Bound whole-chain opportunities using the retained implementation's evidence."""

import json
import statistics
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    retained = json.loads((here / "retained_merged_schedule.json").read_text())
    ratio = retained["equal_weight_geometric_mean_ratio"]
    remaining = retained["target_ratio"] / ratio
    report = dict(
        performance_eligible=False,
        formal_geometric_mean_ratio=ratio,
        frozen_denominator_unchanged=True,
        equal_relative_improvement_needed_on_affected_cases={
            str(cases): dict(
                affected_cases=cases,
                required_relative_ratio=remaining ** (8 / cases),
                required_reduction_pct=100 * (1 - remaining ** (8 / cases)),
                assumption=(
                    "All other retained medians remain unchanged; identical relative improvement on affected cases"
                ),
            )
            for cases in (2, 4, 8)
        },
        phases={},
        deduplicated_directions=[
            dict(
                direction="Q_B weights resident across M",
                decision="Do not repeat",
                reason="Prior BSH full-K and first-M streaming reuse both measured without whole-L1 benefit",
            ),
            dict(
                direction="Indexer QI weights resident across M",
                decision="Already present",
                reason="K1024xN256 load is already outside the M64 loop",
            ),
            dict(
                direction="Coalesce payload invalidations after producer waits",
                decision="Already present in wait_row_owners; adjacent-stage coalescing previously negative",
            ),
            dict(
                direction="Main Query preparation at Attention consumer",
                decision="New isolated candidate",
                reason=(
                    "Score does not consume main q_flat; retain Q_B acquire, change Query "
                    "production ownership and use local FFTS readiness"
                ),
                costs=[
                    "Extra weight-scale loads",
                    "Changed per-query vector batching",
                    "Per-query event handoff",
                    "Potential delay to the existing Attention pipeline",
                ],
            ),
        ],
        limitations=[
            "Phase probes perturb execution and are not performance samples",
            (
                "Milestone spans include overlap, waits and different last-finishing roles; "
                "they are not exclusive compute durations"
            ),
            "Counter pipe times overlap and are not summed or subtracted",
            (
                "Column readiness assumes existing O_B results and completion timestamps; it "
                "is not a predicted speedup of a different output kernel"
            ),
        ],
    )
    milestones = ["QUERY", "SCORE_VECTOR", "ATTN_PLAN", "ATTN_VECTOR", "OUT"]
    for case in ("C04", "C05"):
        path = here / f"retained_{case}_phase_sdk_v1.json"
        data = json.loads(path.read_text())
        assert data["preinstrumentation_matches_calibrated_cpp"] and not data["performance_eligible"]
        ends = {
            phase: statistics.median(w["phases"][phase]["latest_end_us"] for w in data["windows"])
            for phase in milestones
        }
        report["phases"][case] = dict(
            evidence=str(path),
            milestone_ends_us=ends,
            spans_between_milestones_us={f"{a}_to_{b}": ends[b] - ends[a] for a, b in zip(milestones, milestones[1:])},
            query_median_interval_us=statistics.median(
                w["phases"]["QUERY"]["median_interval_us"] for w in data["windows"]
            ),
        )
        if case == "C05":
            columns = []
            for window in data["windows"]:
                rows = json.loads(Path(window["raw"]).read_text())
                ob = {r["core"]: r["notify_end_us"] for r in rows if r["name"] == "OB"}
                assert set(ob) == set(range(24))
                # This domain uses (group*8+N512_column)%24, all eight groups.
                gates = [max(ob[(group * 8 + col) % 24] for group in range(8)) for col in range(8)]
                columns.append(
                    dict(
                        raw=window["raw"],
                        core_spread_us=max(ob.values()) - min(ob.values()),
                        column_ready_us=gates,
                        lead_over_global_us=[max(gates) - x for x in gates],
                    )
                )
            report["C05_output_column_readiness"] = columns
    with (here / "retained_dependency_opportunities.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "required_reductions": report["equal_relative_improvement_needed_on_affected_cases"],
                "phases": report["phases"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
