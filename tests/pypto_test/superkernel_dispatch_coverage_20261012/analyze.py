# SPDX-License-Identifier: Apache-2.0
"""Summarize the archived diagnostic; no device or project imports required."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean


def main():
    root = Path(__file__).resolve().parent
    report = json.loads((root / "results.json").read_text())
    rounds = []
    for sample in report["samples"]:
        records = sample["records"]
        by_id = {(row[0], row[9]): row for row in records}
        details = sample["publication"]["details"]
        misses = [row for row in details if not row["handoff"]]
        late = [row for row in misses if row["published_lo"] > row["producer_full_finish"]]
        late_ticks = [row["published_lo"] - row["producer_full_finish"] for row in late]
        exceptions = []
        for row in misses:
            if row in late:
                continue
            successor = by_id[row["core"], row["sequence"]]
            predecessor = by_id[successor[0], successor[11]]
            exceptions.append(
                {
                    **row,
                    "predecessor_final_wait_ticks": predecessor[7] - predecessor[5],
                    "publication_lo_after_return_ticks": row["published_lo"] - row["producer_return"],
                    "drain_after_publication_hi_ticks": row["producer_drain"] - row["published_hi"],
                }
            )
        before_return = [row for row in details if row["published_hi"] < row["producer_return"]]
        gated = [row for row in records if row[10] & 4]
        rounds.append(
            {
                "repeat": sample["repeat"],
                "executions": len(records),
                "gated": len(gated),
                "gated_with_prefix_callback": sum(bool(row[13] & 1) for row in gated),
                "pending_capable": len(details),
                "handoffs": sum(row["handoff"] for row in details),
                "pending_without_handoff": len(misses),
                "misses_published_after_full_finish": len(late),
                "misses_published_after_full_finish_by_predecessor": dict(Counter(row["predecessor"] for row in late)),
                "late_publication_ticks": {
                    "min": min(late_ticks),
                    "mean": mean(late_ticks),
                    "max": max(late_ticks),
                    "within_5_ticks": sum(value <= 5 for value in late_ticks),
                },
                "published_before_return": len(before_return),
                "published_before_return_with_handoff": sum(row["handoff"] for row in before_return),
                "notification_submitted": sample["totals"]["notification_submitted"],
                "handoff_after_callback": sample["totals"]["handoff_after_callback"],
                "exceptions": exceptions,
                "exact_states": len(sample["exact"]),
                "guard_results": dict(Counter(sample["guards"].values())),
            }
        )
    summary = {
        "scope": "instrumented two executions only; not uninstrumented performance or compute-overlap proof",
        "ticks_per_us": 50,
        "rounds": rounds,
        "totals": {
            key: sum(row[key] for row in rounds)
            for key in (
                "executions",
                "gated",
                "gated_with_prefix_callback",
                "pending_capable",
                "handoffs",
                "pending_without_handoff",
                "misses_published_after_full_finish",
                "published_before_return",
                "published_before_return_with_handoff",
                "notification_submitted",
                "handoff_after_callback",
            )
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
