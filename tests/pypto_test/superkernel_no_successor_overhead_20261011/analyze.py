"""Recompute per-task windows; graph event time is a separate measurement."""

import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent


def profile_stats(prefix):
    capture = json.loads((ROOT / f"{prefix}_profile.json").read_text())
    events = json.loads((ROOT / f"{prefix}_events.json").read_text())
    samples = events["measurements"][0]["samples"]
    boundaries = [v[1] for v in capture["metadata"]["run_boundaries"]]
    assert len(boundaries) == len(samples)
    ticks_per_us = capture["metadata"]["clock_freq_hz"] / 1e6
    by_variant = defaultdict(list)
    run_counts = Counter()
    for core, token, task, start, end, *rest in capture["aicore_tasks"]:
        index = bisect.bisect_right(boundaries, start) - 1
        assert 0 <= index < len(samples) and start <= end
        run_counts[index] += 1
        by_variant[samples[index]["variant"]].append((end - start) / ticks_per_us)
    # The three single-block captures must each contain one task per replay.
    # The earlier 48-block capture was incomplete and is diagnostic only.
    if prefix != "48block_wait":
        assert len(run_counts) == len(samples) and set(run_counts.values()) == {1}
    return {
        "runs": len(samples),
        "records": len(capture["aicore_tasks"]),
        "records_per_run": dict(sorted(run_counts.items())),
        "stats_us": {
            name: {"count": len(v), "min": min(v), "mean": mean(v), "median": median(v), "max": max(v)}
            for name, v in by_variant.items()
        },
    }


report = {
    prefix: profile_stats(prefix)
    for prefix in ("48block_wait", "single_block", "hot_wait_single", "inline_wait")
}
report["hot_wait_48_events"] = [
    {"batch": m["batch"], "stats_us_per_launch": m["stats"]}
    for m in json.loads((ROOT / "hot_wait_48_events.json").read_text())["measurements"]
]
report["overlap"] = {}
for kind in ("aiv", "aic"):
    trace = json.loads((ROOT / f"hot_wait_{kind}_trace.json").read_text())
    report["overlap"][kind] = [len(s["overlaps"]) for s in trace["samples"]]
(ROOT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
for name, result in report.items():
    print(name, result.get("stats_us", result) if isinstance(result, dict) else result)
