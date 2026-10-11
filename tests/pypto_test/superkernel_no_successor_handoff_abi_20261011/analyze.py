"""Recompute complete single-block windows and graph event summaries."""
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
report = {}
for name in ("register", "tagged", "cold_entry"):
    samples = json.loads((root / f"{name}_single_events.json").read_text())["measurements"][0]["samples"]
    capture = json.loads((root / f"{name}_single_profile.json").read_text())
    bounds = [item[1] for item in capture["metadata"]["run_boundaries"]]
    values = defaultdict(list)
    counts = Counter()
    for core, token, task, start, end, *rest in capture["aicore_tasks"]:
        index = bisect.bisect_right(bounds, start) - 1
        assert 0 <= index < len(samples) and start <= end
        counts[index] += 1
        values[samples[index]["variant"]].append((end - start) / (capture["metadata"]["clock_freq_hz"] / 1e6))
    assert len(bounds) == len(samples) == len(counts) and set(counts.values()) == {1}
    report[name] = {
        "records": len(samples),
        "profile_us": {
            key: {"count": len(v), "min": min(v), "mean": mean(v), "max": max(v)}
            for key, v in values.items()
        },
        "events_48": [
            {"batch": m["batch"], "stats_us": m["stats"]}
            for m in json.loads((root / f"{name}_48_events.json").read_text())["measurements"]
        ],
    }
(root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({name: r["profile_us"] for name, r in report.items()}, indent=2))
