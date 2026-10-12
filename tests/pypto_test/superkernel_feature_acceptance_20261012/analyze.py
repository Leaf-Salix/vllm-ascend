"""Recompute the bounded early-start acceptance from the retained raw samples."""

import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
reports = {name: json.loads((root / "steady" / name / "results.json").read_text())
           for name in ("before", "current", "after")}
rows = []
for target in (5.0, 50.0):
    selected = {}
    for name, report in reports.items():
        for mode in ("plain", "early") if name == "current" else ("plain",):
            values = [r["us"] for r in report["samples"]
                      if (r["target_us"], r["length"], r["mode"]) == (target, 17, mode)]
            assert len(values) == 30
            selected[f"{name}_{mode}"] = {"min": min(values), "mean": mean(values), "max": max(values)}
    baseline = mean(selected[key]["mean"] for key in ("before_plain", "after_plain"))
    unmarked = selected["current_plain"]["mean"]
    marked = selected["current_early"]["mean"]
    rows.append({"target_us": target, "stats_us": selected,
                 "unmarked_mean_change_us": unmarked - baseline,
                 "unmarked_mean_change_pct": (unmarked / baseline - 1) * 100,
                 "early_mean_change_us": marked - unmarked,
                 "early_mean_change_pct": (marked / unmarked - 1) * 100})
functional = {}
for kind in ("aiv", "aic"):
    data = json.loads((root / f"function_v2/kernels/local_{kind}/ready_results.json").read_text())
    functional[kind] = {
        "overlaps": [len(s["overlaps"]) for s in data["samples"]],
        "remaining_compute_wait_us": [
            min((row[7] - row[5]) / 50 for s in data["samples"] for row in s["a"]),
            max((row[7] - row[5]) / 50 for s in data["samples"] for row in s["a"]),
        ],
    }
output = {"chain": rows, "functional": functional}
(root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
