"""Compare the ordinary-context candidate against both surrounding baselines."""

import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
parts = ("old_before", "current_before", "candidate", "current_after", "old_after")
reports = {name: json.loads((root / name / "results.json").read_text()) for name in parts}
for name, report in reports.items():
    assert report.get("status") == "PASS" and report.get("output_and_guards_exact") is True, name
    assert (report["task_blocks"], report["stages"], report["scalar_steps"], report["replays_per_graph"]) == (48, 17, 256, 8), name
assert len({report["device"] for report in reports.values()}) == 1
rows = []
for length in (1, 17):
    for target in (5.0, 50.0):
        selected = {}
        for name, report in reports.items():
            modes = ("plain",) if name.startswith("old_") else ("plain", "early")
            for mode in modes:
                values = [s["us"] for s in report["samples"]
                          if (s["target_us"], s["length"], s["mode"]) == (target, length, mode)]
                assert len(values) == 30, (name, target, length, mode, len(values))
                selected[name + "_" + mode] = {"min": min(values), "mean": mean(values), "max": max(values)}
        changes = {}
        for mode in ("plain", "early"):
            candidate = selected["candidate_" + mode]["mean"]
            before = selected["current_before_" + mode]["mean"]
            after = selected["current_after_" + mode]["mean"]
            changes[mode] = {"vs_current_before_us": candidate - before,
                             "vs_current_after_us": candidate - after,
                             "vs_current_midpoint_pct": (candidate / mean((before, after)) - 1) * 100}
        original = mean(selected[p + "_plain"]["mean"] for p in ("old_before", "old_after"))
        plain = selected["candidate_plain"]["mean"]
        early = selected["candidate_early"]["mean"]
        rows.append({"length": length, "target_us": target, "stats_us": selected,
                     "candidate_vs_current": changes,
                     "candidate_plain_vs_original_pct": (plain / original - 1) * 100,
                     "candidate_early_vs_plain_us": early - plain,
                     "candidate_early_vs_plain_pct": (early / plain - 1) * 100})
result = {"device": reports["candidate"]["device"], "rows": rows}
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
