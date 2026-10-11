"""Summarize the seven paired single-card CSA measurements without retesting."""

import argparse
import json
from pathlib import Path
from statistics import mean, median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    cases = [(131072, b) for b in (4, 8, 16, 24)] + [(8192, b) for b in (16, 24, 32)]
    rows = []
    for history, batch in cases:
        path = args.root / f"h{history}_b{batch}" / "results.json"
        data = json.loads(path.read_text())
        assert data["status"] == "FUNCTIONAL_AND_TIMING_COMPLETE", path
        assert data["variants"] == ["baseline", "all"], path
        ordinary, early = data["summary"]
        pairs = {}
        for sample in data["samples"]:
            pairs.setdefault(sample["repeat"], {})[sample["mode"]] = sample["us"]
        assert len(pairs) == 30 and all(set(pair) == {0, 1} for pair in pairs.values()), path
        deltas = [pair[1] - pair[0] for pair in pairs.values()]
        guards = data["guards"]
        assert all(item["status"] == "PASS" for group in guards.values() for item in group.values()), path
        row = {
            "history": history,
            "batch": batch,
            "device": data["device"],
            "ordinary": ordinary,
            "early_start": early,
            "mean_delta_us": early["mean_us"] - ordinary["mean_us"],
            "mean_reduction_pct": 100 * (1 - early["mean_us"] / ordinary["mean_us"]),
            "max_delta_us": early["max_us"] - ordinary["max_us"],
            "paired_delta_median_us": median(deltas),
            "paired_faster_count": sum(delta < 0 for delta in deltas),
            "pairs": len(pairs),
            "eager_exact": data["eager_exact_by_variant"]["all"],
            "changed_input_exact": data["changed_input_exact_by_variant"]["all"],
            "guards_per_variant": {name: len(group) for name, group in guards.items()},
            "source": str(path.relative_to(args.root)),
            "sdk": data["sdk"],
        }
        rows.append(row)
    long_gain = mean(row["mean_reduction_pct"] for row in rows if row["history"] == 131072)
    short_gain = mean(row["mean_reduction_pct"] for row in rows if row["history"] == 8192)
    report = {
        "cases": rows,
        "long_mean_reduction_pct": long_gain,
        "short_mean_reduction_pct": short_gain,
        "weighted_8_2_mean_reduction_pct": 0.8 * long_gain + 0.2 * short_gain,
        "weighting": "equal weight within each sequence-length group; long:short = 8:2",
    }
    (args.root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    table = [
        "| 档位 | 普通 min | 普通 mean | 普通 max | early_start min | early_start mean | early_start max | mean 耗时减少 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        label = f"{row['history'] // 1024}K/B{row['batch']}"
        values = [row[name][key] for name in ("ordinary", "early_start") for key in ("min_us", "mean_us", "max_us")]
        table.append(f"| {label} | " + " | ".join(f"{value:.2f}" for value in values) + f" | {row['mean_reduction_pct']:+.2f}% |")
    (args.root / "table.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"long={long_gain:+.4f}%, short={short_gain:+.4f}%, weighted={0.8 * long_gain + 0.2 * short_gain:+.4f}%")


if __name__ == "__main__":
    main()
