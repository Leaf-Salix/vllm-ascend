"""Summarize existing per-task PMU records; never launch a device job."""

import ast
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    build = root / "standalone_pipe/build"
    tree = ast.parse((build / "kernel_config.py").read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "KERNELS"
                              for target in node.targets))
    names = {}
    for item in assignment.value.elts:
        fields = {ast.literal_eval(key): value for key, value in zip(item.keys, item.values)}
        names[ast.literal_eval(fields["func_id"])] = ast.literal_eval(fields["name"])
    (root / "kernel_names.json").write_text(json.dumps(names, indent=2) + "\n")
    path = build / "dfx_outputs/pmu.csv"
    with path.open() as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"No PMU records: {path}")
    groups = defaultdict(list)
    for row in records:
        if int(row["event_type"]) != 2 or int(row["pmu_total_cycles"]) <= 0:
            raise ValueError(f"Unexpected event type or empty counter window: {row}")
        groups[names[int(row["func_id"])]].append(row)
    counters = [key for key in records[0] if key.endswith("_busy_cycles")]
    summary = []
    for name, rows in groups.items():
        total = sum(int(row["pmu_total_cycles"]) for row in rows)
        summary.append({
            "task": name, "blocks": len(rows),
            "total_cycles_mean": total / len(rows),
            "busy_pct": {key: 100 * sum(int(row[key]) for row in rows) / total for key in counters},
            "icache_miss_rate_pct": 100 * sum(int(row["icache_miss"]) for row in rows)
            / max(1, sum(int(row["icache_req"]) for row in rows)),
        })
    counts = {row["task"]: row["blocks"] for row in summary}
    if counts.get("qk_pv_aic") != 24 or counts.get("qk_pv_aiv") != 48:
        raise ValueError(f"Incomplete qk_pv block coverage: {counts}")
    diagnostic = json.loads((root / "standalone_pipe/report.json").read_text())
    result = {
        "operator_revision": "21d99f8a", "history": 8192, "batch": 40, "event_type": 2,
        "task_id": "task_20260927_154452_175670032077",
        "source": str(path), "kernel_config": str(build / "kernel_config.py"),
        "scope": "Native Q/cache/Top-K; standalone Sparse Attention program; no steady timing claim",
        "aggregation": "sum of per-pipe busy cycles / sum of total cycles across same-name blocks; pipes overlap",
        "comparison": diagnostic["comparison"], "tasks": summary,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
