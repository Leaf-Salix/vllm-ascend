"""Summarize this candidate's existing traces; do not launch device work."""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]


def collect(common, label, cases):
    rows = []
    for history, batch in cases:
        case = RESULTS / f"csa_split_optimization_20260927/{label}/h{history}_b{batch}"
        path = case / "timing/report.json"
        if not path.exists():
            continue
        row = {"history": history, "batch": batch, "timing": common.summarize_timing(path), "swimlane": []}
        for path in sorted((case / "swimlane/dfx").glob("**/merged_swimlane.json")):
            events = json.loads(path.read_text())["traceEvents"]
            workers = {e["pid"] for e in events if e.get("ph") == "M"
                       and e.get("name") == "process_name" and e.get("args", {}).get("name") == "Worker View"}
            window = {"path": str(path)}
            for prefix, count in (("qk_pv_aic", 24), ("qk_pv_aiv", 48), ("merge_norm", 48)):
                values = [e["args"]["kernel-duration-us"] for e in events if e.get("ph") == "X"
                          and e.get("pid") in workers and e.get("name", "").startswith(prefix)]
                if len(values) != count:
                    raise ValueError(f"Incomplete {prefix} records in {path}: {len(values)}")
                window[prefix] = {"blocks": count, "mean_us": statistics.mean(values)}
            row["swimlane"].append(window)
        rows.append(row)
    return rows


def main():
    spec = importlib.util.spec_from_file_location(
        "csa_matrix_summary", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    rows = collect(common, "sparse_kv_early", ((8192, 40), (8192, 16), (131072, 16), (8192, 24), (8192, 32)))
    result = {"base_revision": "daf06ec8", "patch": "candidate.patch",
              "scope": "ungated early-KV candidate; independent single-card runs, not a seven-case acceptance matrix",
              "task_ids": ["task_20260927_160001_18132223006", "task_20260927_160547_184471913231",
                           "task_20260927_161023_188041421777"], "cases": rows}
    (ROOT / "cases.json").write_text(json.dumps(result, indent=2) + "\n")
    gated = collect(common, "sparse_kv_early_gated", ((8192, 40), (8192, 16)))
    if gated:
        result = {"base_revision": "065bd638", "patch": "gated.patch", "early_kv_min_tokens": 144,
                  "task_id": "task_20260927_161547_191901425553",
                  "scope": "final workload dispatch; B40 timing+DFX and B16 timing, not a seven-case matrix",
                  "fixed_input_assert_log_marker": "GATED_FIXED_INPUT_BIT_EQUAL_PASS",
                  "tail_check": json.loads((ROOT / "gated_tail_b3/report.json").read_text())["comparison"],
                  "standalone_vs_native": json.loads((ROOT / "gated_standalone/report.json").read_text())["comparison"],
                  "cases": gated}
        (ROOT / "gated_report.json").write_text(json.dumps(result, indent=2) + "\n")
        rows += gated
    for row in rows:
        aic = [w["qk_pv_aic"]["mean_us"] for w in row["swimlane"]]
        print(row["history"], row["batch"], row["timing"]["body"],
              (min(aic), max(aic)) if aic else "DFX pending")


if __name__ == "__main__":
    main()
