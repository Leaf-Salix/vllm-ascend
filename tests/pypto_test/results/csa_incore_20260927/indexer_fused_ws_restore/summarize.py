"""Collect existing representative task traces and functional diagnostics."""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]
TASKS = ("indexer_score_topk_native_pair_aic_spmd", "indexer_score_topk_native_pair_aiv_spmd",
         "indexer_head_coefficients_spmd", "indexer_topk_query_merge_spmd")


def collect(label, history, batch, names=TASKS):
    directory = RESULTS / f"csa_split_optimization_20260927/{label}/h{history}_b{batch}/swimlane"
    path = directory / "report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    spec = importlib.util.spec_from_file_location("common_summary", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    windows = []
    for item in report["swimlane_windows"]:
        trace = Path(item["merged_swimlane"])
        if not trace.is_absolute():
            trace = directory / trace
        events = json.loads(trace.read_text())["traceEvents"]
        workers = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
                   and e.get("args", {}).get("name") == "Worker View"}
        selected = {}
        for name in names:
            values = [e["args"]["kernel-duration-us"] for e in events if e.get("ph") == "X"
                      and e.get("pid") in workers and e.get("name", "").split("(")[0] == name]
            if not values:
                raise ValueError(f"Missing {name}: {trace}")
            selected[name] = {"blocks": len(values), "mean_us": statistics.mean(values), "max_us": max(values)}
        windows.append({"path": str(trace), "tasks": selected})
    return {
        "history": history, "batch": batch, "report": str(path), "windows": windows,
        "guard_failures": common.guard_failures({k: report[k] for k in ("native_guards", "pto_guards")}),
        "pto_native": {name: {key: values.get(key) for key in ("status", "nonfinite", "max_abs", "rmse", "mismatches")}
                       for name, values in report["pto_native"].items()},
        "pto_self": {name: {key: values.get(key) for key in ("status", "nonfinite", "max_abs", "rmse", "mismatches")}
                     for name, values in report["pto_self"].items()},
        "topk": {k: report["topk_selection"][k] for k in ("replaced_indices", "invalid_rows", "structural_errors")},
    }


def main():
    cases = []
    for history, batch in ((131072, 16), (8192, 40)):
        candidate = collect("indexer_fused_ws_restore", history, batch)
        if candidate is None:
            continue
        cases.append({"baseline": collect("sparse_cross_query", history, batch), "candidate": candidate})
    result = {"baseline_operator_revision": "da2e2368", "source_base_revision": "b9b05001",
              "patch": "candidate.patch", "task_id": "task_20260927_171950_253397227547",
              "scope": "Four DFX windows plus existing single-layer diagnostics; no new unprofiled body timing",
              "cases": cases}
    (ROOT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        for label, row in case.items():
            print(label, row["history"], row["batch"], "guards", row["guard_failures"], "output", row["pto_native"]["x_out"])
            for name in TASKS:
                values = [w["tasks"][name]["mean_us"] for w in row["windows"]]
                print(name, min(values), max(values))


if __name__ == "__main__":
    main()
