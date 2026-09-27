"""Read the existing B40 pilot; no device execution."""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]


def collect(common, label):
    directory = RESULTS / f"csa_split_optimization_20260927/{label}/h8192_b40"
    windows = []
    for path in sorted((directory / "swimlane/dfx").glob("**/merged_swimlane.json")):
        events = json.loads(path.read_text())["traceEvents"]
        workers = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
                   and e.get("args", {}).get("name") == "Worker View"}
        tasks = {}
        for name in ("kv_score_proj_spmd", "kv_score_proj_0_spmd"):
            values = [e["args"]["kernel-duration-us"] for e in events if e.get("ph") == "X"
                      and e.get("pid") in workers and e.get("name", "").split("(")[0] == name]
            if len(values) != 24:
                raise ValueError(f"Expected 24 {name} blocks in {path}, got {len(values)}")
            tasks[name] = {"blocks": len(values), "mean_us": statistics.mean(values), "max_us": max(values)}
        windows.append({"path": str(path), "tasks": tasks})
    return {"timing": common.summarize_timing(directory / "timing/report.json"), "windows": windows}


def main():
    spec = importlib.util.spec_from_file_location("csa_matrix_summary", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    result = {"baseline_operator_revision": "da2e2368", "patch": "candidate.patch",
              "task_id": "task_20260927_171130_243676420225", "scope": "single-card 8K/B40 pilot; no seven-case or model acceptance",
              "baseline": collect(common, "sparse_cross_query"), "candidate": collect(common, "compressor_combined")}
    (ROOT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    for label in ("baseline", "candidate"):
        row = result[label]
        print(label, row["timing"])
        for task in ("kv_score_proj_spmd", "kv_score_proj_0_spmd"):
            values = [w["tasks"][task]["mean_us"] for w in row["windows"]]
            print(task, (min(values), max(values)) if values else "DFX pending")


if __name__ == "__main__":
    main()
