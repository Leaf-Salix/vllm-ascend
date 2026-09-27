"""Read existing V10 Worker traces and Native device traces; no device execution."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent


def read_events(path):
    value = json.loads(path.read_text())
    return value if isinstance(value, list) else value["traceEvents"]


def process_events(path, process_name):
    events = read_events(path)
    pids = {
        e["pid"] for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name"
        and e.get("args", {}).get("name") == process_name
    }
    if not pids:
        raise ValueError(f"Missing {process_name}: {path}")
    return [e for e in events if e.get("ph") == "X" and e.get("pid") in pids]


def main():
    cases = json.loads((RESULTS / "csa_split_optimization_20260927/indexer_progress_v10.json").read_text())
    rows = []
    for case in cases:
        history, batch = case["history"], case["batch"]
        native_path = RESULTS / "csa_native_cube_matrix_20260927/download" / f"h{history}_b{batch}_native_pytorch.json"
        native = defaultdict(list)
        if native_path.exists():
            for event in process_events(native_path, "Ascend Hardware"):
                native[event["name"]].append(event["dur"])
        windows = []
        for window in case["swimlane"]:
            tasks = defaultdict(list)
            for event in process_events(Path(window["path"]), "Worker View"):
                args = event.get("args", {})
                if "kernel-duration-us" in args:
                    tasks[event["name"].split("(")[0]].append(args["kernel-duration-us"])
            windows.append({
                "path": window["path"],
                "tasks": {name: {"blocks": len(values), "mean_us": statistics.mean(values),
                                  "max_us": max(values)} for name, values in sorted(tasks.items())},
            })
        rows.append({
            "history": history, "batch": batch, "operator_revision": case["operator_revision"],
            "native_trace": str(native_path) if native_path.exists() else None,
            "native_kernel_us": dict(native), "pto_windows": windows,
        })
    result = {
        "scope": "Native fused device kernels versus per-window PTO block-mean incore durations; do not add AIC/AIV or compute equal-scope speedups",
        "native_source": "Earlier v7 matrix, unchanged Native implementation; independent capture from V10 PTO windows",
        "cases": rows,
    }
    (ROOT / "v10_incore.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
