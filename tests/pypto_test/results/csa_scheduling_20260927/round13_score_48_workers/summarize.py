"""Require a complete Score cohort, with no merge pre-staging."""
import collections
import importlib.util
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    common = load("timing", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    compare = load("worker", ROOT.parent / "upstream_725/compare.py")
    cases = []
    names = ("mix_x_rms_norm", "qr_proj_matmul", "kv_proj_matmul", "kv_rms_norm_rope", "kv_score_proj", "kv_score_proj_0", "idx_qr_proj_matmul", "qproj_matmul", "qproj_dequant_rms_nope_rope", "idx_qr_dequant_rope", "indexer_head_coefficients", "indexer_score_topk_native_pair_aic", "indexer_score_topk_native_pair_aiv", "indexer_topk_query_merge", "qk_pv_aic")
    for label, history in (("round05_indexer_comp_overlap", 131072), ("round13_score_48_workers", 131072), ("round05_indexer_comp_overlap", 8192), ("round13_score_48_workers", 8192)):
        directory = RESULTS / "csa_split_optimization_20260927" / label / f"h{history}_b16"
        report = json.loads((directory / "swimlane/report.json").read_text())
        windows = []
        for item in report["swimlane_windows"]:
            row = compare.summarize(Path(item["merged_swimlane"]))
            events = json.loads(Path(item["merged_swimlane"]).read_text())["traceEvents"]
            pids = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name" and e.get("args", {}).get("name") == "Worker View"}
            counts = collections.Counter(e["tid"] for e in events if e.get("pid") in pids and e.get("ph") == "X" and "indexer_score_topk_native_pair_aic" in e.get("name", ""))
            windows.append({"path": row["path"], "score_aic_cores": len(counts), "score_max_blocks_per_aic": max(counts.values()), "score_aic_blocks_per_core": dict(counts), "phases_us": row["phases_us"],
                            "worker_span_us": row["worker_span_us"],
                            "tasks": {name: row["tasks"][name] for name in names}})
        cases.append({"label": label, "history": history, "batch": 16, "timing": common.summarize_timing(directory / "timing/report.json"),
                      "windows": windows})
    (ROOT / "report.json").write_text(json.dumps({"baseline": "round05 retained source", "cases": cases,
        "decision": "pending", "limits": "Four independent DFX windows; Worker origins normalized separately. Setup includes dependency gating."}, indent=2)+"\n")
    for c in cases:
        print(c["label"], c["timing"]["body"], [w["phases_us"]["first_worker_to_norm_end"] for w in c["windows"]])
if __name__ == "__main__":
    main()
