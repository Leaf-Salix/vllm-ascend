"""Compare PV N128/K128 with unchanged sparse softmax and scheduling."""
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
    compare = load("worker", RESULTS / "csa_scheduling_20260927/upstream_725/compare.py")
    cases = []
    names = ("mix_x_rms_norm", "qr_proj_matmul", "kv_proj_matmul", "kv_rms_norm_rope", "kv_score_proj", "kv_score_proj_0", "idx_qr_proj_matmul", "qproj_matmul", "qproj_dequant_rms_nope_rope", "idx_qr_dequant_rope", "indexer_head_coefficients", "indexer_score_topk_native_pair_aic", "indexer_score_topk_native_pair_aiv", "indexer_topk_query_merge", "qk_pv_aic", "qk_pv_aiv", "merge_norm")
    for label, history in (("round05_indexer_comp_overlap", 131072), ("incore_sparse_pv_n128_pair", 131072), ("round05_indexer_comp_overlap", 8192), ("incore_sparse_pv_n128_pair", 8192)):
        directory = RESULTS / "csa_split_optimization_20260927" / label / f"h{history}_b16"
        report = json.loads((directory / "swimlane/report.json").read_text())
        windows = []
        for item in report["swimlane_windows"]:
            row = compare.summarize(Path(item["merged_swimlane"]))
            events = json.loads(Path(item["merged_swimlane"]).read_text())["traceEvents"]
            pids = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name" and e.get("args", {}).get("name") == "Worker View"}
            counts = collections.Counter(e["tid"] for e in events if e.get("pid") in pids and e.get("ph") == "X" and compare.canonical(e.get("name", "")) == "indexer_score_topk_native_pair_aic")
            windows.append({"path": row["path"], "score_aic_cores": len(counts), "score_max_blocks_per_aic": max(counts.values()), "score_aic_blocks_per_core": dict(counts), "phases_us": row["phases_us"],
                            "worker_span_us": row["worker_span_us"],
                            "tasks": {name: row["tasks"][name] for name in names}})
        cases.append({"label": label, "history": history, "batch": 16, "timing": common.summarize_timing(directory / "timing/report.json"),
                      "windows": windows})
    b40 = []
    for label in ("round05_indexer_comp_overlap", "incore_sparse_pv_n128_pair"):
        path = RESULTS / "csa_split_optimization_20260927" / label / "h8192_b40/swimlane/report.json"
        raw = json.loads(path.read_text())
        b40.append({"label": label, "report": str(path),
                    "windows": [compare.summarize(Path(w["merged_swimlane"])) for w in raw["swimlane_windows"]],
                    "guard_failures": common.guard_failures(raw["pto_guards"]),
                    "nonfinite_by_tensor": {key: value["nonfinite"] for key, value in raw["pto_native"].items() if value.get("nonfinite")},
                    "output": {key: raw["pto_native"]["x_out"][key] for key in ("status", "max_abs", "rmse")},
                    "topk": {key: raw["topk_selection"][key] for key in ("replaced_indices", "invalid_rows", "structural_errors")}})
    (ROOT / "b40_windows.json").write_text(json.dumps(b40, indent=2)+"\n")
    (ROOT / "report.json").write_text(json.dumps({"baseline": "round05 retained source", "cases": cases,
        "decision": "保留PV分块；B40核内均值降低2.68%，128K/B16降低1.89%，8K/B16持平；本体长尾另记。", "limits": "Four independent DFX windows; Worker origins normalized separately. Setup includes dependency gating."}, indent=2)+"\n")
    for c in cases:
        print(c["label"], c["timing"]["body"], [w["phases_us"]["first_worker_to_norm_end"] for w in c["windows"]])
if __name__ == "__main__":
    main()
