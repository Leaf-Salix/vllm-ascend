"""Release only Indexer Compressor before Q_A, retaining Attention gating."""
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
    for label, history, batch in (("baseline_2dd51f15", 8192, 16), ("round05_indexer_comp_overlap", 8192, 16), ("o_a_row_parallel", 8192, 40), ("round05_indexer_comp_overlap", 8192, 40), ("baseline_2dd51f15", 131072, 16), ("round05_indexer_comp_overlap", 131072, 16)):
        directory = RESULTS / "csa_split_optimization_20260927" / label / f"h{history}_b{batch}"
        report = json.loads((directory / "swimlane/report.json").read_text())
        windows = []
        for item in report["swimlane_windows"]:
            row = compare.summarize(Path(item["merged_swimlane"]))
            windows.append({"path": row["path"], "phases_us": row["phases_us"],
                            "worker_span_us": row["worker_span_us"],
                            "tasks": {name: row["tasks"][name] for name in names}})
        cases.append({"label": label, "batch": batch, "history": history, "timing": common.summarize_timing(directory / "timing/report.json"),
                      "windows": windows})
    (ROOT / "report.json").write_text(json.dumps({"baseline": "2dd51f15", "cases": cases,
        "decision": "retain: 8K B16/B40 gain; 128K B16 no clear regression or stable gain; seven cases pending", "limits": "Four independent DFX windows; Worker origins normalized separately. Setup includes dependency gating."}, indent=2)+"\n")
    for c in cases:
        print(c["label"], c["timing"]["body"], [w["phases_us"]["first_worker_to_norm_end"] for w in c["windows"]])
if __name__ == "__main__":
    main()
