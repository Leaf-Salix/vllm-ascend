"""Interleave O_A row blocks by column without changing per-tile math."""
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
    names = ("merge_norm", "proj_a_mm", "quant", "proj_b_mm", "proj_b_act", "hc_post")
    for label, history, batch in (("round05_indexer_comp_overlap", 8192, 40), ("round07_o_a_column_order", 8192, 40), ("round05_indexer_comp_overlap", 131072, 16), ("round07_o_a_column_order", 131072, 16)):
        directory = RESULTS / "csa_split_optimization_20260927" / label / f"h{history}_b{batch}"
        report = json.loads((directory / "swimlane/report.json").read_text())
        windows = []
        for item in report["swimlane_windows"]:
            row = compare.summarize(Path(item["merged_swimlane"]))
            windows.append({"path": row["path"], "phases_us": row["phases_us"],
                            "worker_span_us": row["worker_span_us"],
                            "tasks": {name: row["tasks"][name] for name in names if name in row["tasks"]}})
        cases.append({"label": label, "history": history, "batch": batch, "timing": common.summarize_timing(directory / "timing/report.json"),
                      "windows": windows})
    (ROOT / "report.json").write_text(json.dumps({"baseline": "round05 retained source", "cases": cases,
        "decision": "reverted: no target-tail improvement, no consistent cross-context gain", "limits": "Four independent DFX windows; Worker origins normalized separately. Setup includes dependency gating."}, indent=2)+"\n")
    for c in cases:
        print(c["label"], c["timing"]["body"], [w["phases_us"]["first_worker_to_norm_end"] for w in c["windows"]])
if __name__ == "__main__":
    main()
