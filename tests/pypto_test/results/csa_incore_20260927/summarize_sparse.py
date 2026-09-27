"""汇总稀疏注意力核内候选；只读取已有计时和泳道，不执行设备测试。"""

import importlib.util
import json
import statistics
from pathlib import Path

from summarize_v10 import process_events

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
SPLIT = RESULTS / "csa_split_optimization_20260927"
CANDIDATES = {
    "v10": SPLIT / "v10_short_followup/h8192_b40",
    "pv_n128": SPLIT / "pv_n128/h8192_b40",
    "pv_n128_pair": SPLIT / "pv_n128_pair/h8192_b40",
    "kv_index_page_ub": ROOT / "kv_index_page_ub/h8192_b40",
}


def main():
    spec = importlib.util.spec_from_file_location("matrix", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    rows = {}
    for label, case in CANDIDATES.items():
        report = case / "timing/report.json"
        row = {"timing": matrix.summarize_timing(report) if report.exists() else None, "windows": []}
        for path in sorted((case / "swimlane/dfx").glob("**/merged_swimlane.json")):
            values = {}
            for event in process_events(path, "Worker View"):
                name = event["name"].split("(")[0]
                if name in ("qk_pv_aic_spmd", "qk_pv_aiv_spmd", "merge_norm_spmd"):
                    values.setdefault(name, []).append(event["args"]["kernel-duration-us"])
            row["windows"].append({
                "path": str(path),
                "tasks": {name: {"blocks": len(v), "mean_us": statistics.mean(v), "max_us": max(v)}
                          for name, v in values.items()},
            })
        rows[label] = row
    (ROOT / "sparse_candidates.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    for label, row in rows.items():
        timing = row["timing"]
        if timing:
            qk = [w["tasks"]["qk_pv_aic_spmd"]["mean_us"] for w in row["windows"]]
            print(label, "body", round(timing["body"]["mean_us"], 3),
                  "qk_pv", [round(v, 3) for v in qk])


if __name__ == "__main__":
    main()
