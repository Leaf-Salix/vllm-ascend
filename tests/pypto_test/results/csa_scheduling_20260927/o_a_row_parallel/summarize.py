"""Measure selective pre-dispatch removal against the current scheduling pilot."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]


def main():
    spec = importlib.util.spec_from_file_location("schedule", ROOT.parent / "qr_before_compressors/summarize.py")
    schedule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schedule)
    common = schedule.load_module("common", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    incore = schedule.load_module("incore", RESULTS / "csa_incore_20260927/indexer_fused_ws_restore/summarize.py")
    cases = []
    for label, batch in (("qr_before_compressors", 40), ("o_a_row_parallel", 40), ("o_a_row_parallel", 24)):
        directory = RESULTS / "csa_split_optimization_20260927" / label / f"h8192_b{batch}"
        case = {"label": label, "history": 8192, "batch": batch}
        timing = directory / "timing/report.json"
        if timing.exists():
            case["timing"] = common.summarize_timing(timing)
        if (directory / "swimlane/report.json").exists():
            report = json.loads((directory / "swimlane/report.json").read_text())
            case["windows"] = [schedule.trace_summary(Path(w["merged_swimlane"])) for w in report["swimlane_windows"]]
            diagnostic = incore.collect(label, 8192, batch)
            case["diagnostics"] = {k: v for k, v in diagnostic.items() if k != "windows"}
        cases.append(case)
    result = {"cases": cases, "patch": "candidate.patch",
              "scope": "Use upstream row-by-column O_A grid with Native NZ input; preserve group dependencies.",
              "limits": "local_setup includes preparation and dependency waiting; a large value alone does not establish a performance defect."}
    (ROOT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        print(case["label"], case["batch"])
        if "timing" in case:
            print("body", case["timing"]["body"], "full", case["timing"]["pto_full"])
        for name in ("proj_a_mm_1_spmd", "quant_1_spmd", "proj_b_act_1_spmd", "hc_post_spmd"):
            rows = [w["tasks"][name] for w in case.get("windows", [])]
            if rows:
                print(name, {key: [min(r[key] for r in rows), max(r[key] for r in rows)] for key in
                             ("mean_local_setup_us", "kernel_start_spread_us", "last_kernel_end_us")})


if __name__ == "__main__":
    main()
