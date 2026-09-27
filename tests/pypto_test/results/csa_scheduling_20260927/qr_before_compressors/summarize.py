"""Keep execution, worker admission and unprofiled body measurements separate."""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]
SPLIT = RESULTS / "csa_split_optimization_20260927"
NAMES = (
    "mix_x_rms_norm_spmd", "qr_proj_matmul_spmd", "qr_rms_norm_quant_spmd",
    "qproj_matmul_spmd", "qproj_dequant_rms_nope_rope_spmd", "idx_qr_proj_matmul_spmd",
    "idx_qr_dequant_rope_spmd", "qr_hadamard_matmul_spmd",
    "kv_score_proj_spmd", "kv_score_proj_0_spmd", "indexer_head_coefficients_spmd",
    "indexer_score_topk_native_pair_aic_spmd", "indexer_score_topk_native_pair_aiv_spmd",
    "indexer_topk_query_merge_spmd", "qk_pv_aic_spmd", "merge_norm_spmd",
    "proj_a_mm_1_spmd", "quant_1_spmd", "proj_b_act_1_spmd", "hc_post_spmd",
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace_summary(path):
    events = json.loads(path.read_text())["traceEvents"]
    workers = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
               and e.get("args", {}).get("name") == "Worker View"}
    events = [e for e in events if e.get("ph") == "X" and e.get("pid") in workers
              and "kernel-duration-us" in e.get("args", {})]
    groups = {}
    for name in NAMES:
        selected = [e for e in events if e["name"].split("(")[0] == name]
        # The O_A function suffix differs with static shape specialization.
        if not selected and name == "proj_a_mm_1_spmd":
            selected = [e for e in events if e["name"].startswith("proj_a_mm")]
        if not selected and name in ("quant_1_spmd", "proj_b_act_1_spmd"):
            prefix = name.split("_1_spmd")[0] + "_"
            selected = [e for e in events if e["name"].startswith(prefix)]
        if not selected:
            raise ValueError(f"Missing {name} in {path}")
        starts = [e["ts"] + e["args"]["local_setup_us"] for e in selected]
        ends = [e["ts"] + e["dur"] for e in selected]
        groups[name] = {
            "blocks": len(selected), "first_kernel_start_us": min(starts), "last_kernel_end_us": max(ends),
            "kernel_start_spread_us": max(starts) - min(starts),
            "kernel_envelope_us": max(ends) - min(starts),
            "incore_block_mean_us": statistics.mean(e["args"]["kernel-duration-us"] for e in selected),
            "mean_local_setup_us": statistics.mean(e["args"]["local_setup_us"] for e in selected),
            "max_local_setup_us": max(e["args"]["local_setup_us"] for e in selected),
        }
    qr = groups["qr_proj_matmul_spmd"]
    norm = groups["mix_x_rms_norm_spmd"]
    return {
        "path": str(path), "tasks": groups,
        "qr_first_start_after_norm_end_us": qr["first_kernel_start_us"] - norm["last_kernel_end_us"],
        "worker_envelope_us": max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events),
    }


def main():
    common = load_module("common", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    incore = load_module("incore", RESULTS / "csa_incore_20260927/indexer_fused_ws_restore/summarize.py")
    cases = []
    for label, timing_label, trace_label in (
        ("baseline", "incore_final", "projection_seed_wide"),
        ("candidate", "qr_before_compressors", "qr_before_compressors"),
    ):
        case = {"label": label, "history": 8192, "batch": 40}
        timing = SPLIT / timing_label / "h8192_b40/timing/report.json"
        if timing.exists():
            case["timing"] = common.summarize_timing(timing)
        trace_report = SPLIT / trace_label / "h8192_b40/swimlane/report.json"
        if trace_report.exists():
            report = json.loads(trace_report.read_text())
            case["windows"] = [trace_summary(Path(w["merged_swimlane"])) for w in report["swimlane_windows"]]
            diagnostics = incore.collect(trace_label, 8192, 40)
            case["diagnostics"] = {k: v for k, v in diagnostics.items() if k != "windows"}
        cases.append(case)
    result = {
        "baseline_revision": "7dc17735", "cases": cases,
        "scope": "Same kernels; expose Q_A TaskId and gate the two Compressor projections after it. Timing and DFX are separate runs.",
        "limits": "Start spread includes resource occupancy/multiple waves; not pure scheduler software overhead. Do not sum block means.",
    }
    (ROOT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        print(case["label"])
        if "timing" in case:
            print("body", case["timing"]["body"], "native", case["timing"]["native"],
                  "full", case["timing"]["pto_full"])
        for name in NAMES:
            rows = [w["tasks"][name] for w in case.get("windows", [])]
            if rows:
                print(name, {key: [min(r[key] for r in rows), max(r[key] for r in rows)] for key in
                             ("incore_block_mean_us", "kernel_start_spread_us", "last_kernel_end_us")})


if __name__ == "__main__":
    main()
