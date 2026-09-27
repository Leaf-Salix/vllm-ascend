"""只读scale预取两档证据，复用现有计时与泳道解析器。"""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]
TASKS = ("indexer_score_topk_native_pair_aic", "indexer_score_topk_native_pair_aiv",
         "indexer_topk_query_merge", "qk_pv_aic", "qk_pv_aiv", "merge_norm")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    common = load("common", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    worker = load("worker", RESULTS / "csa_scheduling_20260927/upstream_725/compare.py")
    baseline = json.loads((ROOT.parent / "final_f76b3ad4/cases.json").read_text())
    report = {"baseline_revision": "f76b3ad4", "candidate": "candidate.patch", "retained": True,
              "task_id": "task_20260927_230953_118467913832", "cases": []}
    for history in (8192, 131072):
        folder = RESULTS / "csa_split_optimization_20260927/cache_scale_prefetch" / f"h{history}_b16"
        old = next(c for c in baseline["cases"] if c["history"] == history and c["batch"] == 16)
        timing = common.summarize_timing(folder / "timing/report.json")
        raw = json.loads((folder / "swimlane/report.json").read_text())
        windows = [worker.summarize(Path(w["merged_swimlane"])) for w in raw["swimlane_windows"]]
        if len(windows) != 4 or timing["pto_full"]["samples"] != 20:
            raise ValueError("需要每档20次无profiler计时和4个独立DFX窗口")
        row = {"history": history, "batch": 16, "before": old["timing"], "after": timing,
               "tasks": {}, "windows": [], "swimlane_guard_failures": common.guard_failures(raw["pto_guards"])}
        timed = json.loads((folder / "timing/report.json").read_text())["timing"]
        row["tail"] = {s: {"samples_us": timed[s]["samples_us"], "max_us": max(timed[s]["samples_us"])}
                       for s in ("native", "pto")}
        for name in TASKS:
            before = old["pto_tasks"][name]["window_block_means_us"]
            after = [w["tasks"][name]["kernel_mean_us"] for w in windows]
            row["tasks"][name] = {"before_window_means_us": before, "after_window_means_us": after,
                                  "change_pct": (statistics.mean(after) / statistics.mean(before) - 1) * 100}
        row["windows"] = [{k: w[k] for k in ("path", "worker_span_us", "phases_us")} for w in windows]
        report["cases"].append(row)
        print(history, timing["pto_full"], timing["guard_failures"], row["swimlane_guard_failures"])
    (ROOT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
