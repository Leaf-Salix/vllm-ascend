"""读取同轮双/三query单卡结果，分别报告核内、完整CSA和尾部。"""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
TASKS = (
    "indexer_score_topk_native_pair_aic",
    "indexer_score_topk_native_pair_aiv",
    "indexer_topk_query_merge",
    "qk_pv_aic",
    "qk_pv_aiv",
    "merge_norm",
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    common = load("timing", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    worker = load("worker", RESULTS / "csa_scheduling_20260927/upstream_725/compare.py")
    result = {
        "baseline_revision": "2a740c1f",
        "candidate": "candidate.patch",
        "task_id": "task_20260928_030634_158754525942",
        "history": 131072,
        "batch": 16,
        "scope": "同轮20次无profiler完整CSA；独立4个DFX窗口，核内包含数据搬运与同步等待。",
        "limits": "单卡正式layer4合成输入，不能代替16卡EP16整模型；不同任务核内均值不可相加。",
        "measurements": {},
        "task_changes": {},
    }
    for side in ("baseline", "candidate"):
        folder = ROOT / side
        timing = common.summarize_timing(folder / "timing/report.json")
        timed = json.loads((folder / "timing/report.json").read_text())["timing"]
        raw = json.loads((folder / "swimlane/report.json").read_text())
        windows = [worker.summarize(Path(w["merged_swimlane"])) for w in raw["swimlane_windows"]]
        if len(windows) != 4 or timing["pto_full"]["samples"] != 20:
            raise ValueError(f"{side}: 需要20次计时及4个泳道窗口")
        result["measurements"][side] = {
            "timing": timing,
            "samples": {
                backend: {"samples_us": timed[backend]["samples_us"],
                          "max_us": max(timed[backend]["samples_us"])}
                for backend in ("native", "pto")
            },
            "windows": windows,
            "swimlane_guard_failures": common.guard_failures(raw["pto_guards"]),
        }
    for task in TASKS:
        values = {
            side: [w["tasks"][task]["kernel_mean_us"] for w in row["windows"]]
            for side, row in result["measurements"].items()
        }
        result["task_changes"][task] = {
            "window_block_means_us": values,
            "change_pct": (statistics.mean(values["candidate"]) / statistics.mean(values["baseline"]) - 1) * 100,
        }
    (ROOT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for side, row in result["measurements"].items():
        print(side, row["timing"], "swimlane guards", row["swimlane_guard_failures"])
    print(json.dumps(result["task_changes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
