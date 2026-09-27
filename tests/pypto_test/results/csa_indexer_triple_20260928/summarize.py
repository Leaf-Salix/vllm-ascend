"""读取同轮query分组单卡结果，分别报告核内、完整CSA和尾部。"""

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--task-id", default="task_20260928_030634_158754525942")
    parser.add_argument("--baseline-revision", default="2a740c1f")
    args = parser.parse_args()
    root = args.root.resolve()
    common = load("timing", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    worker = load("worker", RESULTS / "csa_scheduling_20260927/upstream_725/compare.py")
    result = {
        "baseline_revision": args.baseline_revision,
        "candidate": "candidate.patch",
        "task_id": args.task_id,
        "scope": "同轮20次无profiler完整CSA；独立4个DFX窗口，核内包含数据搬运与同步等待。",
        "limits": "单卡正式layer4合成输入，不能代替16卡EP16整模型；不同任务核内均值不可相加。",
        "measurements": {},
        "task_changes": {},
    }
    for side in ("baseline", "candidate"):
        folder = root / side
        timing = common.summarize_timing(folder / "timing/report.json")
        measured = json.loads((folder / "timing/report.json").read_text())
        timed = measured["timing"]
        for key in (
            "history",
            "batch",
            "layer_index",
            "seed",
            "weight_nz_mode",
            "variant",
            "deterministic_level",
            "hccl_deterministic",
            "pto_reduction",
        ):
            if side == "baseline":
                result[key] = measured[key]
            elif result[key] != measured[key]:
                raise ValueError(f"{side}: 两侧配置{key}不同")
        raw = json.loads((folder / "swimlane/report.json").read_text())
        windows = [worker.summarize(Path(w["merged_swimlane"])) for w in raw["swimlane_windows"]]
        if len(windows) != 4 or timing["pto_full"]["samples"] != 20:
            raise ValueError(f"{side}: 需要20次计时及4个泳道窗口")
        result["measurements"][side] = {
            "timing": timing,
            "samples": {
                backend: {"samples_us": timed[backend]["samples_us"], "max_us": max(timed[backend]["samples_us"])}
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
    (root / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for side, row in result["measurements"].items():
        print(side, row["timing"], "swimlane guards", row["swimlane_guard_failures"])
    print(json.dumps(result["task_changes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
