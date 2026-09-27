"""汇总同轮本体计时及独立泳道；三query泳道明确引用前一作业。"""

import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    common = load("timing", ROOT.parent / "csa_native_cube_matrix_20260927/summarize.py")
    worker = load("worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    result = {"task_id": "task_20260928_033316_2907819818", "timing": {}, "incore": {}}
    for name in ("baseline_short", "triple_short", "triple_long", "six_long"):
        path = ROOT / name / "timing/report.json"
        row = common.summarize_timing(path)
        raw = json.loads(path.read_text())
        row["history"] = raw["history"]
        row["batch"] = raw["batch"]
        row["samples"] = {side: raw["timing"][side]["samples_us"] for side in ("native", "pto")}
        if any(len(v) != 20 for v in row["samples"].values()):
            raise ValueError(f"{name}: 需要20次正式图计时")
        result["timing"][name] = row
        print(name, row["pto_full"], row["guard_failures"])
    old = json.loads((ROOT.parent / "csa_indexer_triple_20260928/report.json").read_text())
    before = old["measurements"]["candidate"]["windows"]
    raw = json.loads((ROOT / "six_long/swimlane/report.json").read_text())
    after = [worker.summarize(Path(w["merged_swimlane"])) for w in raw["swimlane_windows"]]
    if len(before) != 4 or len(after) != 4:
        raise ValueError("需要两侧各4个DFX窗口")
    result["incore"] = {
        "scope": "三query前轮task_20260928_030634_158754525942；S6本轮，各4个独立DFX窗口。",
        "limits": "核内含搬运/同步等待，不能把其均值变化与独立正式CSA均值相减归因调度。",
        "before_windows": before,
        "after_windows": after,
        "guard_failures": common.guard_failures(raw["pto_guards"]),
        "tasks": {},
    }
    for task in (
        "indexer_score_topk_native_pair_aic",
        "indexer_score_topk_native_pair_aiv",
        "indexer_topk_query_merge",
        "qk_pv_aic",
        "qk_pv_aiv",
        "merge_norm",
    ):
        values = {
            side: [w["tasks"][task]["kernel_mean_us"] for w in windows]
            for side, windows in (("triple", before), ("six", after))
        }
        row = {
            "window_block_means_us": values,
            "change_pct": (statistics.mean(values["six"]) / statistics.mean(values["triple"]) - 1) * 100,
        }
        result["incore"]["tasks"][task] = row
        print(task, row)
    (ROOT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
