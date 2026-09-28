"""Collect the compiled empty-worker A/B and its independent level-4 evidence."""
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
CASES = ((131072, 16), (8192, 24))
VARIANT = "pkg:dsv4_csa_coefficient_active_c93ec723"
STATES = {"x_out", "idx_topk", "swa.0", "compressed.0", "state.0",
          "indexer.0", "indexer.1", "indexer_state.0"}
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def schedule_window(path, expected_workers, helper, worker):
    # Official parser verifies clock-domain join, raw row counts and every block.
    analysis = helper._build_analysis(path.parent, path.parent, 2)
    names = {task: worker.canonical(name) for task, name in analysis.graph.name.items()}

    def task_id(name):
        matches = [task for task, actual in names.items() if actual == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one {name}, found {matches}")
        return matches[0]

    coefficient = task_id("indexer_head_coefficients")
    score = task_id("indexer_score_topk_native_pair_aic")
    if len(analysis.rows_by_task[coefficient]) != expected_workers:
        raise ValueError("Runtime coefficient worker count differs from the candidate policy")
    starts = helper._aggregate_min(analysis.rows_by_task, "start_time_us")
    ends = helper._aggregate_max(analysis.rows_by_task, "end_time_us")
    dispatches = helper._aggregate_min(analysis.rows_by_task, "dispatch_time_us")
    finishes = helper._aggregate_max(analysis.rows_by_task, "finish_time_us")
    origin = min(starts.values())
    missing = [p for p in analysis.preds[score] if not helper._is_alloc(p, analysis) and p not in ends]
    producers = [p for p in analysis.preds[score] if p in ends]
    if not producers:
        raise ValueError("Score has no observed producer")
    last_data = max(producers, key=ends.get)
    ready = max(finishes[p] for p in producers)
    early, early_rows, total_rows = helper._observed_early(score, analysis, 2e6 / analysis.graph.freq)
    summary = worker.summarize(path)
    retained = ("indexer_head_coefficients", "idx_kv_scale_commit", "indexer_score_topk_native_pair_aic",
                "indexer_score_topk_native_pair_aiv", "indexer_topk_query_merge")
    return {
        "path": str(path), "joined_rows": len(analysis.rows), "coefficient_workers": expected_workers,
        "coefficient_end_us": ends[coefficient] - origin,
        "coefficient_finish_us": finishes[coefficient] - origin,
        "coefficient_end_to_finish_us": finishes[coefficient] - ends[coefficient],
        "score_dispatch_us": dispatches[score] - origin, "score_start_us": starts[score] - origin,
        "score_end_us": ends[score] - origin, "score_finish_us": finishes[score] - origin,
        "score_untimed_producers": missing, "score_last_data_producer": None if missing else names[last_data],
        "score_data_ready_us": None if missing else ends[last_data] - origin,
        "score_ready_us": None if missing else ready - origin,
        "score_finish_to_dispatch_us": None if missing else dispatches[score] - ready,
        "score_dispatch_to_start_us": starts[score] - dispatches[score],
        "score_early": early, "score_early_rows": early_rows, "score_rows": total_rows,
        "tasks": {name: summary["tasks"][name] for name in retained},
    }


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"Wait for the same task to finish: {status}")
    torch.set_num_threads(4)
    pair = load("coefficient_compiled_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    worker = load("coefficient_worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    helper = load("coefficient_critical_path", WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    result = {"task": task, "variant": VARIANT, "baseline": "c93ec723", "cases": [],
              "scope": "Compiled single-card CSA A/B; not Native, EP16 or token/DSpark acceptance"}
    for history, batch in CASES:
        folder = ROOT / f"h{history}_b{batch}"
        reports = {s: read(folder / "timing" / s / "report.json") for s in ("baseline", "candidate")}
        for side, report in reports.items():
            expected_source = str(WORKSPACE / f".cache/csa-coefficient-active-c93ec723-{side}")
            if (report["source"], report["variant"], report["batch"], report["history"], report["side"]) != (
                    expected_source, VARIANT, batch, history, "pto"):
                raise ValueError("Wrong frozen source/package/shape/side")
            package_name = VARIANT.removeprefix("pkg:")
            expected_service = Path(expected_source) / "vllm_ascend/ops/pypto" / package_name / "service.py"
            if (report["implementation_package"], report["implementation_source"]) != (
                    f"vllm_ascend.ops.pypto.{package_name}", str(expected_service)):
                raise ValueError("PTO loaded a different implementation package")
            if not report["compiler"]["pto_dispatch_calls"] or report["timing"]["topk_selection"]["structural_errors"]:
                raise ValueError("PTO dispatch or Top-K structure failed")
        sides = {s: pair.analyze_side(folder / "timing" / s) for s in reports}
        for key in ("device", "cann", "requested"):
            if sides["baseline"][key] != sides["candidate"][key]:
                raise ValueError(f"A/B configuration differs: {key}")
        states = {s: torch.load(folder / "timing" / s / "states.pt", map_location="cpu", weights_only=True)
                  for s in reports}
        if any(set(state) != STATES for state in states.values()):
            raise ValueError("Incomplete state coverage")
        checks = {name: compare_tensor(states["candidate"][name], states["baseline"][name], 0, 0) for name in STATES}
        if any(check["status"] != "PASS" for check in checks.values()):
            raise ValueError(f"Cross-version state differs: {checks}")
        del states
        for side, value in sides.items():
            value["samples_us"] = reports[side]["timing"]["samples_us"]
            dfx = read(folder / "swimlane" / side / "report.json")
            windows = dfx["swimlane_windows"]
            if dfx["status"] != "MEASURED" or dfx["variant"] != VARIANT or len(windows) != 4:
                raise ValueError("Incomplete DFX result")
            if (dfx["batch"], dfx["history"], dfx["effective_weight_nz_mode"], dfx["deterministic_level"],
                    dfx["pto_reduction"]["atomic_add"]) != (batch, history, 2, 0, 0):
                raise ValueError("DFX shape/layout/reduction differs")
            if any(not w["exported"] or w["execution"] != "graph_replay" for w in windows):
                raise ValueError("Invalid graph DFX capture")
            value["windows"] = [schedule_window(Path(w["merged_swimlane"]),
                                                48 if side == "baseline" else batch, helper, worker) for w in windows]
        before, after = (sides[s]["mean_us"] for s in ("baseline", "candidate"))
        result["cases"].append({"history": history, "batch": batch, "sides": sides,
                                "state_checks": checks, "csa_change_pct": 100 * (after / before - 1)})
    result["weighted_8_2_change_pct"] = sum(w * c["csa_change_pct"] for w, c in zip((.8, .2), result["cases"]))
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 仅删除系数空worker：真实编译CSA及依赖时序", "",
             "同卡CANN9.2/mode2/atomic0/det0；5次预热、20次无profiler图计时，单位μs。", "",
             "| 档位 | CSA基线→候选 | 变化 | P95 | 最大值 |", "| --- | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        a, b = (case["sides"][s] for s in ("baseline", "candidate"))
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {a['mean_us']:.3f}→{b['mean_us']:.3f} "
                     f"| {case['csa_change_pct']:+.3f}% | {a['us_p95']:.3f}→{b['us_p95']:.3f} "
                     f"| {a['us_max']:.3f}→{b['us_max']:.3f} |")
    lines += ["", f"长短8:2变化率：{result['weighted_8_2_change_pct']:+.3f}%。", "",
              "以下是独立四窗口DFX的均值；时刻相对各窗口首个kernel开始，不与正式CSA相减。", "",
              "| 档位 | 系数worker | 系数end→FIN延迟 | Score首次start时刻 | Score AIC/AIV核时 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        windows = [case["sides"][s]["windows"] for s in ("baseline", "candidate")]
        def average_pair(key, windows=windows):
            return "→".join(f"{statistics.mean(w[key] for w in ws):.3f}" for ws in windows)
        score = []
        for ws in windows:
            values = [statistics.mean(w["tasks"]["indexer_score_topk_native_pair_" + core]["kernel_mean_us"]
                                      for w in ws) for core in ("aic", "aiv")]
            score.append("/".join(f"{value:.3f}" for value in values))
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | 48→{case['batch']} "
                     f"| {average_pair('coefficient_end_to_finish_us')} | {average_pair('score_start_us')} "
                     f"| {'→'.join(score)} |")
    lines += ["", "八类跨版本PTO状态、编译图/eager及保护区通过；未增加独立A→B→A或整模型测试。",
              "系数均值的样本构成随空worker删除而改变，不直接据其增减宣称算术加速。",
              "保留每窗口最后前置、实际early派发及FIN→dispatch；不将相关时序冒称资源阻塞因果。",
              "[原始样本、完整检查和泳道](evidence.json)"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
