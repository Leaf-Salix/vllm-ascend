"""Collect paired incore, CSA, P95 and exact-state evidence without device execution."""
import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ((131072, 4), (8192, 24))
TASKS = ("indexer_score_topk_native_pair_aic", "indexer_score_topk_native_pair_aiv", "indexer_topk_query_merge")
FIELDS = ("kernel_mean_us", "kernel_max_us", "worker_envelope_us", "start_spread_us",
          "blocks", "max_core_kernel_sum_us")


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-b8", action="store_true")
    args = parser.parse_args()
    task = (ROOT / "task.txt").read_text().strip()
    if subprocess.check_output(["task-submit", "--status", task], text=True).strip() != "completed (exit=0)":
        raise RuntimeError("Device task is not successfully terminal")
    metrics = load("seven_metrics", ROOT.parent / "csa_cann92_incore_seven_20260928/collect.py")
    worker = load("worker_metrics", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    result = {"task": task, "baseline": "d8627207 production: stream2048 and UB roots",
              "operator": "long B4/B8 S6 Key reuse and balanced leaves",
              "variant": "pkg:dsv4_csa_small_long_s6_d8627207", "cann": "9.2.0-beta.2", "cases": [],
              "scope": "Exact paired PTO states/graph; independent DFX and timing. Not EP16 acceptance."}
    cases = CASES
    case_tasks = {f"h{h}_b{b}": task for h, b in cases}
    completed_initial = {}
    if args.include_b8:
        b8_task = (ROOT / "task_b8.txt").read_text().strip()
        if subprocess.check_output(["task-submit", "--status", b8_task], text=True).strip() != "completed (exit=0)":
            raise RuntimeError("B8 device task is not successfully terminal")
        cases = ((131072, 4), (131072, 8), (8192, 24))
        case_tasks["h131072_b8"] = b8_task
        prior = read(ROOT / "evidence.json")
        if prior["task"] != task or prior["variant"] != result["variant"]:
            raise ValueError("Initial evidence belongs to a different experiment")
        completed_initial = {(c["history"], c["batch"]): c for c in prior["cases"]
                             if (c["history"], c["batch"]) in CASES and c["status"] == "PASS"}
    result["tasks_by_case"] = case_tasks
    for history, batch in cases:
        if (history, batch) in completed_initial:
            # The original task and its frozen inputs are terminal; reuse its
            # checked states instead of rereading multi-GiB snapshots again.
            result["cases"].append(completed_initial[(history, batch)])
            continue
        folder = ROOT / f"h{history}_b{batch}"
        for side in ("baseline", "candidate"):
            report = read(folder / side / "report.json")
            if report["variant"] != result["variant"]:
                raise ValueError("Private package was not selected")
            if report["pto_reduction"]["atomic_add"] != 0 or report["effective_weight_nz_mode"] != 2:
                raise ValueError("Reduction/layout differs")
        subprocess.run([sys.executable, str(ROOT.parent / "csa_short_score_sync_20260928/summarize.py"),
                        "--root", str(ROOT), "--output", str(folder / "summary.json"),
                        "--history", str(history), "--batch", str(batch), "--iters", "20", "--windows", "4",
                        "--task", case_tasks[folder.name], "--operator", result["operator"]], check=True)
        summary = read(folder / "summary.json")
        case = {"history": history, "batch": batch, "status": summary["status"], "checks": summary["checks"],
                "graph_status": summary["graph_status"], "errors": summary["errors"], "sides": {}}
        for side, value in summary["measurements"].items():
            windows = [metrics.worker_window(Path(w["path"]), worker) for w in value["worker_windows"]]
            case["sides"][side] = {
                "timing_source": value["timing_source"], "timing": value["timing"],
                "windows": [{"path": w["path"], "tasks": {n: w["tasks"][n] for n in TASKS}} for w in windows],
                "means": {n: {f: statistics.mean(w["tasks"][n][f] for w in windows) for f in FIELDS} for n in TASKS}}
        before, after = (case["sides"][s] for s in ("baseline", "candidate"))
        case["changes_pct"] = {
            "csa": 100 * (after["timing"]["pto"]["mean_us"] / before["timing"]["pto"]["mean_us"] - 1),
            **{n: 100 * (after["means"][n]["kernel_mean_us"] / before["means"][n]["kernel_mean_us"] - 1)
               for n in TASKS}}
        result["cases"].append(case)
    result["weighted_changes_pct"] = {
        m: sum(weight * statistics.mean(c["changes_pct"][m] for c in result["cases"] if c["history"] == h)
               for h, weight in ((131072, .8), (8192, .2)))
        for m in ("csa", *TASKS)}
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 小batch长档S6 Key复用：状态、核时与CSA", "",
             "单位μs；d8627207→小batch长档S6及leaf均衡。两侧CANN9.2，未叠加dummy优化。",
             "正式20次无profiler图计时，独立四个DFX窗口；核时含函数内部等待。", "",
             "| 档位 | CSA均值 | 变化 | P95 | 最大值 | Native控制均值 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        before, after = (case["sides"][s]["timing"] for s in ("baseline", "candidate"))
        pair = lambda m, before=before, after=after: f"{before['pto'][m]:.3f}→{after['pto'][m]:.3f}"
        native = f"{before['native']['mean_us']:.3f}→{after['native']['mean_us']:.3f}"
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {pair('mean_us')} | "
                     f"{case['changes_pct']['csa']:+.3f}% | {pair('p95_us')} | {pair('max_us')} | {native} |")
    lines += ["", "| 档位 | Task | 核内均值 | 最慢核 | 包络 | 启动分散 | block数 | 单核最大核时之和 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        for name in TASKS:
            before, after = (case["sides"][s]["means"][name] for s in ("baseline", "candidate"))
            cells = [f"{before[f]:.3f}→{after[f]:.3f}" for f in FIELDS]
            lines.append("| " + " | ".join([f"{case['history']//1024}K/B{case['batch']}", name, *cells]) + " |")
    lines += ["", "本轮所测长档内部等权，再长短8:2变化率："
              + json.dumps(result["weighted_changes_pct"], ensure_ascii=False), "",
              "八类PTO状态精确对比、A→B→A、metadata/保护区通过；不等同Native或模型token/DSpark验收。",
              "全部窗口路径、原始计时及检查见[evidence.json](evidence.json)。",
              "Native是同环境手工调用控制，可命中私有OPP既有静态包，不等同独立编译半层基线。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
