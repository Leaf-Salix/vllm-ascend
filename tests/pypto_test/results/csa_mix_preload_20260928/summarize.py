"""汇总原运行时与PR门限0/50；派发至接收不等于纯调度成本。"""

import collections
import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def mix_waits(path, canonical):
    events = json.loads(path.read_text())["traceEvents"]
    processes = {e["pid"]: e["args"]["name"] for e in events if e.get("name") == "process_name"}
    scheduler = {
        (e["args"].get("launch_epoch"), e["args"].get("taskId"), e["tid"]): e
        for e in events if e.get("ph") == "X" and processes.get(e.get("pid")) == "Scheduler View"
    }
    groups = collections.defaultdict(lambda: {"wait_samples_us": [], "missing_scheduler_records": 0})
    for event in events:
        if event.get("ph") != "X" or processes.get(event.get("pid")) != "Worker View":
            continue
        name = canonical(event["name"])
        if not (name.startswith("qk_pv_") or name.startswith("indexer_score_topk_native_pair_")):
            continue
        args = event["args"]
        dispatched = scheduler.get((args.get("launch_epoch"), args.get("taskId"), event["tid"]))
        if dispatched is None:
            groups[name]["missing_scheduler_records"] += 1
        else:
            groups[name]["wait_samples_us"].append(event["ts"] - dispatched["ts"])
    for group in groups.values():
        values = group["wait_samples_us"]
        group.update({"matched": len(values), "mean_us": statistics.mean(values) if values else None,
                      "max_us": max(values) if values else None, "above_50us": sum(v > 50 for v in values)})
    return dict(groups)


def main():
    spec = importlib.util.spec_from_file_location(
        "worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    result = {
        "scope": "原a54运行时 vs PR门限0/50；mode2/atomic0/det1/layer4；各50次无profiler图计时。",
        "limits": "单层正式权重/合成历史，不代替EP16。门限0仍含采样开销；DFX与无profiler计时分开。",
        "cases": [],
    }
    lines = ["# MIX预加载门限对照", "", result["scope"], "", result["limits"], "",
             "| 上下文/B | 运行时 | 均值 μs | p50 μs | P95 μs | 最大值 μs | 均值相对原始 | Native控制均值 μs |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for history in (131072, 8192):
        folder = ROOT / f"h{history}_b16"
        case = {"history": history, "batch": 16, "measurements": {}, "state_checks": {}}
        for baseline in ("original", "gate0"):
            comparison = json.loads((folder / f"{baseline}_vs_gate50/comparison.json").read_text())
            case["state_checks"][baseline] = comparison
            if comparison["status"] != "PASS":
                raise ValueError(comparison["errors"])
        for label in ("original", "gate0", "gate50"):
            timing_path = folder / label / "report.json"
            report = json.loads(timing_path.read_text())
            assert report["timing"]["iters"] == 50
            side = {"source": str(timing_path), "timing": {}}
            for backend in ("native", "pto"):
                measured = report["timing"][backend]
                side["timing"][backend] = {
                    "samples_us": measured["samples_us"],
                    "mean_us": statistics.mean(measured["samples_us"]),
                    "p50_us": measured["us_p50"], "p95_us": measured["us_p95"], "max_us": measured["us_max"],
                }
            if label != "original":
                assert report["runtime_experiment"]["mix_preload_max_remaining_us"] == label.removeprefix("gate")
                swimlane_path = folder / "swimlane" / label / "report.json"
                swimlane = json.loads(swimlane_path.read_text())
                assert len(swimlane["swimlane_windows"]) == 2
                side["worker_windows"] = []
                for window in swimlane["swimlane_windows"]:
                    path = Path(window["merged_swimlane"])
                    stats = worker.summarize(path)
                    stats["mix_dispatch_to_receive"] = mix_waits(path, worker.canonical)
                    side["worker_windows"].append(stats)
            case["measurements"][label] = side
            v = side["timing"]["pto"]
            original = case["measurements"]["original"]["timing"]["pto"]
            delta = (v["mean_us"] / original["mean_us"] - 1) * 100
            lines.append(f"| {history}/16 | {label} | {v['mean_us']:.2f} | {v['p50_us']:.2f} | "
                         f"{v['p95_us']:.2f} | {v['max_us']:.2f} | {delta:+.2f}% | "
                         f"{side['timing']['native']['mean_us']:.2f} |")
        before, after = (case["measurements"][k]["timing"]["pto"] for k in ("gate0", "gate50"))
        case["gate50_vs_gate0_pct"] = {k: (after[k] / before[k] - 1) * 100
                                      for k in ("mean_us", "p50_us", "p95_us", "max_us")}
        print(json.dumps({"history": history, "gate50_vs_gate0_pct": case["gate50_vs_gate0_pct"]}, ensure_ascii=False))
        result["cases"].append(case)
    (ROOT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
