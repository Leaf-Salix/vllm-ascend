"""Collect the direct-dependency A/B without rerunning device measurements."""
import collections
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ((131072, 16), (8192, 24))
TASKS = ("csa_rope_sign", "qr_proj_matmul", "kv_score_proj", "weights_proj",
         "indexer_score_topk_native_pair_aic", "indexer_score_topk_native_pair_aiv",
         "indexer_topk_query_merge", "qk_pv_aic", "qk_pv_aiv", "merge_norm")
FIELDS = ("first_receive_us", "last_end_us", "kernel_mean_us", "start_spread_us", "worker_envelope_us")


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_evidence(path):
    events = read(path)["traceEvents"]
    names = collections.defaultdict(set)
    dummy = []
    for event in events:
        args = event.get("args", {})
        if event.get("ph") != "X":
            continue
        if "kernel-duration-us" in args:
            names[str(args["taskId"])].add(event["name"].split("(")[0])
        elif args.get("phase") == "dummy_task":
            dummy.append({"task_id": str(args["task_id"]), "ts": event["ts"]})
    deps = read(path.with_name("deps.json"))
    explicit = sorted({(a, b) for edge in deps["edges"] if edge["source"] == "explicit"
                       and "wait" in edge["flags"]
                       for a in names.get(edge["pred"], ()) for b in names.get(edge["succ"], ())})
    return {"path": str(path.with_name("deps.json")), "dummy_events": dummy,
            "explicit_real_task_edges": explicit}


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"Device job is not successfully terminal: {status}")
    worker = load("worker_metrics", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    result = {"task": task, "operator": "d3adbe04 + four dummy removals", "cann": "9.2.0-beta.2",
              "scope": "Same-config PTO A/B; manual NPUGraph. Native is a control, not template-aligned acceptance.",
              "cases": []}
    for history, batch in CASES:
        folder = ROOT / f"h{history}_b{batch}"
        if not (folder / "summary.json").exists():
            subprocess.run([sys.executable, str(ROOT.parent / "csa_short_score_sync_20260928/summarize.py"),
                            "--root", str(ROOT), "--output", str(folder / "summary.json"),
                            "--history", str(history), "--batch", str(batch), "--iters", "20", "--windows", "4",
                            "--task", task, "--operator", result["operator"]], check=True)
        summary = read(folder / "summary.json")
        assert summary["task"] == task and summary["status"] == "PASS"
        case = {"history": history, "batch": batch, "status": summary["status"], "checks": summary["checks"],
                "graph_status": summary["graph_status"], "errors": summary["errors"], "sides": {}}
        for side, value in summary["measurements"].items():
            paths = [Path(w["path"]) for w in value["worker_windows"]]
            windows = [worker.summarize(p) for p in paths]
            dependencies = dependency_evidence(paths[0])
            if side == "candidate":
                assert not dependencies["dummy_events"]
                required = {("csa_rope_sign", "kv_proj_matmul"),
                            ("csa_rope_sign", "kv_score_proj"),
                            ("qr_proj_matmul", "kv_score_proj"),
                            ("csa_rope_sign", "kv_score_proj_0")}
                actual_edges = {(worker.canonical(a), worker.canonical(b))
                                for a, b in dependencies["explicit_real_task_edges"]}
                assert required <= actual_edges
            else:
                assert len(dependencies["dummy_events"]) == 4
            case["sides"][side] = {
                "timing_source": value["timing_source"], "timing": value["timing"],
                "dependencies": dependencies,
                "worker_span_us": statistics.mean(w["worker_span_us"] for w in windows),
                "windows": [{"path": w["path"], "tasks": {n: w["tasks"][n] for n in TASKS}} for w in windows],
                "means": {n: {f: statistics.mean(w["tasks"][n][f] for w in windows) for f in FIELDS} for n in TASKS}}
        before, after = (case["sides"][s] for s in ("baseline", "candidate"))
        case["csa_change_pct"] = 100 * (after["timing"]["pto"]["mean_us"] / before["timing"]["pto"]["mean_us"] - 1)
        result["cases"].append(case)
    result["weighted_csa_change_pct"] = (
        .8 * result["cases"][0]["csa_change_pct"] + .2 * result["cases"][1]["csa_change_pct"])
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 去除AICPU dummy：直接任务依赖对照", "", "单位μs；基线→候选。两侧CANN9.2、mode2/atomic0/det0。",
             "正式5预热/20次无profiler计时；四个独立DFX窗口。这里只判断PTO自身改动。", "",
             "| 档位 | CSA均值 | 变化 | P95 | 最大值 | Native控制均值 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        before, after = (case["sides"][s]["timing"] for s in ("baseline", "candidate"))
        cells = [f"{before['pto'][m]:.3f}→{after['pto'][m]:.3f}" for m in ("mean_us", "p95_us", "max_us")]
        native = f"{before['native']['mean_us']:.3f}→{after['native']['mean_us']:.3f}"
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {cells[0]} | "
                     f"{case['csa_change_pct']:+.3f}% | {cells[1]} | {cells[2]} | {native} |")
    lines += ["", "| 档位 | Task | 首次接收 | 最后结束 | 核内均值 | 启动分散 | 包络 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        for name in TASKS:
            before, after = (case["sides"][s]["means"][name] for s in ("baseline", "candidate"))
            cells = [f"{before[f]:.3f}→{after[f]:.3f}" for f in FIELDS]
            lines.append("| " + " | ".join([f"{case['history']//1024}K/B{case['batch']}", name, *cells]) + " |")
    lines += ["", f"长短8:2 CSA变化率：{result['weighted_csa_change_pct']:+.3f}%。", "",
              "任务时间以各窗口首个Worker为原点；核内含等待。不能将多核/多线程重叠耗时相加作为调度开销。",
              "保留原读写依赖，DFX确认4→0 dummy与RoPE/Q_A直接边；状态及图检查见[evidence.json](evidence.json)。",
              "Native列只作环境漂移控制，尚非对齐decode模板后的性能基线。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
