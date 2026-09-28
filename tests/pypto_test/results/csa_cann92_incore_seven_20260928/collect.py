"""Collect the completed CANN 9.2 seven-case single-card matrix, without NPU work."""

import collections
import csv
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ((131072, 4), (131072, 8), (131072, 16), (8192, 16), (8192, 24), (8192, 32), (8192, 40))
TASKS = (
    "indexer_score_topk_native_pair_aic", "indexer_score_topk_native_pair_aiv",
    "indexer_topk_query_merge", "qk_pv_aic", "qk_pv_aiv", "merge_norm",
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def stats(values):
    if len(values) != 20 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Expected 20 finite positive timing samples")
    return {"samples_us": values, "mean_us": statistics.mean(values),
            "p50_us": statistics.median(values), "p95_us": sorted(values)[18], "max_us": max(values)}


def profile(folder, side):
    paths = list((folder / "profile" / side).glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one exported {side} profile: {paths}")
    with paths[0].open() as stream:
        rows = list(csv.DictReader(stream))
    trace = paths[0].with_name("trace_view.json")
    payload = read(trace)
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    if not rows or not any(e.get("ph") == "X" for e in events):
        raise ValueError(f"Empty profile: {trace}")
    fields = ("Name", "Type", "Duration(us)", "aicore_time(us)", "aiv_time(us)")
    return {"trace": str(trace), "csv": str(paths[0]),
            "kernels": [{k: row[k] for k in fields} for row in rows]}


def worker_window(path, worker):
    result = worker.summarize(path)
    events = read(path)["traceEvents"]
    pids = {e["pid"] for e in events if e.get("name") == "process_name"
            and e.get("args", {}).get("name") == "Worker View"}
    groups = collections.defaultdict(list)
    for event in events:
        if event.get("pid") in pids and "kernel-duration-us" in event.get("args", {}):
            groups[worker.canonical(event["name"])].append(event)
    for name, rows in groups.items():
        work = collections.defaultdict(float)
        counts = collections.Counter(e["tid"] for e in rows)
        for event in rows:
            work[event["tid"]] += event["args"]["kernel-duration-us"]
        result["tasks"][name].update(
            kernel_max_us=max(e["args"]["kernel-duration-us"] for e in rows),
            used_cores=len(work), max_blocks_per_core=max(counts.values()),
            max_core_kernel_sum_us=max(work.values()),
        )
    return result


def collect_case(history, batch, checks, worker):
    folder = ROOT / "layer" / f"h{history}_b{batch}"
    report, swimlane = read(folder / "report.json"), read(folder / "swimlane/report.json")
    config = checks.check_config(report, history, batch)
    if config != checks.check_config(swimlane, history, batch):
        raise ValueError("DFX and timing configuration differ")
    timing = report["timing"]
    if (timing["status"], timing["iters"], timing["warmup"], timing["compact_metadata_policy"]) != (
        "MEASURED", 20, 5, "reuse"
    ):
        raise ValueError("Timing contract differs")
    measurements = {}
    for side in ("native", "pto"):
        values = timing[side]
        if len(set(values["start_timestamps_raw"])) != 20:
            raise ValueError("Timing events did not advance on each replay")
        checks.require_pass(values["guards"], side + " timing guards")
        if values["topk_selection"]["structural_errors"]:
            raise ValueError("Timing Top-K structure failed")
        measurements[side] = stats(values["samples_us"])
    checks.require_pass(timing["pto"]["eager_comparison"], "PTO timing versus eager")
    graph = report["graph"]
    if graph["status"] != "PASS" or [r["input"] for r in graph["replays"]] != ["A", "B", "A"]:
        raise ValueError("A-B-A graph replay failed")
    windows = swimlane["swimlane_windows"]
    if len(windows) != 4:
        raise ValueError("Expected four DFX windows")
    for window in windows:
        if not window["exported"] or (window["layer_index"], window["compact_metadata_policy"],
                                      window["input_source"], window["execution"]) != (
            4, "reuse", "formal_layer_weights_synthetic_history", "graph_replay"
        ):
            raise ValueError("DFX capture contract differs")
    profiles = {s: profile(folder, s) for s in ("native", "pto")}
    for op in ("VllmQuantLightningIndexer", "SparseAttnSharedkv"):
        if sum(row["Type"] == op for row in profiles["native"]["kernels"]) != 1:
            raise ValueError(f"Expected one Native {op} invocation")
    return {
        "history": history, "batch": batch, "device": timing["device"], "config": config,
        "functional_status": "PASS", "timing": measurements,
        "change_pct": (measurements["pto"]["mean_us"] / measurements["native"]["mean_us"] - 1) * 100,
        "sources": {"report": str(folder / "report.json"), "swimlane": str(folder / "swimlane/report.json")},
        "native_arithmetic_diagnostics": report["pto_native"], "topk_selection": report["topk_selection"],
        "profiles": profiles,
        "worker_windows": [worker_window(Path(w["merged_swimlane"]), worker) for w in windows],
    }


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise SystemExit(f"{task}: {status}; collect only after terminal success")
    # Reuse the existing configuration and state contracts, without its historical path routing.
    sys.path.insert(0, str(ROOT.parent / "csa_key_l1_seven_20260928"))
    checks = load_module("existing_layer_contract", ROOT.parent / "csa_key_l1_seven_20260928/collect_layer.py")
    worker = load_module("worker_metrics", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    cases = [collect_case(h, b, checks, worker) for h, b in CASES]
    if len({row["device"] for row in cases}) != 1:
        raise ValueError("This matrix promised one allocated card")
    by_history = {str(h): statistics.mean(r["change_pct"] for r in cases if r["history"] == h)
                  for h in (131072, 8192)}
    result = {
        "operator": "e33d842a", "task": task, "cann": "9.2.0-beta.2", "cases": cases,
        "mean_change_pct_by_history": by_history,
        "weighted_change_pct": .7 * by_history["131072"] + .3 * by_history["8192"],
        "scope": "Single-card layer4 formal weights/synthetic history; graph timing excludes profiling. "
                 "Functional PASS means self-replay, metadata/guards and structure; "
                 "not Native bitwise or model acceptance.",
    }
    (ROOT / "matrix.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# CANN 9.2：e33d842a七档单卡结果", "",
             "同一卡、同一已保留算子；每侧5次预热/20次无profiler图计时。单位μs。",
             "Native算术差异单列于matrix.json；自重放通过不等于Native逐bit或整网token/DSpark通过。", "",
             "| 档位 | Native均值 | PTO均值 | 变化 | Native/PTO P95 | Native/PTO最大值 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in cases:
        n, p = (row["timing"][s] for s in ("native", "pto"))
        lines.append(f"| {row['history']//1024}K/B{row['batch']} | {n['mean_us']:.2f} | {p['mean_us']:.2f} | "
                     f"{row['change_pct']:+.2f}% | {n['p95_us']:.2f}/{p['p95_us']:.2f} | "
                     f"{n['max_us']:.2f}/{p['max_us']:.2f} |")
    lines += ["", f"各history内batch等权后，七三耗时变化{result['weighted_change_pct']:+.3f}%。",
              "不使用权重覆盖P95异常；不与旧9.0或不同候选拼表。", "", "## Native独立kernel profile", "",
              "以下为单次独立profile的原始duration/aicore_time/aiv_time，非20次正式计时均值。",
              "QLI包含本地Top-K/最终归并，Sparse包含其内部规约；与PTO拆分任务范围不同，不直接相减归因。", "",
              "| 档位 | Native kernel | Duration | aicore_time | aiv_time |",
              "| --- | --- | ---: | ---: | ---: |"]
    for row in cases:
        for kernel in row["profiles"]["native"]["kernels"]:
            if kernel["Type"] in ("VllmQuantLightningIndexer", "SparseAttnSharedkv"):
                lines.append(f"| {row['history']//1024}K/B{row['batch']} | {kernel['Type']} | "
                             f"{kernel['Duration(us)']} | {kernel['aicore_time(us)']} | {kernel['aiv_time(us)']} |")
    lines += ["", "## PTO独立四窗口DFX", "",
              "每格为四个窗口指标的均值。最慢核指各窗口kernel最大值；核时包含内部等待。",
              "包络含启动分散，不能称为纯调度时间；各任务有交叠，不累加成整层时长。", "",
              "| 档位 | Task | 核内均值 | 最慢核 | 包络 | 启动分散 | 单核最多份数 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in cases:
        for name in TASKS:
            items = [w["tasks"][name] for w in row["worker_windows"]]
            cells = [f"{statistics.mean(x[key] for x in items):.3f}" for key in
                     ("kernel_mean_us", "kernel_max_us", "worker_envelope_us", "start_spread_us")]
            lines.append("| " + " | ".join([f"{row['history']//1024}K/B{row['batch']}", name, *cells,
                                             str(max(x["max_blocks_per_core"] for x in items))]) + " |")
    lines += ["", "## 原始JSON", "", "每档两份PyTorch JSON及四份PTO泳道；均为独立采样。", ""]
    for row in cases:
        label = f"{row['history']//1024}K/B{row['batch']}"
        for side, data in row["profiles"].items():
            lines.append(f"- [{label} {side} PyTorch]({Path(data['trace']).relative_to(ROOT)})")
        for i, window in enumerate(row["worker_windows"]):
            lines.append(f"- [{label} PTO泳道{i}]({Path(window['path']).relative_to(ROOT)})")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cases": len(cases), "device": cases[0]["device"],
                      "weighted_change_pct": result["weighted_change_pct"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
