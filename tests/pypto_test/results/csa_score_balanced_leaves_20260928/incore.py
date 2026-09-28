"""均衡leaf的核内分布与包络；只解析已完成DFX，不运行设备。"""

import argparse
import collections
import importlib.util
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = (
    "indexer_score_topk_native_pair_aic",
    "indexer_score_topk_native_pair_aiv",
    "indexer_topk_query_merge",
    "qk_pv_aic",
    "qk_pv_aiv",
)
FIELDS = ("kernel_mean_us", "kernel_max_us", "worker_envelope_us", "start_spread_us")


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--long-batch", type=int, default=16)
    parser.add_argument("--title", default="均衡leaf")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    cases = []
    lines = [
        f"# {args.title}：核内分布与包络", "",
        "每格为基线→候选，单位μs；数值是四个独立DFX窗口指标的均值。",
        "kernel最大值是每窗口最慢任务，包含核内等待；包络含启动分散，不能视为纯调度耗时。",
        "无profiler CSA/P95、状态门禁和全部原始窗口另见各档evidence.json。", "",
        "| 档位 | task | 核内均值 | 最慢任务 | Worker包络 | 启动分散 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for history, batch in ((131072, args.long_batch), (8192, 16)):
        source = read(args.root / f"h{history}_b{batch}" / "summary.json")
        case = {"history": history, "batch": batch, "operator": source["operator"],
                "state_status": source["status"], "sides": {}}
        for side, data in source["measurements"].items():
            windows = []
            for window in data["worker_windows"]:
                events = read(Path(window["path"]))["traceEvents"]
                pids = {e["pid"] for e in events if e.get("name") == "process_name"
                        and e.get("args", {}).get("name") == "Worker View"}
                tasks = {}
                for name in TASKS:
                    rows = [e for e in events if e.get("pid") in pids and e.get("ph") == "X"
                            and worker.canonical(e["name"]) == name
                            and "kernel-duration-us" in e.get("args", {})]
                    durations = [e["args"]["kernel-duration-us"] for e in rows]
                    core_work = collections.defaultdict(float)
                    for event, duration in zip(rows, durations):
                        core_work[event["tid"]] += duration
                    tasks[name] = {
                        **window["tasks"][name], "kernel_min_us": min(durations),
                        "kernel_max_us": max(durations), "kernel_median_us": statistics.median(durations),
                        "kernel_samples_us": durations, "used_cores": len(core_work),
                        "max_core_accumulated_kernel_us": max(core_work.values()),
                    }
                windows.append({"path": window["path"], "tasks": tasks})
            case["sides"][side] = {
                "windows": windows,
                "means": {name: {field: statistics.mean(w["tasks"][name][field] for w in windows)
                                 for field in FIELDS} for name in TASKS},
            }
        for name in TASKS:
            cells = [f"{case['sides']['baseline']['means'][name][f]:.3f}→"
                     f"{case['sides']['candidate']['means'][name][f]:.3f}" for f in FIELDS]
            lines.append("| " + " | ".join([f"{history // 1024}K/B{batch}", name, *cells]) + " |")
        cases.append(case)
    result = {"scope": "同轮独立DFX，具体源码见各档operator；非Native/整网性能。", "cases": cases}
    (args.root / "incore.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.root / "INCORE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
