"""Correlate the twenty profiled replays in submission order without retesting."""
import csv
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(status)
    folder = ROOT / "h131072_b16/pto"
    report = json.loads((folder / "report.json").read_text())
    timing = report["timing"]
    path = next((folder / "profile").rglob("kernel_details.csv"))
    with path.open() as stream:
        kernels = list(csv.DictReader(stream))
    roots = sorted((r for r in kernels if r["Name"].startswith("aicore_kernel_mode")),
                   key=lambda r: float(r["Start Time(us)"]))
    aicpu = sorted((r for r in kernels if r["Name"].startswith("simpler_aicpu_kernel_exec")),
                   key=lambda r: float(r["Start Time(us)"]))
    windows = timing["profiled_replays"]
    if len(roots) != 20 or len(aicpu) != 20 or len(windows) != 20:
        raise ValueError("Expected exactly one root and AICPU execution per ordered replay")
    rows = []
    for window, root, cpu in zip(windows, roots, aicpu):
        begin, cpu_begin = float(root["Start Time(us)"]), float(cpu["Start Time(us)"])
        duration, cpu_duration = float(root["Duration(us)"]), float(cpu["Duration(us)"])
        if not cpu_begin <= begin < begin + duration <= cpu_begin + cpu_duration + 1:
            raise ValueError("Root/AICPU interval pairing is inconsistent")
        rows.append({**window, "root_us": duration, "aicpu_us": cpu_duration,
                     "event_minus_root_us": window["event_us"] - duration,
                     "event_minus_aicpu_us": window["event_us"] - cpu_duration})
    result = {
        "task": task, "operator": "d8627207", "device": report["device"],
        "status": "LARGE_TAIL_NOT_REPRODUCED_NOT_FIXED",
        "unprofiled": {"mean_us": report["mean_us"], "p95_us": timing["us_p95"],
                       "max_us": timing["us_max"], "samples_us": timing["samples_us"]},
        "profiled": rows, "profile_json": str(path.with_name("trace_view.json")),
        "profile_csv": str(path),
        "limits": "Profile changes timing; event minus root is not proven host or scheduler cost. "
                  "This run cannot explain the earlier unprofiled 1391/1456 us outliers.",
    }
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    spans = {key: (min(r[key] for r in rows), max(r[key] for r in rows),
                   statistics.mean(r[key] for r in rows))
             for key in ("event_us", "root_us", "aicpu_us", "event_minus_root_us")}
    lines = ["# 编译入口长档拖尾：本次未复现，不宣称修复", "",
             f"128K/B16，无profiler20次均值{report['mean_us']:.3f}μs，"
             f"P95 {timing['us_p95']:.3f}μs，max {timing['us_max']:.3f}μs。", "",
             "独立profile连续20次，按提交顺序一一对应根kernel/AICPU执行；单位μs。", "",
             "| 范围 | 最小 | 最大 | 均值 |", "| --- | ---: | ---: | ---: |"]
    lines.extend(f"| {key} | {lo:.3f} | {hi:.3f} | {mean:.3f} |"
                 for key, (lo, hi, mean) in spans.items())
    lines += ["", "根外差额含图执行/事件之间的空隙，不等同纯CPU开销或纯调度开销。",
              "本次profile没有捕获首轮的大拖尾，不能用正常窗口替首轮异常归因。",
              "没有改算子，也未加sync_start；先保留异常及诊断证据，后续实际编译入口继续观察。",
              "[逐次配对与原始profile路径](evidence.json)"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
