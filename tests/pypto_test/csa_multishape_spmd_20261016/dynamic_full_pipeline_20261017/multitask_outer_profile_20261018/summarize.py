#!/usr/bin/env python3
"""保存30次外部设备跨度；Event列另存，不混入L1或作为纯AICPU耗时。"""

import json
import statistics
import sys
from pathlib import Path


out = Path(sys.argv[1])
report = json.loads((out / "report.json").read_text())
if report["status"] != "MEASURED":
    raise ValueError("任务没有完成有效性能测量")
profile = report["device_profile"]
samples = profile["span_us_all"]
if len(samples) != 30 or profile["replays"] != 30:
    raise ValueError("外部设备跨度必须保留全部30次样本")
summary = {
    "status": "MEASURED", "history": report["history"], "batch": report["batch"],
    "query_tokens": 6, "tokens": report["batch"] * 6, "device": report["device"],
    "source_report": str(out / "report.json"), "raw_csv": profile["raw_csv"],
    "metric": "PyTorch profiler全部设备流首个CSA kernel开始至最后一个结束，包含PTO AICPU/AICore调度；只排除图外reset和poison",
    "scope": "正式第4层W8A8、合成历史、原分task服务入口、固定地址图重放",
    "samples_us": samples, "median_us": statistics.median(samples),
    "min_us": min(samples), "max_us": max(samples), "count": len(samples),
    "kernels_per_replay": profile["kernels_per_replay"],
    "event_samples_us": report["timing"]["samples_us"],
    "event_median_us": statistics.median(report["timing"]["samples_us"]),
    "dependency_environment": report["dependency_environment"],
    "comparison_boundary": "本轮只补外层PTO；既有L1与Native列是不同轮次，不能相减为纯AICPU调度开销",
    "static_spmd_executed": False, "native_executed": False,
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in ("status", "history", "batch", "median_us", "event_median_us")}, ensure_ascii=False))
