"""配对既有 Scheduler/Worker 记录，定位 KV 分组在忙核上的预派发。"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "csa_kv_adaptive_20260928/h8192_b40/swimlane/candidate/report.json"


def core_id(event):
    match = re.search(r"CoreId:(\d+)", event["args"]["event-hint"])
    if match is None:
        raise ValueError(event)
    return int(match[1])


def analyze(path):
    events = json.loads(path.read_text())["traceEvents"]
    pids = {e["args"]["name"]: e["pid"] for e in events if e.get("name") == "process_name"}
    workers = [e for e in events if e.get("pid") == pids["Worker View"] and e.get("ph") == "X"
               and "kernel-duration-us" in e.get("args", {})]
    schedulers = [e for e in events if e.get("pid") == pids["Scheduler View"] and e.get("ph") == "X"]
    origin = min(e["ts"] for e in workers)
    blocks = []
    for worker in workers:
        if not worker["name"].startswith("kv_proj_matmul"):
            continue
        core = core_id(worker)
        matches = [e for e in schedulers if e["args"].get("taskId") == worker["args"]["taskId"]
                   and core_id(e) == core]
        if len(matches) != 1:
            raise ValueError(f"task/core配对不唯一: {worker}")
        dispatch = matches[0]["ts"]
        previous = [e for e in workers if core_id(e) == core and e["ts"] < worker["ts"]]
        previous = max(previous, key=lambda e: e["ts"], default=None)
        block = {
            "core": core, "scheduler_dispatch_us": dispatch - origin,
            "worker_receive_us": worker["ts"] - origin,
            "kernel_start_us": worker["ts"] + worker["args"]["local_setup_us"] - origin,
            "kernel_end_us": worker["ts"] + worker["dur"] - origin,
            "dispatch_to_receive_us": worker["ts"] - dispatch,
        }
        if previous:
            block.update(previous_task=previous["name"],
                         previous_end_us=previous["ts"] + previous["dur"] - origin,
                         dispatched_before_previous_end_us=previous["ts"] + previous["dur"] - dispatch)
        blocks.append(block)
    if len(blocks) != 12:
        raise ValueError(f"预期12个KV块，实际{len(blocks)}")
    return {"path": str(path), "blocks": sorted(blocks, key=lambda b: b["core"])}


def main():
    source = json.loads(SOURCE.read_text())
    result = {
        "source": str(SOURCE),
        "scope": "88d0744f等效算子，B40四个既有DFX窗口；同份转换时间轴按task/core配对。",
        "limits": "dispatch到receive包含等待，不等于纯调度CPU时间；不能据此估计整层或整网收益。",
        "windows": [analyze(Path(w["merged_swimlane"])) for w in source["swimlane_windows"]],
    }
    (ROOT / "pending_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for i, window in enumerate(result["windows"]):
        delayed = [b for b in window["blocks"] if b.get("dispatched_before_previous_end_us", 0) > 5]
        print(i, [(b["core"], round(b["dispatch_to_receive_us"], 2), b["previous_task"]) for b in delayed])


if __name__ == "__main__":
    main()
