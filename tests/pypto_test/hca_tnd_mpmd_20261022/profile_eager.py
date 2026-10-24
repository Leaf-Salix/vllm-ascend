# SPDX-License-Identifier: Apache-2.0
"""直接调用的外部 profiling：不创建图，不调用 ACLGraph replay。

每次调用都用 CPU 标记包住，并在标记结束前同步设备。因此同一 trace 的
Ascend Hardware 事件必须完整落在该标记中。恢复状态、输出填毒均在标记外。
这避免依赖 graph ID，也不按 memcpy 名称删除 Native 层内真实搬运。
"""

import json
import statistics
from pathlib import Path


def extract_windows(trace_path, prefix, count):
    raw = json.loads(Path(trace_path).read_text())
    events = raw if isinstance(raw, list) else raw["traceEvents"]
    device_pids = {event["pid"] for event in events if event.get("ph") == "M"
                   and event.get("name") == "process_name"
                   and event.get("args", {}).get("name") == "Ascend Hardware"}
    if not device_pids:
        raise ValueError("trace 没有 Ascend Hardware 设备进程")
    device_events = [event for event in events if event.get("ph") == "X"
                     and event.get("pid") in device_pids and float(event.get("dur", 0)) > 0]
    windows = []
    for index in range(count):
        name = f"{prefix}/{index}"
        markers = [event for event in events if event.get("ph") == "X" and event.get("name") == name]
        if len(markers) != 1:
            raise ValueError(f"需要唯一 CPU 标记 {name}，得到 {len(markers)}")
        begin = float(markers[0]["ts"])
        end = begin + float(markers[0]["dur"])
        rows = [event for event in device_events if begin <= float(event["ts"]) < end]
        if not rows or any(float(row["ts"]) + float(row["dur"]) > end for row in rows):
            raise ValueError(f"{name} 没有完整包住设备执行；不能猜测窗口边界")
        first = min(float(row["ts"]) for row in rows)
        last = max(float(row["ts"]) + float(row["dur"]) for row in rows)
        merged = []
        for lo, hi in sorted((float(row["ts"]), float(row["ts"]) + float(row["dur"])) for row in rows):
            if not merged or lo > merged[-1][1]:
                merged.append([lo, hi])
            else:
                merged[-1][1] = max(merged[-1][1], hi)
        busy = sum(hi - lo for lo, hi in merged)
        windows.append({"sample": index, "device_span_us": last - first,
                        "cpu_elapsed_us": end - begin,
                        "busy_union_us": busy, "uncovered_gap_us": last - first - busy,
                        "events": len(rows), "streams": sorted({str(row["tid"]) for row in rows}),
                        "task_types": sorted({row.get("args", {}).get("Task Type", "unknown") for row in rows}),
                        "cpu_marker_start_us": begin, "cpu_marker_end_us": end})
    values = [window["device_span_us"] for window in windows]
    return {"execution_mode": "eager_direct", "aclgraph_replay": False,
            "scope": "每次直接调用内全部设备流的最早活动到最晚结束；含派发造成的设备空隙",
            "samples_us": values, "median_us": statistics.median(values),
            "cpu_elapsed_median_us": statistics.median(window["cpu_elapsed_us"] for window in windows),
            "busy_union_median_us": statistics.median(window["busy_union_us"] for window in windows),
            "min_us": min(values), "max_us": max(values), "windows": windows,
            "trace": str(trace_path)}


def profile_call(run, reset, *, output_dir, side, samples=10, validate=None):
    import torch
    import torch_npu

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for _ in range(3):
        reset()
        run()
        torch.npu.synchronize()
    prefix = "HCA_EAGER/" + side
    checks = []
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_dir)),
    ) as trace:
        for index in range(samples):
            reset()
            torch.npu.synchronize()
            with torch.profiler.record_function(f"{prefix}/{index}"):
                run()
                torch.npu.synchronize()
            if validate is not None:
                checks.append(validate())
        trace.step()
    paths = list(output_dir.glob("*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    if len(paths) != 1:
        raise RuntimeError(f"要求唯一 PyTorch trace，得到 {paths}")
    result = extract_windows(paths[0], prefix, samples)
    if validate is not None:
        result["window_checks"] = checks
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
