# SPDX-License-Identifier: Apache-2.0
"""从服务trace中定位完整PTO算子的所有设备活动，不把服务前导计入AICPU列。"""

import json
from pathlib import Path
import statistics


def operator_interval(service_profile):
    """以根Torch op的进入和外层同步完成界定事件归属，耗时仍取设备首尾。

    根op返回时设备可能尚未完成，不能用CPU op的dur当设备时间或截断异步活动。
    每窗只允许一个PTO根op，前面的唯一Native compact producer须已结束。
    根op之后的全部设备stream事件（包括Event、DMA、AICPU、MIX）都保留，
    不按kernel名称删除PTO内部活动，也不以L1加常数估算外部时间。
    """
    raw = json.loads(Path(service_profile["trace"]).read_text())
    events = raw if isinstance(raw, list) else raw["traceEvents"]
    device_pids = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
                   and e.get("args", {}).get("name") == "Ascend Hardware"}
    roots = {"dsv4_hca_tnd::attention", "dsv4_hca_tnd::attention_long", "dsv4_hca_tnd::attention_tall"}
    windows = []
    for outer in service_profile["windows"]:
        begin, end = outer["cpu_marker_start_us"], outer["cpu_marker_end_us"]
        scoped = [e for e in events if e.get("ph") == "X" and begin <= float(e["ts"]) < end]
        calls = [e for e in scoped if e.get("name") in roots]
        if len(calls) != 1:
            raise ValueError("每窗必须有唯一完整PTO Torch根op")
        root_start = float(calls[0]["ts"])
        device = [e for e in scoped if e.get("pid") in device_pids and float(e.get("dur", 0)) > 0]
        before = [e for e in device if float(e["ts"]) < root_start]
        if (len(before) != 1 or before[0]["name"] != "CompressorMetadata"
                or float(before[0]["ts"]) + float(before[0]["dur"]) > root_start):
            raise ValueError("服务前导不是唯一且已完成的Native compact producer，不能自动划分")
        rows = [e for e in device if float(e["ts"]) >= root_start]
        task_types = {e.get("args", {}).get("Task Type") for e in rows}
        if not {"AI_CPU", "MIX_AIC"}.issubset(task_types):
            raise ValueError("缺少PTO AICPU/AICore完整记录")
        first = min(float(e["ts"]) for e in rows)
        last = max(float(e["ts"]) + float(e["dur"]) for e in rows)
        if last > end:
            raise ValueError("设备活动超出同步窗口")
        windows.append({"sample": outer["sample"], "device_span_us": last - first,
                        "cpu_root": calls[0]["name"], "cpu_root_start_us": root_start,
                        "device_start_us": first, "device_end_us": last,
                        "streams": sorted({str(e["tid"]) for e in rows}),
                        "task_types": sorted(task_types), "events": len(rows),
                        "excluded_service_prefix": [{"name": e["name"], "ts": float(e["ts"]),
                                                     "dur": float(e["dur"])} for e in before]})
    values = [w["device_span_us"] for w in windows]
    return {"samples_us": values, "median_us": statistics.median(values), "windows": windows,
            "trace": service_profile["trace"], "scope": "完整PTO根调用全部相关设备stream首尾",
            "service_scope_samples_us": service_profile["samples_us"],
            "service_scope_median_us": service_profile["median_us"],
            "aclgraph_replay": False}
