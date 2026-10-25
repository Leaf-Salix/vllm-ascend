# SPDX-License-Identifier: Apache-2.0
"""包含内部初始化的完整L1边界；主MIX时间只作为诊断子项。"""

import json
from collections import defaultdict


def summarize(raw):
    frequency = raw["metadata"]["clock_freq_hz"]
    rows = raw["aicore_tasks"]
    if raw["chip_swimlane_level"] != 1 or not rows or frequency <= 0:
        raise ValueError("缺少有效L1记录")
    if any(len(r) != 6 or r[4] < r[3] or r[3] <= 0 for r in rows):
        raise ValueError("incore时间戳非法")
    tasks = defaultdict(list)
    for row in rows:
        tasks[row[1]].append(row)
    mix = [group for group in tasks.values() if len(group) == 72]
    init = [group for group in tasks.values() if len(group) == 48]
    if len(tasks) != 2 or len(mix) != 1 or len(init) != 1:
        raise ValueError(f"应有48AIV初始化和72核主MIX，实际分组：{[len(g) for g in tasks.values()]}")
    if len({r[0] for r in mix[0]}) != 72 or len({r[0] for r in init[0]}) != 48:
        raise ValueError("任务参与核不完整或重复")
    if not {r[0] for r in init[0]} <= {r[0] for r in mix[0]}:
        raise ValueError("初始化参与核不属于主MIX核集合")
    first = min(r[3] for r in rows)
    last = max(r[4] for r in rows)
    mix_first, mix_last = min(r[3] for r in mix[0]), max(r[4] for r in mix[0])
    init_first, init_last = min(r[3] for r in init[0]), max(r[4] for r in init[0])
    if init_last > mix_first:
        raise ValueError("主MIX在内部信号清零结束前启动")
    scale = 1e6 / frequency
    return {
        "incore_span_us": (last - first) * scale,
        "main_mix_us": (mix_last - mix_first) * scale,
        "initialization_us": (init_last - init_first) * scale,
        "initialization_to_mix_gap_us": (mix_first - init_last) * scale,
        "records": len(rows),
        "task_groups": len(tasks),
        "cores": len({r[0] for r in rows}),
        "task_roles": {str(key): len(value) for key, value in tasks.items()},
    }


def export_l1(source, destination, side, kernel_config):
    destination.mkdir()
    path = source / "chip_swimlane_records.json"
    raw = json.loads(path.read_text())
    result = summarize(raw)
    (destination / path.name).write_bytes(path.read_bytes())
    (destination / "kernel_config.py").write_bytes(kernel_config.read_bytes())
    if (source / "deps.json").exists():
        (destination / "deps.json").write_bytes((source / "deps.json").read_bytes())
    return {
        "side": side,
        **result,
        "raw": str(destination / path.name),
        "kernel_config": str(destination / "kernel_config.py"),
    }
