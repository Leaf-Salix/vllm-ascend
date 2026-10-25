# SPDX-License-Identifier: Apache-2.0
"""显式区分外部清零task与核内清零；两种都覆盖调用的全部核内时间。"""

import json
import math
from collections import Counter

from complete_l1 import summarize as separate_summary


def summarize(raw, *, init_layout="separate"):
    metadata = raw["metadata"]
    frequency = metadata["clock_freq_hz"]
    rows = raw["aicore_tasks"]
    if isinstance(frequency, bool) or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("无效L1频率")
    boundaries = metadata.get("run_boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != 1 or metadata.get("dropped_run_boundaries") != 0:
        raise ValueError("窗口必须完整对应一次根调用")
    boundary = boundaries[0]
    if not isinstance(boundary, list) or len(boundary) != 2 or any(type(v) is not int or v < 0 for v in boundary):
        raise ValueError("根调用边界格式非法")
    if not rows or any(
        not isinstance(r, list)
        or len(r) != 6
        or any(type(v) is not int or v < 0 for v in r)
        or r[3] <= 0
        or r[4] < r[3]
        for r in rows
    ):
        raise ValueError("incore记录必须包含有限、非负的整型时钟及标识")
    if boundary[1] > min(r[3] for r in rows):
        raise ValueError("incore记录早于本窗口根调用，不能混入旧样本")
    if metadata["num_cores"] != 72 or Counter(metadata["core_types"]) != {"aic": 24, "aiv": 48}:
        raise ValueError("当前CSA计时要求设备实际全占用24AIC+48AIV")
    if any(r[0] not in range(72) for r in rows):
        raise ValueError("记录的核编号超出实际设备范围")
    if init_layout == "separate":
        result = separate_summary(raw)
        init_task = next(k for k, count in result["task_roles"].items() if count == 48)
        if any(metadata["core_types"][r[0]] != "aiv" for r in rows if str(r[1]) == init_task):
            raise ValueError("独立清零必须由全部48个AIV完成")
        return {**result, "init_layout": "separate"}
    if init_layout != "fused":
        raise ValueError("初始化布局必须显式声明为separate或fused")
    if raw["chip_swimlane_level"] != 1 or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("无效L1频率或采集层级")
    if len(rows) != 72 or {r[0] for r in rows} != set(range(72)) or len({r[1] for r in rows}) != 1:
        raise ValueError("融合候选必须有唯一完整72角色MIX任务，不得丢弃额外task")
    begin, end = min(r[3] for r in rows), max(r[4] for r in rows)
    span = (end - begin) * 1e6 / frequency
    return {
        "incore_span_us": span,
        "main_mix_us": span,
        "initialization_us": None,
        "initialization_to_mix_gap_us": None,
        "init_layout": "fused",
        "scope": "从MIX最早进入到最晚退出，包含核内清零、硬屏障及完整CSA",
        "records": 72,
        "task_groups": 1,
        "cores": 72,
        "task_roles": {str(rows[0][1]): 72},
    }


def export_l1(source, destination, side, kernel_config, *, init_layout="separate"):
    if side == "reference" and init_layout != "separate":
        raise ValueError("当前冻结对照始终保留独立清零，不能改其边界")
    destination.mkdir()
    path = source / "chip_swimlane_records.json"
    result = summarize(json.loads(path.read_text()), init_layout=init_layout)
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
