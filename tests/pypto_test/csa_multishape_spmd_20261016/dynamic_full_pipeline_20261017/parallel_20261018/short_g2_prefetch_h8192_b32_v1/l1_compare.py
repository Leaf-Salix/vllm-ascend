# SPDX-License-Identifier: Apache-2.0
"""完整动态CSA的配对L1计时；沿用历史六列原始记录的incore起止口径。

两侧先各自捕获并检查图重放，再交替AB/BA采样。每个窗口只有一次完整CSA，
fixture恢复在窗口外；计时包含内部等待、公共metadata准备及阶段空隙。
本文件只采集和汇总，不生成数学源码、不改编译器、不运行Native attention。
"""

import ast
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def export_window(source, destination, side, *, kernel_config=None):
    """记录第3/4下标是incore开始/结束，不能使用派发或收任务时刻。"""
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("window_*"))
    raw = json.loads((destination / "chip_swimlane_records.json").read_text())
    frequency = raw["metadata"]["clock_freq_hz"]
    rows = raw["aicore_tasks"]
    if raw["chip_swimlane_level"] != 1 or not rows or frequency <= 0:
        raise ValueError("缺少有效L1核内时间记录")
    if any(len(row) != 6 or row[4] < row[3] or row[3] <= 0 for row in rows):
        raise ValueError("原始泳道时间戳异常")
    if len({row[0] for row in rows}) != 72:
        raise ValueError("泳道必须覆盖24 AIC与48 AIV")
    is_spmd = side == "single_spmd" or side.startswith("spmd_")
    if is_spmd and (len(rows) != 72 or len({row[1] for row in rows}) != 1):
        raise ValueError("候选没有保持一个完整MIX任务")
    names = {}
    # 独立边界probe也会生成kernel_config，其func_id从0重新编号，不能混入
    # CSA名称表覆盖基线。每侧只接受自己的唯一编译入口；原始时间戳不受命名影响。
    prefix = "decode_csa_spmd_" if is_spmd else "_decode_csa_tp1_layer_"
    tables = (
        [Path(kernel_config)]
        if kernel_config is not None
        else list(Path(os.environ["PYPTO_PROG_BUILD_DIR"]).glob(prefix + "*/kernel_config.py"))
    )
    if len(tables) != 1:
        raise ValueError(f"{side}应有一个独立kernel配置，实际{len(tables)}个")
    for table in tables:
        for node in ast.parse(table.read_text()).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "KERNELS" for t in node.targets
            ):
                for item in node.value.elts:
                    value = {ast.literal_eval(k): v for k, v in zip(item.keys, item.values)}
                    names[str(ast.literal_eval(value["func_id"]))] = ast.literal_eval(value["name"])
        shutil.copy2(table, destination / (table.parent.name + "_kernel_config.py"))
    (destination / "name_map.json").write_text(json.dumps({"callable_id_to_name": names}, indent=2) + "\n")
    command = [
        sys.executable,
        "-m",
        "simpler_setup.tools.swimlane_converter",
        str(destination / "chip_swimlane_records.json"),
        "--func-names",
        str(destination / "name_map.json"),
        "-o",
        str(destination / "swimlane.json"),
    ]
    converted = subprocess.run(command, text=True, capture_output=True)
    (destination / "converter.log").write_text(converted.stdout + converted.stderr)
    converted.check_returncode()
    return {
        "side": side,
        "incore_span_us": (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / frequency,
        "clock_freq_hz": frequency,
        "core_records": len(rows),
        "task_count": len({r[1] for r in rows}),
        "cores": len({r[0] for r in rows}),
        "raw": str(destination / "chip_swimlane_records.json"),
        "trace": str(destination / "swimlane.json"),
    }


def profile_pair(*, entries, reset, verify, output, windows, warmup=3):
    """返回逐次原始证据与中位数；失败不产生达标结论。"""
    import pypto.torch
    import torch

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graphs, replay_checks, records = {}, {}, []
    spmd_calls = 0
    with torch.npu.stream(stream):
        for side, entry in entries.items():
            reset()
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                entry()
            graphs[side] = graph
            reset()
            graph.replay()
            stream.synchronize()
            spmd_calls += side == "single_spmd"
            replay_checks[side] = verify()
        for sample in range(warmup):
            order = list(entries) if sample % 2 == 0 else list(reversed(entries))
            for side in order:
                reset()
                graphs[side].replay()
                stream.synchronize()
                spmd_calls += side == "single_spmd"
        window_index = 0
        for sample in range(windows):
            order = list(entries) if sample % 2 == 0 else list(reversed(entries))
            for side in order:
                reset()
                stream.synchronize()
                pypto.torch.begin_dfx()
                try:
                    graphs[side].replay()
                finally:
                    pypto.torch.end_dfx()
                spmd_calls += side == "single_spmd"
                source = output / "dfx"
                if window_index:
                    source = source / f"window_{window_index}"
                records.append(export_window(source, output / f"l1_{sample:02d}_{side}", side))
                window_index += 1
                if sample == windows - 1:
                    replay_checks[side + "_post_l1"] = verify()
    samples = {side: [r["incore_span_us"] for r in records if r["side"] == side] for side in entries}
    statistics_us = {
        side: {"samples_us": v, "median_us": statistics.median(v), "min_us": min(v), "max_us": max(v)}
        for side, v in samples.items()
    }
    baseline = statistics_us["dynamic_pto"]["median_us"]
    candidate = statistics_us["single_spmd"]["median_us"]
    return {
        "scope": "完整L1 incore最早开始至最晚结束，包含内部等待和间隙；恢复输入在窗口外",
        "graph_checks": replay_checks,
        "windows": records,
        "statistics_us": statistics_us,
        "dynamic_pto_us": baseline,
        "single_spmd_us": candidate,
        "latency_reduction_pct": (1 - candidate / baseline) * 100,
        "target_reduction_pct": 30,
        "target_met": candidate <= baseline * 0.7,
        "spmd_calls": spmd_calls,
    }
