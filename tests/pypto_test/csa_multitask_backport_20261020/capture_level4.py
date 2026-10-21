# SPDX-License-Identifier: Apache-2.0
"""补采同一正式多task的一个L4窗口；诊断数据不进入L1性能表。"""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path


def export_window(source, destination, kernel_config):
    from simpler_setup.tools.swimlane_converter import read_perf_data

    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("window_*"))
    raw = destination / "chip_swimlane_records.json"
    joined = read_perf_data(raw)
    tasks = joined["tasks"]
    schedulers = joined.get("scheduler_records", [])
    orchestrators = joined.get("aicpu_orchestrator_phases", [])
    if joined["chip_swimlane_level"] != 4 or not tasks or not schedulers or not orchestrators:
        raise ValueError("缺少完整L4的AICore、Scheduler或Orchestrator记录")
    if len({task["core_id"] for task in tasks}) != 72:
        raise ValueError("L4窗口没有覆盖24 AIC与48 AIV")
    if any(task.get("dispatch_time_us", 0) <= 0 or task.get("finish_time_us", 0) <= 0 for task in tasks):
        raise ValueError("L4任务未成功关联调度时间戳")
    names = {}
    for node in ast.parse(Path(kernel_config).read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "KERNELS" for target in node.targets
        ):
            for item in node.value.elts:
                values = {ast.literal_eval(key): value for key, value in zip(item.keys, item.values)}
                names[str(ast.literal_eval(values["func_id"]))] = ast.literal_eval(values["name"])
    if not names:
        raise ValueError("当前正式CSA的kernel名称映射为空")
    name_map = destination / "name_map.json"
    name_map.write_text(json.dumps({"callable_id_to_name": names}, indent=2) + "\n")
    trace = destination / "swimlane.json"
    command = [
        sys.executable,
        "-m",
        "simpler_setup.tools.swimlane_converter",
        str(raw),
        "--func-names",
        str(name_map),
        "-o",
        str(trace),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    (destination / "converter.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    return {
        "level": 4,
        "core_records": len(tasks),
        "physical_cores": len({task["core_id"] for task in tasks}),
        "scheduler_streams": len(schedulers),
        "orchestrator_streams": len(orchestrators),
        "raw": str(raw),
        "trace": str(trace),
    }


def capture(*, entry, reset, verify, output, kernel_config):
    import pypto.torch
    import torch

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    windows, checks = [], {}
    with torch.npu.stream(stream):
        reset()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            entry()
        for _ in range(3):
            reset()
            graph.replay()
            stream.synchronize()
        checks["after_warmup"] = verify()
        for index in range(1):
            reset()
            stream.synchronize()
            pypto.torch.begin_dfx()
            try:
                graph.replay()
            finally:
                pypto.torch.end_dfx()
            stream.synchronize()
            source = output / "dfx"
            if index:
                source = source / f"window_{index}"
            window = export_window(source, output / f"l4_{index:02d}_formal", kernel_config)
            windows.append({"window": index, **window})
        checks["after_capture"] = verify()
    return {
        "scope": "正式多task，预热后一个L4单CSA窗口；有完整调度插桩，不作为L1基线或优化收益",
        "level": 4,
        "windows": windows,
        "graph_checks": checks,
    }
