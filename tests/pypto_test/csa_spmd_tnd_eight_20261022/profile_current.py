# SPDX-License-Identifier: Apache-2.0
"""Hardware PipeUtilization for one validated complete SPMD graph, not L1 timing."""

import csv
import json
import math
import statistics
from pathlib import Path

REQUIRED = (
    "aic_mac_time(us)",
    "aic_scalar_time(us)",
    "aic_mte1_time(us)",
    "aic_mte2_time(us)",
    "aic_fixpipe_time(us)",
    "aiv_vec_time(us)",
    "aiv_scalar_time(us)",
)


def summarize_csv(path):
    with Path(path).open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["Accelerator Core"] == "MIX_AIC"]
    if len(rows) != 3:
        raise ValueError("Expected exactly one complete MIX kernel in each of three profiler steps")
    if "Step Id" in rows[0]:
        if {int(row["Step Id"]) for row in rows} != {0, 1, 2}:
            raise ValueError("Profiler step IDs are incomplete or duplicated")
        rows.sort(key=lambda row: int(row["Step Id"]))
        sample_binding = "Explicit profiler Step Id column"
    else:
        trace_path = Path(path).with_name("trace_view.json")
        if not trace_path.exists():
            raise ValueError("No Step Id: need trace markers to bind the three graph replays")
        trace = json.loads(trace_path.read_text())
        events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
        markers = sorted((e for e in events if e.get("name") == "CSA_TND_CURRENT_PMU"), key=lambda e: float(e["ts"]))
        if len(markers) != 3:
            raise ValueError("Need exactly three explicit graph replay markers")
        rows.sort(key=lambda row: float(row["Start Time(us)"].strip()))
        for index, (row, marker) in enumerate(zip(rows, markers)):
            start = float(row["Start Time(us)"].strip())
            end = start + float(row["Duration(us)"])
            begin = float(marker["ts"])
            finish = begin + float(marker["dur"])
            if not begin <= start <= end <= finish:
                raise ValueError("MIX kernel is outside its explicit graph replay marker")
            if index and begin <= float(markers[index - 1]["ts"]) + float(markers[index - 1]["dur"]):
                raise ValueError("Graph replay markers overlap")
        sample_binding = "Chronological MIX records contained one-to-one in CSA_TND_CURRENT_PMU trace markers"
    if any(int(row["Block Num"]) != 24 or int(row["Mix Block Num"]) != 48 for row in rows):
        raise ValueError("Missing full 24-AIC/48-AIV participation")
    keys = [
        key for key in rows[0] if key.startswith(("aic_", "aiv_")) or key in ("Duration(us)", "cube_utilization(%)")
    ]
    if not set(REQUIRED) <= set(keys):
        raise ValueError("Required hardware pipe counters are missing")
    summary = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Invalid counter values: {key}")
        summary[key] = dict(samples=values, median=statistics.median(values))
    for key in ("aic_mac_time(us)", "aiv_vec_time(us)", "aic_mte2_time(us)"):
        if any(value <= 0 for value in summary[key]["samples"]):
            raise ValueError(f"Expected active hardware counter: {key}")
    return dict(raw_rows=rows, summary=summary, kernel_details=str(path), sample_binding=sample_binding)


def collect(*, graph, entry, reset, stream, output):
    import torch
    import torch_npu

    destination = Path(output) / "pipe_profile"
    destination.mkdir(exist_ok=False)
    before = entry.buffers["signal"].cpu().view(-1, 72, 128)
    assert not before[:, :, 1:].count_nonzero()
    assert bool((before[0, :, 0] == before[0, 0, 0]).all())
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(destination)),
    ) as profiler:
        for _ in range(3):
            reset()
            stream.synchronize()
            with torch.profiler.record_function("CSA_TND_CURRENT_PMU"):
                graph.replay()
                stream.synchronize()
            profiler.step()
    after = entry.buffers["signal"].cpu().view(-1, 72, 128)
    assert not after[:, :, 1:].count_nonzero()
    assert bool((after[0, :, 0] == before[0, :, 0] + 3).all())
    paths = list(destination.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    assert len(paths) == 1, paths
    return dict(
        capture_passed=True,
        performance_eligible=False,
        scope="Three aggregate hardware-PMU records for the complete 24-AIC/48-AIV operator",
        signal_generation_before=int(before[0, 0, 0]),
        signal_generation_after=int(after[0, 0, 0]),
        signal_padding_passed=True,
        kernel_details=str(paths[0]),
        trace=str(paths[0].with_name("trace_view.json")),
        limitations=[
            "Pipe times overlap and cannot be added or treated as removable time",
            "Scalar activity may include spin waits; cache miss ratios do not measure stall time",
            "Whole-kernel aggregates do not identify individual phase or core utilization",
            "Profiler Duration includes persistent-kernel overhead and is not complete incore L1",
        ],
    )
