# SPDX-License-Identifier: Apache-2.0
"""同一MPMD图另采全设备跨度；不用Event或L1加常数替代AICPU记录。"""

import csv
from collections import Counter


def profile_mpmd(fixture, graph, output, directory):
    import torch
    import torch_npu
    from csa_measure import layer_device_interval

    initial = {name: group["initial"].to(group["allocation"].device) for name, group in fixture["groups"].items()}
    torch.npu.synchronize()
    profile_dir = directory / "profile_mpmd"
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
    ) as profiler:
        for _ in range(30):
            for name, group in fixture["groups"].items():
                group["allocation"].copy_(initial[name])
            output.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
        profiler.step()
    result = layer_device_interval(profile_dir)
    with open(result["raw_csv"]) as stream:
        types = Counter(row["Type"] for row in csv.DictReader(stream))
    if not any("aicpu" in name.lower() for name in types) or not any("aicore" in name.lower() for name in types):
        raise ValueError("外层记录缺少AICPU或AICore，不能填入含调度列")
    result["task_types"] = dict(types)
    result["paired_l1_note"] = "同一进程、同一图和输入，先采10次L1，再独立采30次Profiler；避免DFX导出与Profiler互相污染"
    return result
