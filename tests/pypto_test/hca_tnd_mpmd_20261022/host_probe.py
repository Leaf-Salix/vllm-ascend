# SPDX-License-Identifier: Apache-2.0
"""对已经预热的真实调用做Host归因；不改SDK、调用路径、精度或设备数学。"""

import cProfile
import json
import pstats
import statistics
import time
from pathlib import Path


def prepared_operator(kernel, schema):
    """只作诊断：复用已预热的SDK registration，仍走SDK校验及异步生命周期。

    这些是SDK内部接口，不在正式适配层使用。测试期间源码/constexpr/config不变，
    不把静态绑定视为支持自动感知Python源码或常量热更新的新公共契约。
    """
    from pypto.runtime.kernel.context import get_process_kernel_state
    from pypto.torch.launch import enqueue

    state = get_process_kernel_state()
    bound = state.require_config()
    # 只在当前单进程、预热完成后读取；不得缓存裸地址或其他Worker的registration。
    found = [(key, state.find_specialization(key, bound)) for key in state._specializations if key[0] is kernel]
    if len(found) != 1 or found[0][1] is None:
        raise RuntimeError(f"诊断要求当前root唯一已预热specialization，实际{len(found)}")
    registration = found[0][1]

    class PreparedOperator:
        _schema = schema

        def __call__(self, *args):
            return enqueue(registration, args)

    return PreparedOperator()


def profile_host(calls, reset, *, output_dir, samples=7):
    import torch

    report = {"scope": "无PyTorch profiler的直接调用Host耗时，cProfile仅用于函数归因",
              "aclgraph_replay": False, "sdk_modified": False, "sides": {}}
    for side, run in calls.items():
        for _ in range(2):
            reset()
            run()
            torch.npu.synchronize()
        durations = []
        for _ in range(samples):
            reset()
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            run()
            submitted = time.perf_counter_ns()
            torch.npu.synchronize()
            finished = time.perf_counter_ns()
            durations.append({"enqueue_us": (submitted - start) / 1000,
                              "including_completion_us": (finished - start) / 1000})
        profile = cProfile.Profile()
        for _ in range(3):
            reset()
            torch.npu.synchronize()
            profile.runcall(run)
            torch.npu.synchronize()
        rows = []
        for (file, line, name), (primitive, total, own, cumulative, _) in pstats.Stats(profile).stats.items():
            rows.append({"file": file, "line": line, "function": name, "primitive_calls": primitive,
                         "total_calls": total, "own_us": own * 1e6, "cumulative_us": cumulative * 1e6})
        rows.sort(key=lambda value: value["cumulative_us"], reverse=True)
        report["sides"][side] = {
            "samples": durations,
            "enqueue_median_us": statistics.median(v["enqueue_us"] for v in durations),
            "completion_median_us": statistics.median(v["including_completion_us"] for v in durations),
            "cprofile_calls": 3, "cprofile_functions": rows,
        }
    path = Path(output_dir) / "host_profile.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
