# SPDX-License-Identifier: Apache-2.0
"""正式CSA与原多task的同进程PipeUtilization诊断，不采新的性能验收窗口。

复用完整驱动已建立的正式权重、Native输入、精确比较和保护检查，只替换
测试层的策略矩阵回调。捕获两个完整图，轮换六步得到每方三份PMU记录。
所有72个硬件角色参与原算子；PMU字段为工具的全核聚合，不能冒充逐核图。
Scalar与pipe占用可能重叠，不能相加成总时延；外层Duration不是L1验收数据。
"""

import csv
import json
from pathlib import Path
from statistics import median


def run_matrix(*, existing, baseline_entry, reset, verify, cache_guards,
               compile_options, output, update_stage, **unused):
    import torch
    import torch_npu
    from signal_checks import signal_padding_ok, signal_padding_status, signal_view

    destination = Path(output) / "pipe_profile"
    destination.mkdir(exist_ok=False)
    result = {
        "passed": False, "same_process": True, "same_root_addresses": True,
        "reference_source": "source_after_score_split/13259f92",
        "scope": "完整真实72角色持久核的外部PMU聚合，只用于诊断",
        "performance_eligible": False, "windows": [], "checks": {},
        "samples": [], "protocol": {}, "policy": existing.policy.constexprs(),
    }

    def save():
        (destination / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )

    def checked():
        numerical = verify()
        guards = cache_guards()
        assert all(x["status"] == "PASS" for x in guards.values())
        return {"numerical": numerical, "guards": guards}

    entries = {"dynamic_pto": baseline_entry,
               "spmd_current": lambda: existing(config=compile_options)}
    before = int(signal_view(existing)[0, 0, 0])
    spmd_calls = 0
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graphs = {}
    try:
        update_stage("hardware_pipe_graph_capture")
        with torch.npu.stream(stream):
            for name, entry in entries.items():
                reset()
                stream.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    entry()
                graphs[name] = graph
                reset()
                graph.replay()
                stream.synchronize()
                spmd_calls += int(name == "spmd_current")
                result["checks"][name + "_graph"] = checked()
            for iteration in range(3):
                order = list(entries) if iteration % 2 == 0 else list(reversed(entries))
                for name in order:
                    reset()
                    graph = graphs[name]
                    graph.replay()
                    stream.synchronize()
                    spmd_calls += int(name == "spmd_current")
            update_stage("hardware_pipe_sampling")
            order = ["dynamic_pto", "spmd_current", "spmd_current",
                     "dynamic_pto", "dynamic_pto", "spmd_current"]
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(destination)),
            ) as trace:
                for step, name in enumerate(order):
                    reset()
                    stream.synchronize()
                    with torch.profiler.record_function("CSA_PMU_" + name):
                        graphs[name].replay()
                        stream.synchronize()
                    spmd_calls += int(name == "spmd_current")
                    result["samples"].append({"step": step, "side": name})
                    trace.step()
            # 不在PMU采样窗内做D2H数值对照；补查两图，记录真实增加的代际。
            for name in entries:
                reset()
                graphs[name].replay()
                stream.synchronize()
                spmd_calls += int(name == "spmd_current")
                result["checks"][name + "_post_profile"] = checked()
        signal = signal_view(existing)
        protocol_ok = bool((signal[0, :, 0] == before + spmd_calls).all()
                           and signal_padding_ok(signal))
        result["protocol"]["spmd_current"] = {
            "passed": protocol_ok, "before": before, "calls": spmd_calls,
            "padding": signal_padding_status(signal),
        }
        assert protocol_ok
        files = list(destination.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
        assert len(files) == 1, files
        with files[0].open() as handle:
            rows = [r for r in csv.DictReader(handle) if r["Accelerator Core"] == "MIX_AIC"]
        assert len(rows) == 6, [(r["Step Id"], r["Name"]) for r in rows]
        rows_by_step = {int(r["Step Id"]): r for r in rows}
        assert set(rows_by_step) == set(range(6))
        by_side = {name: [] for name in entries}
        for item in result["samples"]:
            row = rows_by_step[item["step"]]
            assert int(row["Block Num"]) == 24 and int(row["Mix Block Num"]) == 48
            item["raw_row"] = row
            by_side[item["side"]].append(row)
        numeric = [key for key in rows[0]
                   if key.startswith(("aic_", "aiv_")) or key in ("Duration(us)", "cube_utilization(%)")]
        result["summary"] = {
            name: {key: {"samples": [float(row[key]) for row in side_rows],
                         "median": median(float(row[key]) for row in side_rows)}
                   for key in numeric}
            for name, side_rows in by_side.items()
        }
        result["kernel_details"] = str(files[0])
        result["trace"] = str(files[0].with_name("trace_view.json"))
        result["passed"] = True
        return result
    finally:
        save()
