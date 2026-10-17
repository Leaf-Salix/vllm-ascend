# SPDX-License-Identifier: Apache-2.0
"""同卡同进程按单一轴核对输出策略或HC行依赖，不从跨轮历史差值推断收益。

正式源码通过policy选择N128/N256和组通知开关；每候选独立工作区/信号，
共享权重、Native输入输出/cache地址。先逐候选精确对照，再捕获并检查图，
最后轮换各候选的先后顺序，采完整L1首尾。只用PTO，不运行Native forward。
"""

import contextlib
import json
import os
import statistics
from dataclasses import replace
from pathlib import Path

from l1_compare import export_window


def run_matrix(
    *,
    existing,
    max_seq_len,
    allocate,
    baseline_entry,
    reset,
    verify,
    cache_guards,
    compile_options,
    output,
    windows,
    wait_backoff,
    update_stage,
    axis="output",
):
    # 设备依赖延迟导入，离线报告读取不初始化NPU。
    import pypto.torch
    import torch
    from compile_diagnostics import compilation_probe

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.binding import PreparedDynamicCSA
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.runtime_abi import PHASE_COUNT

    output = Path(output)
    if axis not in ("output", "hc_rows"):
        raise ValueError("未知策略矩阵轴")
    matrix_dir = output / ("hc_row_policy_matrix" if axis == "hc_rows" else "output_policy_matrix")
    matrix_dir.mkdir(exist_ok=False)
    report = dict(
        passed=False, matrix_axis=axis, same_process=True, same_root_addresses=True, policies={}, checks={}, windows=[]
    )

    def save():
        (matrix_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def name_for(policy):
        if axis == "hc_rows":
            return "spmd_hc_" + ("row_dependencies" if policy.hc_row_dependencies else "global_dependencies")
        return f"spmd_n{policy.oa_cols}_" + ("grouped" if policy.output_group_pipeline else "global")

    def tables():
        return set(Path(os.environ["PYPTO_PROG_BUILD_DIR"]).glob("decode_csa_spmd_*/kernel_config.py"))

    def checked():
        numerical = verify()
        guards = cache_guards()
        assert all(c["status"] == "PASS" for c in guards.values())
        return dict(numerical=numerical, native_guards=guards)

    candidates = {name_for(existing.policy): existing}
    initial_tables = tables()
    assert len(initial_tables) == 1
    configs = {name_for(existing.policy): next(iter(initial_tables))}
    try:
        if axis == "hc_rows":
            # 只开关HC行依赖，OA与其它工作量策略完全相同，不混入tile候选。
            policies = [replace(existing.policy, hc_row_dependencies=enabled) for enabled in (False, True)]
        else:
            # 小中T只比较输出流水；大T另核对N128/N256，不扩成无依据笛卡尔积。
            cols_set = (128, 256) if existing.buffers["x"].shape[0] > 96 else (existing.policy.oa_cols,)
            policies = [
                replace(existing.policy, oa_cols=cols, output_group_pipeline=grouped)
                for cols in cols_set for grouped in (False, True)
            ]
        for policy in policies:
            name = name_for(policy)
            report["policies"][name] = policy.constexprs()
            if name in candidates:
                continue
            update_stage("output_matrix_compile_" + name)
            candidate = PreparedDynamicCSA(
                existing.native,
                max_query_tokens=6,
                max_seq_len=max_seq_len,
                policy=policy,
                allocator=lambda key, shape, dtype, device, prefix=name: allocate(
                    prefix + "/" + key, shape, dtype, device
                ),
            )
            for key, value in existing.native.items():
                if isinstance(value, torch.Tensor):
                    assert candidate.native[key].data_ptr() == value.data_ptr(), (name, key)
            for key, value in candidate.allocations.items():
                if key not in ("signal", "ffts"):
                    value.fill_(37)
            previous_tables = tables()
            reset()
            probe = (
                compilation_probe(matrix_dir / (name + "_codegen"), wait_backoff=True)
                if wait_backoff
                else contextlib.nullcontext()
            )
            with probe:
                candidate(config=compile_options)
            torch.npu.synchronize()
            report["checks"][name + "_eager"] = checked()
            signal = candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)
            assert bool((signal[0, :, 0] == 1).all() and (signal[:, :, 1:] == 0).all())
            new_tables = tables() - previous_tables
            assert len(new_tables) == 1, (name, new_tables)
            configs[name] = next(iter(new_tables))
            candidates[name] = candidate
            save()
        entries = {"dynamic_pto": baseline_entry}
        entries.update({name: lambda item=item: item(config=compile_options) for name, item in candidates.items()})
        configs["dynamic_pto"] = next(
            Path(os.environ["PYPTO_PROG_BUILD_DIR"]).glob("_decode_csa_tp1_layer_*/kernel_config.py")
        )
        report["kernel_configs"] = {name: str(config) for name, config in configs.items()}
        before = {}
        for name, item in candidates.items():
            signal = item.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)[0, :, 0]
            assert bool((signal == signal[0]).all())
            before[name] = int(signal[0])
        calls = dict.fromkeys(candidates, 0)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graphs = {}
        update_stage("output_matrix_graphs")
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
                if name in calls:
                    calls[name] += 1
                report["checks"][name + "_graph"] = checked()
            sides = list(entries)
            for warmup in range(3):
                for name in sides if warmup % 2 == 0 else list(reversed(sides)):
                    reset()
                    graphs[name].replay()
                    stream.synchronize()
                    if name in calls:
                        calls[name] += 1
            update_stage("output_matrix_l1")
            window_index = 0
            for sample in range(windows):
                order = sides[sample % len(sides) :] + sides[: sample % len(sides)]
                if sample % 2:
                    order.reverse()
                for name in order:
                    reset()
                    stream.synchronize()
                    pypto.torch.begin_dfx()
                    try:
                        graphs[name].replay()
                    finally:
                        pypto.torch.end_dfx()
                    if name in calls:
                        calls[name] += 1
                    source = output / "dfx"
                    if window_index:
                        source = source / f"window_{window_index}"
                    report["windows"].append(
                        export_window(source, matrix_dir / f"l1_{sample:02d}_{name}", name, kernel_config=configs[name])
                    )
                    window_index += 1
                    if sample == windows - 1:
                        report["checks"][name + "_post_l1"] = checked()
                save()
        report["statistics_us"] = {}
        for name in entries:
            samples = [w["incore_span_us"] for w in report["windows"] if w["side"] == name]
            report["statistics_us"][name] = dict(samples_us=samples, median_us=statistics.median(samples))
        baseline = report["statistics_us"]["dynamic_pto"]["median_us"]
        report["comparison"] = {
            name: dict(
                latency_reduction_pct=(1 - report["statistics_us"][name]["median_us"] / baseline) * 100,
                target_met=report["statistics_us"][name]["median_us"] <= baseline * 0.7,
            )
            for name in candidates
        }
        report["protocol"] = {}
        for name, item in candidates.items():
            signal = item.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)
            passed = bool((signal[0, :, 0] == before[name] + calls[name]).all() and (signal[:, :, 1:] == 0).all())
            report["protocol"][name] = dict(passed=passed, before=before[name], calls=calls[name])
            assert passed, name
        report["passed"] = True
        return report
    finally:
        save()
