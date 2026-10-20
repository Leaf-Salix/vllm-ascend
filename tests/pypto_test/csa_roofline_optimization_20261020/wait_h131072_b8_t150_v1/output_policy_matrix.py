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
from signal_checks import signal_padding_ok, signal_padding_status, signal_view


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
    reference_source=None,
):
    # 设备依赖延迟导入，离线报告读取不初始化NPU。
    import pypto.torch
    import torch
    from compile_diagnostics import compilation_probe

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.runtime_abi import PHASE_COUNT, PHASE_RAW

    output = Path(output)
    if axis not in ("output", "hc_rows", "attention_query", "hc_consumer", "source"):
        raise ValueError("未知策略矩阵轴")
    matrix_dir = output / {
        "output": "output_policy_matrix",
        "hc_rows": "hc_row_policy_matrix",
        "attention_query": "attention_query_policy_matrix",
        "hc_consumer": "hc_consumer_policy_matrix",
        "source": "source_matrix",
    }[axis]
    matrix_dir.mkdir(exist_ok=False)
    report = dict(
        passed=False, matrix_axis=axis, same_process=True, same_root_addresses=True, policies={}, checks={}, windows=[]
    )

    def save():
        (matrix_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def name_for(policy):
        if axis == "source":
            return "spmd_candidate"
        if axis == "hc_consumer":
            return "spmd_hc_" + ("consumer_reduce" if policy.hc_consumer_reduce else "raw_stage")
        if axis == "attention_query":
            return "spmd_attention_query_" + ("plan" if policy.attention_query_metadata else "scan")
        if axis == "hc_rows":
            return "spmd_hc_" + ("row_dependencies" if policy.hc_row_dependencies else "global_dependencies")
        return f"spmd_n{policy.oa_cols}_" + ("grouped" if policy.output_group_pipeline else "global")

    def tables():
        return set(Path(os.environ["PYPTO_PROG_BUILD_DIR"]).glob("*decode_csa_spmd_*/kernel_config.py"))

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
        if axis == "source":
            from source_loader import load_binding

            if reference_source is None:
                raise ValueError("源码矩阵必须显式提供基线包目录")
            reference = load_binding(reference_source, "reference")
            assert reference.phase_count == PHASE_COUNT, "源码矩阵必须保留原60-phase语义前缀"
            reference_policy = reference.select_policy(
                tokens=existing.tokens, requests=existing.requests, max_query_tokens=6, max_seq_len=max_seq_len)
            reference_dir = matrix_dir / "reference_source"
            reference_dir.mkdir()
            for path in reference.source.glob("*.py"):
                (reference_dir / path.name).write_text(path.read_text())
            report["reference_source"] = str(reference.source)
            plans = [("spmd_candidate", type(existing), existing.policy),
                     ("spmd_reference", reference.prepared, reference_policy)]
        elif axis == "hc_consumer":
            # 保持行依赖、tile与全部其它阶段不变，只比较RAW独立交接/消费者归约。
            policies = [replace(existing.policy, hc_consumer_reduce=enabled) for enabled in (False, True)]
        elif axis == "attention_query":
            # 只比较复用plan与重复扫描Q/L，不混入HC、OA或其它数学策略。
            policies = [replace(existing.policy, attention_query_metadata=enabled) for enabled in (False, True)]
        elif axis == "hc_rows":
            # 只开关HC行依赖，OA与其它工作量策略完全相同，不混入tile候选。
            policies = [replace(existing.policy, hc_row_dependencies=enabled) for enabled in (False, True)]
        else:
            # 小中T只比较输出流水；大T另核对N128/N256，不扩成无依据笛卡尔积。
            cols_set = (128, 256) if existing.buffers["x"].shape[0] > 96 else (existing.policy.oa_cols,)
            policies = [
                replace(existing.policy, oa_cols=cols, output_group_pipeline=grouped)
                for cols in cols_set for grouped in (False, True)
            ]
        if axis != "source":
            plans = [(name_for(policy), type(existing), policy) for policy in policies]
        for name, prepared_type, policy in plans:
            report["policies"][name] = policy.constexprs()
            if name in candidates:
                continue
            update_stage("output_matrix_compile_" + name)
            candidate = prepared_type(
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
            signal = signal_view(candidate)
            assert bool((signal[0, :, 0] == 1).all() and signal_padding_ok(signal))
            if axis == "hc_consumer" and policy.hc_consumer_reduce:
                # RAW保持填毒且没有RAW通知，直接消费路径不得隐式依赖旧中间量。
                untouched = bool((candidate.buffers["raw"].cpu() == 37).all())
                no_publish = bool((signal[PHASE_RAW, :, 0] == 0).all())
                report.setdefault("raw_elision", {})[name] = dict(
                    raw_stays_poisoned=untouched, no_raw_phase_publish=no_publish)
                assert untouched and no_publish
            new_tables = tables() - previous_tables
            assert len(new_tables) == 1, (name, new_tables)
            configs[name] = next(iter(new_tables))
            candidates[name] = candidate
            save()
        entries = {"dynamic_pto": baseline_entry}
        entries.update({name: lambda item=item: item(config=compile_options) for name, item in candidates.items()})
        configs["dynamic_pto"] = next(
            Path(os.environ["PYPTO_PROG_BUILD_DIR"]).glob("*decode_csa_tp1_layer_*/kernel_config.py")
        )
        report["kernel_configs"] = {name: str(config) for name, config in configs.items()}
        before = {}
        for name, item in candidates.items():
            signal = signal_view(item)[0, :, 0]
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
        if axis == "source":
            reference_us = report["statistics_us"]["spmd_reference"]["median_us"]
            candidate_us = report["statistics_us"]["spmd_candidate"]["median_us"]
            report["candidate_vs_reference"] = dict(
                reference_us=reference_us, candidate_us=candidate_us,
                latency_reduction_pct=(1 - candidate_us / reference_us) * 100)
        report["protocol"] = {}
        for name, item in candidates.items():
            signal = signal_view(item)
            passed = bool((signal[0, :, 0] == before[name] + calls[name]).all() and signal_padding_ok(signal))
            report["protocol"][name] = dict(
                passed=passed, before=before[name], calls=calls[name],
                phase_count=signal.shape[0], padding=signal_padding_status(signal))
            assert passed, name
        report["passed"] = True
        return report
    finally:
        save()
