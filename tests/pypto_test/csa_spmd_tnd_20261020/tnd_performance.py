# SPDX-License-Identifier: Apache-2.0
"""Paired complete L1 spans for genuine TND inputs; no fixed-S6 golden."""

import json
import statistics


def profile(
    *,
    candidate,
    baseline,
    fixture,
    output,
    expected,
    scores,
    options,
    configs,
    directory,
    windows,
    shared_workspace=False,
    padding_checks=None,
):
    import pypto.torch
    import torch
    from dsv4_csa_single_layer import collect_state, guard_checks, restore
    from dsv4_csa_validation import compare_tensor
    from l1_compare import export_window

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.runtime_abi import PHASE_COUNT

    original_buffers = baseline.buffers
    if shared_workspace:
        if candidate.policy.constexprs() != baseline.policy.constexprs():
            raise ValueError("共享工作区诊断要求相同资源策略")
        if {name: (value.shape, value.dtype, value.stride()) for name, value in candidate.buffers.items()} != {
            name: (value.shape, value.dtype, value.stride()) for name, value in baseline.buffers.items()
        }:
            raise ValueError("共享工作区诊断要求完全相同的Tensor ABI")
        # 仅用于串行交替的性能归因，不允许两个流并发执行。两侧数值验收此前已
        # 使用独立工作区完成；这里控制全部GM地址，包括signal及workspace。
        baseline.buffers = candidate.buffers
    entries = {"spmd_tnd": candidate, "spmd_original": baseline}
    graphs, checks, records = {}, {}, []
    before, calls = {}, dict.fromkeys(entries, 0)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())

    def verify(entry):
        actual = collect_state(fixture, output, candidate.native["idx_topk"])
        result = {name: compare_tensor(value, expected[name], 0, 0) for name, value in actual.items()}
        result["scores"] = compare_tensor(candidate.native["idx_topk_scores"], scores, 0, 0)
        result.update(guard_checks(fixture))
        if not all(value["status"] == "PASS" for value in result.values()):
            raise AssertionError("TND L1窗口前后出现数值或写界回归")
        if padding_checks is not None:
            padding = padding_checks(entry)
            if not all(padding.values()):
                raise AssertionError("TND L1窗口前后私有工作区padding被改写")
            result["workspace_padding"] = padding
        return result

    for name, entry in entries.items():
        before[name] = entry.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)[0, :, 0].clone()
    with torch.npu.stream(stream):
        for name, entry in entries.items():
            restore(fixture)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                entry(config=options)
            graphs[name] = graph
            restore(fixture)
            graph.replay()
            calls[name] += 1
            stream.synchronize()
            checks[name + "_graph"] = verify(entry)
        for _ in range(3):
            for name, graph in graphs.items():
                restore(fixture)
                graph.replay()
                calls[name] += 1
                stream.synchronize()
        index = 0
        for sample in range(windows):
            order = list(entries) if sample % 2 == 0 else list(reversed(entries))
            for name in order:
                restore(fixture)
                stream.synchronize()
                pypto.torch.begin_dfx()
                try:
                    graphs[name].replay()
                finally:
                    pypto.torch.end_dfx()
                calls[name] += 1
                source = directory / "dfx"
                if index:
                    source /= f"window_{index}"
                records.append(
                    export_window(
                        source,
                        directory / f"l1_{sample:02d}_{name}",
                        name,
                        kernel_config=configs[name],
                    )
                )
                index += 1
        # Preserve the final timed output check, then check both graph entries
        # again outside the complete sampling sequence. A large host comparison
        # between the final paired samples changes the device's idle interval.
        checks["last_timed_" + name] = verify(entries[name])
        for name, graph in graphs.items():
            restore(fixture)
            graph.replay()
            calls[name] += 1
            stream.synchronize()
            checks[name + "_post_l1"] = verify(entries[name])
    protocol = {}
    for name, entry in entries.items():
        signal = entry.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)
        count = sum(calls.values()) if shared_workspace else calls[name]
        protocol[name] = bool((signal[0, :, 0] == before[name] + count).all() and (signal[:, :, 1:] == 0).all())
    stats = {}
    for name in entries:
        samples = [r["incore_span_us"] for r in records if r["side"] == name]
        stats[name] = dict(samples_us=samples, median_us=statistics.median(samples))
    result = dict(
        passed=all(protocol.values()),
        windows=records,
        statistics_us=stats,
        scope="完整HC_pre/RMSNorm/CSA/HC_post L1首尾，包含等待与阶段间隙；恢复在计时窗口外",
        original_is_no_regression_reference=True,
        fixed_s6_multitask_used=False,
        shared_workspace=shared_workspace,
        checks=checks,
        protocol=protocol,
        verification_timing="Before/after sampling; both postlude replays are unmeasured and counted in protocol",
    )
    (directory / "paired_tnd_l1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    baseline.buffers = original_buffers
    return result


def service_cost(*, prepared, fixture, service_call, output, expected):
    """Separate event/wall measurements; these do not replace full incore L1."""
    import time

    import torch
    from dsv4_csa_single_layer import collect_state, restore
    from dsv4_csa_validation import compare_tensor

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        prepared()
    entries = {"direct_spmd_graph": graph.replay, "service_graph": service_call}
    samples = {name: [] for name in entries}
    for sample in range(13):
        for name in list(entries) if sample % 2 == 0 else list(reversed(entries)):
            restore(fixture)
            torch.npu.synchronize()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            begin = time.perf_counter_ns()
            entries[name]()
            submitted = time.perf_counter_ns()
            end.record()
            end.synchronize()
            done = time.perf_counter_ns()
            if sample >= 3:
                samples[name].append(
                    dict(
                        event_us=start.elapsed_time(end) * 1000,
                        host_submit_us=(submitted - begin) / 1000,
                        synchronous_wall_us=(done - begin) / 1000,
                    )
                )
    actual = collect_state(fixture, output, prepared.native["idx_topk"])
    checks = {name: compare_tensor(value, expected[name], 0, 0) for name, value in actual.items()}
    medians = {
        name: {metric: statistics.median(row[metric] for row in values) for metric in values[0]}
        for name, values in samples.items()
    }
    return dict(
        passed=all(value["status"] == "PASS" for value in checks.values()),
        scope="固定输入稳态图：服务含Native compact生产者及框架合同；NPU事件可能含Host提交空隙，不是L1",
        samples=samples,
        median_us=medians,
        checks=checks,
        service_minus_direct_us={
            metric: medians["service_graph"][metric] - medians["direct_spmd_graph"][metric]
            for metric in medians["service_graph"]
        },
    )
