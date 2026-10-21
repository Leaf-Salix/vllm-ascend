# SPDX-License-Identifier: Apache-2.0
"""真实单层服务入口诊断；使用正式模型准备钩子和Torch custom-op，未模拟设备提交。"""

from types import SimpleNamespace


def prepare(layer):
    from vllm_ascend.models.pypto_deepseek_v4 import install_csa_forward, prepare_csa_model
    from vllm_ascend.ops.pypto.variant import selected_variant

    if selected_variant() != "spmd_tnd":
        raise ValueError("服务诊断须明确选择PTO_CSA_VARIANT=spmd_tnd")
    install_csa_forward(layer)
    prepare_csa_model(SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    return layer.self_attn.dsa_attn._pto_csa_runtime


def validate(
    runtime,
    layer,
    config,
    fixture,
    output,
    expected,
    directory,
    *,
    wait_backoff,
    framework_graph=False,
    metadata_replay=False,
    baseline=None,
    options=None,
    timing=False,
    timing_only=False,
):
    """比较服务实际提交与同一SPMD算子、固定地址图重放、空调用及范围外回退。"""
    import torch
    from compile_diagnostics import compilation_probe
    from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, restore
    from dsv4_csa_validation import compare_tensor
    from vllm.forward_context import get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.dsv4_csa import _native_attention_half

    wrapped = None
    descriptor = None
    if framework_graph:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor

        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        def forward(hidden, positions, target):
            torch.ops.vllm.dsv4_csa_forward(
                hidden,
                positions,
                target,
                layer.self_attn.dsa_attn.prefix,
            )
            return target

        descriptor = BatchDescriptor(
            fixture["tokens"],
            num_reqs=fixture["request_capacity"],
            uniform=False,
        )
        wrapped = ACLGraphWrapper(forward, config, CUDAGraphMode.FULL)

    def call(case, *, model_graph=False):
        extra = {}
        if model_graph:
            extra = dict(aclgraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=descriptor)
        with set_ascend_forward_context(
            case["metadata"],
            config,
            num_tokens=case["tokens"],
            num_actual_tokens=next(iter(case["metadata"].values())).num_actual_tokens,
            **extra,
        ):
            context = get_forward_context()
            eligible = runtime.eligible(context, case["hidden"], case["positions"])
            if model_graph:
                wrapped(case["hidden"], case["positions"], output)
            else:
                torch.ops.vllm.dsv4_csa_forward(
                    case["hidden"],
                    case["positions"],
                    output,
                    layer.self_attn.dsa_attn.prefix,
                )
            return eligible, context

    before = runtime.submissions
    restore(fixture)
    with compilation_probe(directory / "service_compile", wait_backoff=wait_backoff):
        eligible, _ = call(fixture)
    torch.npu.synchronize()
    if not eligible or runtime.submissions != before + 1:
        raise AssertionError("支持范围内的实际Torch custom-op调用没有命中TND单SPMD")
    prepared = runtime.last_binding
    actual = collect_state(fixture, output, prepared.native["idx_topk"])
    comparisons = {name: compare_tensor(value, expected[name], 0, 0) for name, value in actual.items()}
    guards = guard_checks(fixture)
    workspace_addresses = {name: value.data_ptr() for name, value in prepared.allocations.items()}

    restore(fixture)
    if framework_graph:
        _, captured_context = call(fixture, model_graph=True)
        contracts = wrapped.concrete_aclgraph_entries[descriptor].csa_tnd_contracts
    else:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            _, captured_context = call(fixture)
        contracts = captured_context.additional_kwargs.get("pto_csa_tnd_graph_contracts", {})
    capture_submissions = runtime.submissions
    addresses = {name: value.data_ptr() for name, value in prepared.buffers.items()}
    workspace_reused = workspace_addresses == {name: value.data_ptr() for name, value in prepared.allocations.items()}
    replays = []
    for _ in range(0 if timing_only else 2):
        restore(fixture)
        with set_ascend_forward_context(
            fixture["metadata"],
            config,
            num_tokens=fixture["tokens"],
            num_actual_tokens=fixture.get("actual_tokens", fixture["tokens"]),
        ):
            context = get_forward_context()
            allowed = bool(contracts) and all(value.allows_replay(context) for value in contracts.values())
        if not allowed:
            raise AssertionError("同址服务图合同失配")
        if framework_graph:
            call(fixture, model_graph=True)
        else:
            graph.replay()
        torch.npu.synchronize()
        replay = collect_state(fixture, output, prepared.native["idx_topk"])
        checks = {name: compare_tensor(value, actual[name], 0, 0) for name, value in replay.items()}
        replays.append(
            {
                "checks": checks,
                "python_not_resubmitted": runtime.submissions == capture_submissions,
                "same_addresses": addresses == {name: value.data_ptr() for name, value in prepared.buffers.items()},
                "guards": guard_checks(fixture),
            }
        )

    timing_result = None
    if timing:
        from tnd_performance import service_cost

        # Both sides start with the same existing forward context. Exclude the
        # test harness's context setup and extra eligibility probe from timing.
        with set_ascend_forward_context(
            fixture["metadata"],
            config,
            num_tokens=fixture["tokens"],
            num_actual_tokens=fixture["actual_tokens"],
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=descriptor,
        ):
            timing_result = service_cost(
                prepared=prepared,
                fixture=fixture,
                output=output,
                expected=actual,
                service_call=lambda: wrapped(fixture["hidden"], fixture["positions"], output),
            )
    if timing_only:
        return {
            "passed": timing_result["passed"]
            and all(value["status"] == "PASS" for value in (*comparisons.values(), *guards.values())),
            "scope": "服务封装成本补充；复用已完成的正式服务/框架图功能验证，不重跑Native对照和回退",
            "framework_acl_graph_wrapper": framework_graph,
            "service_cost": timing_result,
            "submissions": runtime.submissions,
            "comparisons": comparisons,
            "guards": guards,
        }

    changed_metadata = None
    if metadata_replay:
        from service_metadata_replay import validate as validate_changed_metadata

        if not framework_graph or baseline is None:
            raise ValueError("变metadata服务图需要框架包装器和原SPMD对照")
        changed_metadata = validate_changed_metadata(
            runtime=runtime,
            layer=layer,
            config=config,
            fixture=fixture,
            output=output,
            baseline=baseline,
            options=options,
            call=call,
            contracts=contracts,
        )

    # 全dummy实际通过同一个custom-op，检查没有触碰任何cache/state。
    saved_counts = {name: item.num_actual_tokens for name, item in fixture["metadata"].items()}
    cache_before = {name: group["allocation"].cpu() for name, group in fixture["groups"].items()}
    old_empty_calls, old_submissions = runtime.empty_calls, runtime.submissions
    try:
        for item in fixture["metadata"].values():
            item.num_actual_tokens = 0
        _, empty_context = call(fixture, model_graph=framework_graph)
        torch.npu.synchronize()
        empty = {
            "output": compare_tensor(output, fixture["hidden"], 0, 0),
            "cache_unchanged": {
                name: compare_tensor(group["allocation"], cache_before[name], 0, 0)
                for name, group in fixture["groups"].items()
            },
            "empty_handler_called": runtime.empty_calls == old_empty_calls + 1,
            "no_spmd_submission": runtime.submissions == old_submissions,
            "old_graph_rejected": not all(value.allows_replay(empty_context) for value in contracts.values()),
        }
    finally:
        for name, count in saved_counts.items():
            fixture["metadata"][name].num_actual_tokens = count

    # 独立真实短历史Native输入，用新fixture验证范围外回退；没有更改S6假设。
    fallback = make_fixture(
        config,
        layer.self_attn,
        1,
        2048,
        1024,
        fixture["hidden"].device,
        query_tokens=1,
    )
    fallback_output = torch.empty_like(fallback["hidden"])
    with set_ascend_forward_context(
        fallback["metadata"],
        config,
        num_tokens=1,
        num_actual_tokens=1,
    ):
        _native_attention_half(
            layer.self_attn.dsa_attn,
            fallback["hidden"],
            fallback["positions"],
            fallback_output,
        )
    torch.npu.synchronize()
    native_output = fallback_output.cpu()
    native_state = {name: group["allocation"].cpu() for name, group in fallback["groups"].items()}
    restore(fallback)
    with set_ascend_forward_context(
        fallback["metadata"],
        config,
        num_tokens=1,
        num_actual_tokens=1,
    ):
        fallback_eligible = runtime.eligible(get_forward_context(), fallback["hidden"], fallback["positions"])
        torch.ops.vllm.dsv4_csa_forward(
            fallback["hidden"],
            fallback["positions"],
            fallback_output,
            layer.self_attn.dsa_attn.prefix,
        )
    torch.npu.synchronize()
    fallback_checks = {
        "eligible": fallback_eligible,
        "no_spmd_submission": runtime.submissions == old_submissions,
        "output": compare_tensor(fallback_output, native_output, 0, 0),
        "cache": {
            name: compare_tensor(group["allocation"], native_state[name], 0, 0)
            for name, group in fallback["groups"].items()
        },
        "guards": guard_checks(fallback),
    }
    passed = (
        workspace_reused
        and (timing_result is None or timing_result["passed"])
        and (changed_metadata is None or changed_metadata["passed"])
        and all(value["status"] == "PASS" for value in (*comparisons.values(), *guards.values()))
        and all(
            row["python_not_resubmitted"]
            and row["same_addresses"]
            and all(value["status"] == "PASS" for value in (*row["checks"].values(), *row["guards"].values()))
            for row in replays
        )
        and empty["output"]["status"] == "PASS"
        and empty["empty_handler_called"]
        and empty["no_spmd_submission"]
        and empty["old_graph_rejected"]
        and all(value["status"] == "PASS" for value in empty["cache_unchanged"].values())
        and not fallback_checks["eligible"]
        and fallback_checks["no_spmd_submission"]
        and fallback_checks["output"]["status"] == "PASS"
        and all(
            value["status"] == "PASS"
            for value in (*fallback_checks["cache"].values(), *fallback_checks["guards"].values())
        )
    )
    return {
        "passed": passed,
        "scope": "正式模型准备钩子+单层Torch custom-op设备路由；不等价于完整模型端到端推理",
        "framework_acl_graph_wrapper": framework_graph,
        "comparisons": comparisons,
        "guards": guards,
        "fixed_input_graph": replays,
        "workspace_reused_across_capture": workspace_reused,
        "changed_metadata_graph": changed_metadata,
        "service_cost": timing_result,
        "empty": empty,
        "short_history_fallback": fallback_checks,
        "submissions": runtime.submissions,
    }
