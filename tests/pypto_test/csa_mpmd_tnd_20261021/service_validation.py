# SPDX-License-Identifier: Apache-2.0
"""真实MPMD模型准备钩子、custom-op和ACLGraphWrapper检查；不采性能。"""

from types import SimpleNamespace


def prepare(layer):
    from vllm_ascend.models.pypto_deepseek_v4 import install_csa_forward, prepare_csa_model
    from vllm_ascend.ops.pypto.variant import selected_variant

    if selected_variant() != "mpmd_tnd":
        raise ValueError("本服务检查必须显式命中mpmd_tnd")
    install_csa_forward(layer)
    prepare_csa_model(SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    return layer.self_attn.dsa_attn._pto_csa_runtime


def preflight(runtime, fixture, config):
    """编译前保存真实Host描述符及服务判定，拒绝路径不等到核编译后才暴露。"""
    from vllm.forward_context import get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.service_config import native_query_bound

    groups = {}
    for name, prefix in runtime.prefixes.items():
        metadata = fixture["metadata"][prefix]
        request = metadata.decode
        groups[name] = {
            "num_decodes": metadata.num_decodes,
            "num_prefills": metadata.num_prefills,
            "actual_tokens": metadata.num_actual_tokens,
            "request_capacity": request.seq_lens.numel(),
            "active_requests": request.num_reqs_actual,
            "native_tiling_query_bound": request.max_seqlen_q,
            "native_actual_query_bound": getattr(request, "max_seqlen_q_actual", None),
            "service_query_bound": native_query_bound(request),
            "max_seq_len": request.max_seq_lens,
        }
    with set_ascend_forward_context(
        fixture["metadata"], config, num_tokens=fixture["tokens"], num_actual_tokens=fixture["actual_tokens"]
    ):
        context = get_forward_context()
        eligible = runtime.eligible(context, fixture["hidden"], fixture["positions"])
    return {"eligible": eligible, "token_capacity": fixture["tokens"], "groups": groups}


def validate(*, runtime, layer, config, fixture, output, expected, compare, save):
    import torch
    from dsv4_csa_single_layer import guard_checks, restore
    from dsv4_csa_validation import compare_tensor
    from scenarios import make_fixture
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.ops.dsv4_csa import _native_attention_half

    def forward(hidden, positions, target):
        torch.ops.vllm.dsv4_csa_forward(hidden, positions, target, layer.self_attn.dsa_attn.prefix)
        return target

    descriptor = BatchDescriptor(fixture["tokens"], num_reqs=fixture["request_capacity"], uniform=False)
    wrapper = ACLGraphWrapper(forward, config, CUDAGraphMode.FULL)

    def call(case, target, graph=False):
        extra = dict(aclgraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=descriptor) if graph else {}
        with set_ascend_forward_context(
            case["metadata"],
            config,
            num_tokens=case["tokens"],
            num_actual_tokens=next(iter(case["metadata"].values())).num_actual_tokens,
            **extra,
        ):
            context = get_forward_context()
            eligible = runtime.eligible(context, case["hidden"], case["positions"])
            (wrapper if graph else forward)(case["hidden"], case["positions"], target)
            return eligible, context

    def snapshot(prepared):
        state = {
            "output": output.cpu(),
            "topk": prepared.native["idx_topk"].cpu(),
            "scores": prepared.native["idx_topk_scores"].cpu(),
        }
        for name, group in fixture["groups"].items():
            slots = (
                fixture["compact"][name][2]
                if name in fixture["compact"]
                else fixture["metadata"][group["prefix"]].decode.slot_mapping
            )
            pairs = sorted({(p, r) for p, r in slots.cpu().tolist() if p >= 0 and r >= 0})
            indices = ([p for p, _ in pairs], [r for _, r in pairs])
            raw = group["allocation"].cpu()
            for index, view in enumerate(group["views"]):
                cpu = raw.view(view.dtype).as_strided(view.shape, view.stride(), view.storage_offset())
                state[f"{name}.{index}"] = cpu[indices].clone()
        return state

    save("service_eager")
    before = runtime.submissions
    restore(fixture)
    eligible, _ = call(fixture, output)
    torch.npu.synchronize()
    if not eligible or runtime.submissions != before + 1:
        raise AssertionError("真实custom-op未命中MPMD提交")
    prepared = runtime.last_binding
    actual = snapshot(prepared)
    numerical = compare(actual, expected)
    initial_guards = guard_checks(fixture)
    workspace = {k: v.data_ptr() for k, v in prepared.allocations.items()}
    restore(fixture)
    call(fixture, output, graph=True)
    torch.npu.synchronize()
    contracts = wrapper.concrete_aclgraph_entries[descriptor].csa_tnd_contracts
    capture_submissions = runtime.submissions
    same_workspace = workspace == {k: v.data_ptr() for k, v in runtime.last_binding.allocations.items()}
    addresses = {k: v.data_ptr() for k, v in runtime.last_binding.buffers.items()}
    replays = []
    for _ in range(2):
        restore(fixture)
        _, context = call(fixture, output, graph=True)
        torch.npu.synchronize()
        replays.append(
            {
                "numerical": compare(snapshot(runtime.last_binding), expected),
                "guards": guard_checks(fixture),
                "python_not_resubmitted": runtime.submissions == capture_submissions,
                "same_addresses": addresses == {k: v.data_ptr() for k, v in runtime.last_binding.buffers.items()},
                "contract_allows": bool(contracts) and all(c.allows_replay(context) for c in contracts.values()),
            }
        )
    saved_counts = {name: metadata.num_actual_tokens for name, metadata in fixture["metadata"].items()}
    prior_cache = {name: g["allocation"].cpu() for name, g in fixture["groups"].items()}
    old_empty, old_submissions = runtime.empty_calls, runtime.submissions
    try:
        for metadata in fixture["metadata"].values():
            metadata.num_actual_tokens = 0
        _, context = call(fixture, output, graph=True)
        torch.npu.synchronize()
        empty = {
            "handler_called": runtime.empty_calls == old_empty + 1,
            "no_submission": runtime.submissions == old_submissions,
            "graph_rejected": not all(c.allows_replay(context) for c in contracts.values()),
            "output": compare_tensor(output, fixture["hidden"], 0, 0),
            "cache_unchanged": all(
                torch.equal(g["allocation"].cpu(), prior_cache[name]) for name, g in fixture["groups"].items()
            ),
        }
    finally:
        for name, count in saved_counts.items():
            fixture["metadata"][name].num_actual_tokens = count

    save("service_native_fallback")
    fallback = make_fixture(config, layer.self_attn, 1, 2048, 1024, output.device, query_tokens=1)
    fallback_output = torch.empty_like(fallback["hidden"])
    with set_ascend_forward_context(fallback["metadata"], config, num_tokens=1, num_actual_tokens=1):
        _native_attention_half(layer.self_attn.dsa_attn, fallback["hidden"], fallback["positions"], fallback_output)
    torch.npu.synchronize()
    native_output = fallback_output.cpu()
    native_cache = {name: g["allocation"].cpu() for name, g in fallback["groups"].items()}
    restore(fallback)
    fallback_eligible, _ = call(fallback, fallback_output)
    torch.npu.synchronize()
    fallback_checks = {
        "not_eligible": not fallback_eligible,
        "no_submission": runtime.submissions == old_submissions,
        "output": compare_tensor(fallback_output, native_output, 0.01, 0.01),
        "cache": {
            name: compare_tensor(g["allocation"], native_cache[name], 0, 0) for name, g in fallback["groups"].items()
        },
        "guards": guard_checks(fallback),
    }
    passed = (
        same_workspace
        and all(v["status"] == "PASS" for v in numerical.values())
        and all(v["status"] == "PASS" for v in initial_guards.values())
        and all(
            row["python_not_resubmitted"]
            and row["same_addresses"]
            and row["contract_allows"]
            and all(v["status"] == "PASS" for v in (*row["numerical"].values(), *row["guards"].values()))
            for row in replays
        )
        and all(empty[k] for k in ("handler_called", "no_submission", "graph_rejected", "cache_unchanged"))
        and empty["output"]["status"] == "PASS"
        and fallback_checks["not_eligible"]
        and fallback_checks["no_submission"]
        and all(
            v["status"] == "PASS"
            for v in (
                fallback_checks["output"],
                *fallback_checks["cache"].values(),
                *fallback_checks["guards"].values(),
            )
        )
    )
    return {
        "passed": passed,
        "scope": "正式模型准备、custom-op及ACLGraphWrapper；不是16卡整模型",
        "runtime_type": type(runtime).__module__ + "." + type(runtime).__name__,
        "submissions": runtime.submissions,
        "numerical": numerical,
        "guards": initial_guards,
        "workspace_reused_across_capture": same_workspace,
        "graph_replays": replays,
        "empty": empty,
        "fallback": fallback_checks,
    }
