# SPDX-License-Identifier: Apache-2.0
"""单卡真实TND服务图：固定C04/C08/C05，另测尾批/padding同址复用。"""

import argparse
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
from csa_mpmd_tnd_20261021.scenarios import make_fixture
from dsv4_csa_native_case import native_session
from dsv4_csa_single_layer import guard_checks, make_layer, restore
from dsv4_csa_validation import compare_tensor

MIXED_QUERY_LENGTHS = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 3, 4, 6]


def bit_equal(actual, expected):
    return torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def snapshot(scene, output):
    result = {"output": output[: scene["actual_tokens"]].detach().cpu().clone()}
    for name, group in scene["groups"].items():
        for index, view in enumerate(group["views"]):
            result[f"{name}.{index}"] = view.detach().cpu().clone()
    return result


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    failures = [name for name in actual if not bit_equal(actual[name], expected[name])]
    assert not failures, failures
    return list(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("mpmd_tnd", "spmd_tnd"), required=True)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    package = "deepseek_v4_flash_csa_tnd" if args.variant == "mpmd_tnd" else "deepseek_v4_flash_csa_spmd_tnd"
    report = dict(
        status="RUNNING",
        scope="真实custom-op/ACLGraphWrapper单卡功能，不作为整模型或性能结论",
        variants={package: dict(status="RUNNING", scenes=[])},
        batches=[16, 24],
        service_binding=True,
        framework_graph_replay=True,
        performance_measured=False,
        ring_config=dict(heap_mb=[16, 256, 16, 16], task_window=4096),
        toolchain=json.loads(Path(os.environ["CSA_SHARED_TOOLCHAIN_MANIFEST"]).read_text()),
    )

    def save(stage):
        report["stage"] = stage
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("TND_GRAPH_PROBE " + stage, flush=True)

    save("initialize")
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_deterministic_level(1)
    from vllm.config import CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import BatchDescriptor, get_forward_context
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.models.pypto_deepseek_v4 import install_csa_forward, prepare_csa_model
    from vllm_ascend.ops.dsv4_csa import _native_attention_half
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.native_adapter import (
        bind_native_inputs,
        token_capacity_view,
    )
    from vllm_ascend.ops.pypto.variant import selected_variant
    from vllm_ascend.utils import enable_custom_op

    assert selected_variant() == args.variant and enable_custom_op()
    model = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    config = EngineArgs(
        model=str(model),
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        dtype="bfloat16",
        quantization="ascend",
        tensor_parallel_size=1,
        hf_overrides={"sliding_window": 128},
        max_model_len=131584,
        max_num_seqs=24,
        max_num_batched_tokens=256,
        block_size=32,
        enable_prefix_caching=False,
        enforce_eager=True,
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        additional_config={
            "weight_nz_mode": 2,
            "enable_kv_nz": False,
            "enable_dsa_cp": False,
            "pto_csa_ring_config": report["ring_config"],
        },
    ).create_engine_config()
    device = torch.device("npu:0")
    try:
        with native_session(config, 0), torch.inference_mode():
            layer, _ = make_layer(config, model, device, 4)
            install_csa_forward(layer)
            shell_model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
            prepare_csa_model(shell_model)
            assert shell_model._pto_csa_tnd_capture_sequence_length == config.model_config.max_model_len
            runtime = layer.self_attn.dsa_attn._pto_csa_runtime
            binding = importlib.import_module(f"vllm_ascend.ops.pypto.{package}.binding").PreparedDynamicCSA
            # Match model lifetime: all bucket graphs sharing the global pool
            # remain alive, together with their inputs and outputs, until exit.
            retained_graphs = {}
            report["graph_lifetime"] = "各档图及其输入/输出/metadata/binding保留至测试结束"
            for name, queries in (("C04", [6] * 16), ("C08", MIXED_QUERY_LENGTHS), ("C05", [6] * 24), ("tail_B1", [6])):
                capacity, requests = sum(queries), len(queries)
                base = make_fixture(
                    config, layer.self_attn, requests, 131072, 1024, device, table_history=131072, query_lengths=queries
                )
                assert base["tokens"] == capacity and base["request_capacity"] == requests
                output = torch.empty_like(base["hidden"])
                topk = torch.empty((capacity, 512), dtype=torch.int32, device=device)
                scores = torch.empty((capacity, 512), dtype=torch.float32, device=device)
                descriptor = BatchDescriptor(capacity, num_reqs=requests, uniform=False)

                def forward(hidden, positions, target):
                    torch.ops.vllm.dsv4_csa_forward(hidden, positions, target, layer.self_attn.dsa_attn.prefix)
                    return target

                graph = ACLGraphWrapper(forward, config, CUDAGraphMode.FULL)
                retained_graphs[name] = dict(graph=graph, scene=base, output=output, topk=topk, scores=scores)
                report["retained_graph_cases"] = list(retained_graphs)
                targets = {}
                for group_name, group in base["groups"].items():
                    req = base["metadata"][group["prefix"]].decode
                    target = {
                        field: getattr(req, field)
                        for field in ("query_start_loc", "seq_lens", "block_table", "start_pos")
                    }
                    if group_name in ("swa", "state", "indexer_state"):
                        target["slot_mapping"] = token_capacity_view(req.slot_mapping, capacity, 2)
                    if group_name == "compressed":
                        target.update(
                            {
                                field: token_capacity_view(getattr(req, field)[runtime.layer_name], capacity, 64)
                                for field in ("cos", "sin")
                            }
                        )
                    targets[group_name] = target

                def activate(scene):
                    for group in scene["groups"].values():
                        group["owner"].kv_cache = [group["views"]] if len(group["views"]) == 2 else group["views"]

                def call(
                    graph_mode=False, *, base=base, capacity=capacity, descriptor=descriptor, graph=graph, output=output
                ):
                    extra = (
                        dict(aclgraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=descriptor)
                        if graph_mode
                        else {}
                    )
                    with set_ascend_forward_context(
                        base["metadata"], config, num_tokens=capacity, num_actual_tokens=base["actual_tokens"], **extra
                    ):
                        context = get_forward_context()
                        assert runtime.eligible(context, base["hidden"], base["positions"])
                        (graph if graph_mode else forward)(base["hidden"], base["positions"], output)
                        return context

                def install(scene, *, base=base, targets=targets, capacity=capacity):
                    base["hidden"].copy_(scene["hidden"])
                    base["positions"].copy_(scene["positions"])
                    base["actual_tokens"] = scene["actual_tokens"]
                    readonly = {"positions": base["positions"]}
                    for group_name, group in base["groups"].items():
                        meta = base["metadata"][group["prefix"]]
                        other = scene["metadata"][scene["groups"][group_name]["prefix"]]
                        req, source = meta.decode, other.decode
                        for field in ("num_actual_tokens", "num_decode_tokens"):
                            setattr(meta, field, getattr(other, field))
                        for field in (
                            "num_reqs_actual",
                            "num_compressed_tokens",
                            "max_seqlen_q",
                            "max_seqlen_q_actual",
                            "max_seq_lens",
                        ):
                            setattr(req, field, getattr(source, field))
                        for field, target in targets[group_name].items():
                            value = getattr(source, field)
                            if field in ("cos", "sin"):
                                value = token_capacity_view(value[runtime.layer_name], capacity, 64)
                                target.copy_(value)
                                setattr(req, field, {runtime.layer_name: target[: scene["actual_tokens"]]})
                            elif field == "slot_mapping":
                                target.copy_(token_capacity_view(value, capacity, 2))
                                req.slot_mapping = target[: scene["actual_tokens"]]
                            else:
                                target.copy_(value)
                            readonly[f"{group_name}.{field}"] = target
                        req.input_positions = base["positions"][: scene["actual_tokens"]]
                        group["allowed"] = scene["groups"][group_name]["allowed"]
                    base["readonly"] = {key: (value, value.cpu().clone()) for key, value in readonly.items()}
                    activate(base)

                row = dict(
                    case=name,
                    history=131072,
                    request_capacity=requests,
                    token_capacity=capacity,
                    query_lengths=queries,
                    layer=4,
                    seed=1024,
                    steps=[],
                )
                report["variants"][package]["scenes"].append(row)
                save(name + ":warmup")
                restore(base)
                call()
                restore(base)
                call(True)
                torch.npu.synchronize()
                contracts = graph.concrete_aclgraph_entries[descriptor].csa_tnd_contracts
                assert contracts and all(c.signature is not None for c in contracts.values())
                captured_submissions = runtime.submissions
                captured = runtime.last_binding
                retained_graphs[name]["binding"] = captured
                assert not captured.allocations

                # Keep the original case immutable; derived steps exercise the same graph.
                schedules = [("fixed", list(queries))]
                if requests > 1:
                    schedules += [("tail_padding", list(queries[:7])), ("last_request_padding", list(queries[:1]))]
                if name == "C08":
                    schedules += [("mixed_reordered", list(reversed(queries)))]
                schedules += [("fixed_again", list(queries))]
                for step_name, step_queries in schedules:
                    save(name + ":" + step_name)
                    scene = make_fixture(
                        config,
                        layer.self_attn,
                        len(step_queries),
                        131072,
                        1024,
                        device,
                        table_history=131072,
                        query_lengths=step_queries,
                        token_capacity=capacity,
                        request_capacity=requests,
                        cache_template=base["groups"],
                    )
                    # Reference consumes the original Native producer with the actual request count.
                    native = bind_native_inputs(
                        runtime.weights,
                        scene["hidden"],
                        scene["positions"],
                        {key: (scene["metadata"][g["prefix"]], g["views"]) for key, g in scene["groups"].items()},
                        layer_name=runtime.layer_name,
                        compact_metadata=scene["compact"],
                        output=output,
                        topk=topk,
                        topk_scores=scores,
                    )
                    control = binding(native, max_query_tokens=6, max_seq_len=131078, policy=captured.policy)
                    restore(scene)
                    output.fill_(float("nan"))
                    topk.fill_(-777)
                    scores.fill_(float("nan"))
                    control()
                    torch.npu.synchronize()
                    expected = snapshot(scene, output)
                    expected_topk = topk[: scene["actual_tokens"]].cpu().clone()
                    expected_scores = scores[: scene["actual_tokens"]].cpu().clone()
                    assert torch.isfinite(expected["output"]).all() and torch.isfinite(expected_scores).all()
                    assert all(v["status"] == "PASS" for v in guard_checks(scene).values())
                    if step_name == "fixed":
                        restore(scene)
                        activate(scene)
                        with set_ascend_forward_context(
                            scene["metadata"], config, num_tokens=capacity, num_actual_tokens=scene["actual_tokens"]
                        ):
                            _native_attention_half(
                                layer.self_attn.dsa_attn, scene["hidden"], scene["positions"], output
                            )
                        torch.npu.synchronize()
                        gap = compare_tensor(expected["output"], output[: scene["actual_tokens"]].cpu(), 0, 0)
                        gap["bit_equal"] = gap.pop("status") == "PASS"
                        row["native_output_gap"] = gap
                        row["native_gap_scope"] = "记录既有Native/TND差异；新图必须与TND独立eager参考逐bit一致"
                    install(scene)
                    restore(base)
                    output.fill_(float("nan"))
                    context = call(True)
                    torch.npu.synchronize()
                    checks = compare(snapshot(base, output), expected)
                    guards = guard_checks(base)
                    allowed = all(c.allows_replay(context) for c in contracts.values())
                    assert allowed and runtime.submissions == captured_submissions
                    assert all(v["status"] == "PASS" for v in guards.values())
                    # Inspect Top-K via the explicit diagnostic entry with the graph-produced metadata.
                    diagnostic = dict(captured.native, idx_topk=topk, idx_topk_scores=scores)
                    debug = binding(diagnostic, max_query_tokens=6, max_seq_len=131078, policy=captured.policy)
                    restore(base)
                    topk.fill_(-777)
                    scores.fill_(float("nan"))
                    debug()
                    torch.npu.synchronize()
                    assert bit_equal(topk[: scene["actual_tokens"]].cpu(), expected_topk)
                    assert bit_equal(scores[: scene["actual_tokens"]].cpu(), expected_scores)
                    row["steps"].append(
                        dict(
                            name=step_name,
                            actual_requests=len(step_queries),
                            actual_tokens=scene["actual_tokens"],
                            query_lengths=step_queries,
                            graph_contract_allows=True,
                            python_not_resubmitted=True,
                            output_cache_bit_equal=checks,
                            guards=guards,
                            diagnostic_topk_score_bit_equal=True,
                            external_workspace_bytes=0,
                        )
                    )
                    save(name + ":" + step_name + ":PASS")
                    del scene, native, control, expected, expected_topk, expected_scores, debug, diagnostic
                del graph, base, targets, captured, output, topk, scores
            torch.npu.synchronize()
            report["variants"][package].update(status="PASS", guard_status="PASS")
            report["status"] = "PASS"
            save("complete")
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = repr(exc)
        save(report.get("stage", "unknown") + ":FAIL")
        raise


if __name__ == "__main__":
    main()
