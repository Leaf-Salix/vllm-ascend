# SPDX-License-Identifier: Apache-2.0
"""HCA真实custom op与ACLGraphWrapper验证；只测功能，不测性能。"""

import argparse
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

SCENES = {
    "C04": [6] * 16,
    "C08": [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 3, 4, 6],
    "C05": [6] * 24,
    "S1": [1] * 16,
    "S5": [5] * 16,
    "B1": [6],
}


def snapshot(scene, output):
    result = {"output": output[:scene["actual_tokens"]].cpu().clone()}
    for name, group in scene["groups"].items():
        for index, view in enumerate(group["views"]):
            result[f"{name}.{index}"] = view.cpu().clone()
    return result


def exact(actual, expected):
    failures = [name for name in actual if not torch.equal(
        actual[name].contiguous().view(torch.uint8), expected[name].contiguous().view(torch.uint8))]
    if failures:
        raise AssertionError({name: compare_tensor(actual[name], expected[name], 0, 0) for name in failures})
    return list(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", type=int, help="接受队列注入的物理卡号；单卡可见映射后使用逻辑0")
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    args = parser.parse_args()
    if args.device is not None and args.device != int(os.environ["TASK_DEVICE"]):
        raise ValueError("设备参数与队列分配不符")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "performance_measured": False, "service_binding": True,
              "framework_graph_replay": True, "layer": 3, "scenes": [],
              # HCA不同根/嵌套编排使用多个ring：B16 ring2申请22044672B，B24 ring0申请20372480B。
              # 小ring统一64MiB，避免直接套用CSA只有ring1较大的配置；最终须实测全部目标档。
              "ring_config": {"heap_mb": [64, 256, 64, 64], "task_window": 4096},
              "toolchain": json.loads(Path(os.environ["CSA_SHARED_TOOLCHAIN_MANIFEST"]).read_text())}

    def save(stage):
        report["stage"] = stage
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("HCA_MODEL_PROBE " + stage, flush=True)

    save("initialize")
    try:
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        torch.npu.set_compile_mode(jit_compile=False)
        torch_npu.npu.set_deterministic_level(1)
        from vllm.config import CUDAGraphMode
        from vllm.engine.arg_utils import EngineArgs
        from vllm.forward_context import BatchDescriptor, get_forward_context
        from vllm.platforms import current_platform

        current_platform.pre_register_and_update()
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        from vllm_ascend.models.pypto_deepseek_v4_hca import PyptoHCATNDDeepseekV4ForCausalLM, install_hca_forward
        from vllm_ascend.ops.dsv4_hca import native_attention_half
        from vllm_ascend.ops.pypto.deepseek_v4_flash_hca_tnd.native_adapter import NativeHCACall, token_capacity_view
        from vllm_ascend.utils import enable_custom_op

        assert enable_custom_op()
        model = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
        config = EngineArgs(model=str(model), tokenizer_mode="deepseek_v4", trust_remote_code=True,
                            dtype="bfloat16", quantization="ascend", tensor_parallel_size=1,
                            hf_overrides={"sliding_window": 128}, max_model_len=131584,
                            max_num_seqs=24, max_num_batched_tokens=256, block_size=32,
                            enable_prefix_caching=False, enforce_eager=True,
                            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
                            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False,
                                               "pto_csa_ring_config": report["ring_config"]}).create_engine_config()
        device = torch.device("npu:0")
        with native_session(config, 0), torch.inference_mode():
            layer, report["weights"] = make_layer(config, model, device, 3)
            install_hca_forward(layer)
            shell = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
            PyptoHCATNDDeepseekV4ForCausalLM.process_weights_after_loading(shell)
            runtime = layer.self_attn.dsa_attn._pto_hca_runtime
            report["runtime_class"] = type(runtime).__module__
            report["capture_sequence_length"] = shell._pto_hca_tnd_capture_sequence_length
            retained = {}
            for name in args.scenes:
                queries = SCENES[name]
                capacity, requests = sum(queries), len(queries)
                base = make_fixture(config, layer.self_attn, requests, 131071, 1024, device,
                                    table_history=131584, query_lengths=queries)
                output = torch.empty_like(base["hidden"])
                descriptor = BatchDescriptor(capacity, num_reqs=requests, uniform=False)

                def forward(hidden, positions, target):
                    torch.ops.vllm.dsv4_hca_forward(hidden, positions, target, layer.self_attn.dsa_attn.prefix)
                    return target

                graph = ACLGraphWrapper(forward, config, CUDAGraphMode.FULL)
                # 共享图池的所有档位及输入/输出在测试结束前保持存活，匹配真实模型生命周期。
                retained[name] = dict(graph=graph, base=base, output=output)
                targets = {}
                for key, group in base["groups"].items():
                    req = base["metadata"][group["prefix"]].decode
                    target = {field: getattr(req, field) for field in ("query_start_loc", "seq_lens", "block_table", "start_pos")}
                    if key in ("swa", "state"):
                        target["slot_mapping"] = token_capacity_view(req.slot_mapping, capacity, 2)
                    if key == "compressed":
                        target.update({field: token_capacity_view(getattr(req, field)[runtime.layer_name], capacity, 64)
                                       for field in ("cos", "sin")})
                    targets[key] = target

                def activate(scene):
                    for group in scene["groups"].values():
                        group["owner"].kv_cache = group["views"]

                def call(replay=False):
                    extra = dict(aclgraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=descriptor) if replay else {}
                    with set_ascend_forward_context(base["metadata"], config, num_tokens=capacity,
                                                   num_actual_tokens=base["actual_tokens"], **extra):
                        context = get_forward_context()
                        assert runtime.eligible(context, base["hidden"], base["positions"])
                        (graph if replay else forward)(base["hidden"], base["positions"], output)
                        return context

                def install(scene):
                    base["hidden"].copy_(scene["hidden"])
                    base["positions"].copy_(scene["positions"])
                    base["actual_tokens"] = scene["actual_tokens"]
                    readonly = {"positions": base["positions"]}
                    for key, group in base["groups"].items():
                        meta = base["metadata"][group["prefix"]]
                        other = scene["metadata"][scene["groups"][key]["prefix"]]
                        req, source = meta.decode, other.decode
                        for field in ("num_actual_tokens", "num_decode_tokens"):
                            setattr(meta, field, getattr(other, field))
                        for field in ("num_reqs_actual", "num_compressed_tokens", "max_seqlen_q", "max_seqlen_q_actual", "max_seq_lens"):
                            setattr(req, field, getattr(source, field))
                        for field, target in targets[key].items():
                            value = getattr(source, field)
                            if field in ("cos", "sin"):
                                target.copy_(token_capacity_view(value[runtime.layer_name], capacity, 64))
                                setattr(req, field, {runtime.layer_name: target[:scene["actual_tokens"]]})
                            elif field == "slot_mapping":
                                target.copy_(token_capacity_view(value, capacity, 2))
                                req.slot_mapping = target[:scene["actual_tokens"]]
                            else:
                                target.copy_(value)
                            readonly[f"{key}.{field}"] = target
                        req.input_positions = base["positions"][:scene["actual_tokens"]]
                        group["allowed"] = scene["groups"][key]["allowed"]
                    base["readonly"] = {key: (value, value.cpu().clone()) for key, value in readonly.items()}
                    activate(base)

                row = {"case": name, "token_capacity": capacity, "request_capacity": requests,
                       "query_lengths": queries, "steps": []}
                report["scenes"].append(row)
                save(name + ":warmup")
                restore(base)
                call()
                restore(base)
                call(True)
                torch.npu.synchronize()
                contracts = graph.concrete_aclgraph_entries[descriptor].csa_tnd_contracts
                assert contracts and all(c.signature is not None for c in contracts.values())
                captured_submissions = runtime.submissions
                row["policy"] = runtime.last_policy
                schedules = [("fixed", queries, 131071), ("history_next", queries, 131073)]
                if requests > 1:
                    schedules += [("tail_padding", queries[:7], 131199), ("last_request_padding", queries[:1], 131071)]
                if name == "C08":
                    schedules += [("mixed_reordered", list(reversed(queries)), 131199)]
                schedules += [("fixed_again", queries, 131071)]
                for step, step_queries, history in schedules:
                    save(name + ":" + step)
                    scene = make_fixture(config, layer.self_attn, len(step_queries), history, 1024, device,
                                         table_history=131584, query_lengths=step_queries,
                                         token_capacity=capacity, request_capacity=requests, cache_template=base["groups"])
                    reference = NativeHCACall(runtime.operators, runtime.weights, scene["hidden"], scene["positions"],
                                              {key: (scene["metadata"][g["prefix"]], g["views"]) for key, g in scene["groups"].items()},
                                              layer_name=runtime.layer_name, compact_metadata=scene["compact"]["compressed"], output=output)
                    restore(scene)
                    output.fill_(float("nan"))
                    reference()
                    torch.npu.synchronize()
                    expected = snapshot(scene, output)
                    assert torch.isfinite(expected["output"]).all()
                    assert all(v["status"] == "PASS" for v in guard_checks(scene).values())
                    if step == "fixed":
                        restore(scene)
                        activate(scene)
                        with set_ascend_forward_context(scene["metadata"], config, num_tokens=capacity,
                                                       num_actual_tokens=scene["actual_tokens"]):
                            native_attention_half(layer.self_attn.dsa_attn, scene["hidden"], scene["positions"], output)
                        torch.npu.synchronize()
                        row["native_output"] = compare_tensor(expected["output"], output.cpu(), 0.01, 0.01)
                        assert row["native_output"]["status"] == "PASS", row["native_output"]
                    install(scene)
                    restore(base)
                    output.fill_(float("nan"))
                    context = call(True)
                    torch.npu.synchronize()
                    checks = exact(snapshot(base, output), expected)
                    guards = guard_checks(base)
                    assert runtime.submissions == captured_submissions, "Python重新派发，不能冒充真实replay"
                    assert all(c.allows_replay(context) for c in contracts.values())
                    assert all(v["status"] == "PASS" for v in guards.values())
                    row["steps"].append({"name": step, "history": history, "actual_requests": len(step_queries),
                                          "actual_tokens": scene["actual_tokens"], "query_lengths": step_queries,
                                          "output_cache_bit_equal": checks, "guards": guards,
                                          "graph_contract_allows": True, "python_not_resubmitted": True})
                    save(name + ":" + step + ":PASS")
            report["status"] = "PASS"
            save("complete")
    except BaseException as exc:
        import traceback
        report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        save(report["stage"] + ":FAIL")
        raise


if __name__ == "__main__":
    main()
