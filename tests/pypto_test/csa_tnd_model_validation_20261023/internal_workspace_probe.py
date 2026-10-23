# SPDX-License-Identifier: Apache-2.0
"""单卡验证Simpler内部临时GM：旧外部版本仅用作数值参考。"""

import argparse
import importlib
import importlib.util
import json
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch_npu
from csa_tnd_model_validation_20261023.workspace_probe import compare, snapshot
from dsv4_csa_native_case import native_session
from dsv4_csa_single_layer import guard_checks, make_fixture, make_layer, restore


def run_entries(entries):
    for entry in entries:
        entry()


def external_reference(source, package, function):
    """只在测试进程装载已冻结旧实现，作为内部服务的数值参考。"""
    parent = f"vllm_ascend.ops.pypto.{package}"
    directory = source / "vllm_ascend/ops/pypto" / package
    for name, filename in (("external_reference_entry", "entry.py"), ("external_reference_binding", "binding.py")):
        path = directory / filename
        spec = importlib.util.spec_from_file_location(f"{parent}.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        code = path.read_text()
        if filename == "binding.py":
            original = f"from .entry import {function}"
            assert code.count(original) == 1
            code = code.replace(original, f"from .external_reference_entry import {function}")
        exec(compile(code, str(path), "exec"), module.__dict__)
    return module.PreparedDynamicCSA


def snapshot_with_scores(fixture, native):
    state = snapshot(fixture, native)
    state["idx_topk_scores"] = native["idx_topk_scores"].detach().cpu().clone()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", type=int)
    parser.add_argument("--variant", choices=("both", "mpmd_tnd", "spmd_tnd"), default="both")
    parser.add_argument("--service-binding", action="store_true", help="验证实际PreparedDynamicCSA服务绑定")
    parser.add_argument("--reference-source", type=Path, help="已冻结的旧外部工作区源码，仅用于数值参考")
    parser.add_argument("--ring-heap-mb", type=int, nargs=4, default=[256, 256, 256, 256])
    parser.add_argument("--ring-task-window", type=int, default=4096)
    args = parser.parse_args()
    if args.service_binding and not args.reference_source:
        parser.error("服务绑定验证须提供独立的冻结参考源码")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "scope": "内部工作区实际服务绑定单卡" if args.service_binding else "内部工作区原型单卡",
        "requested_variant": args.variant,
        "variants": {},
        "batches": [16, 24],
        "service_binding": args.service_binding,
        "ring_config": {"heap_mb": args.ring_heap_mb, "task_window": args.ring_task_window},
        "toolchain": json.loads(Path(os.environ["CSA_SHARED_TOOLCHAIN_MANIFEST"]).read_text()),
        "reference_source": str(args.reference_source.resolve()) if args.reference_source else "same_frozen_source",
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_deterministic_level(1)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    from vllm_ascend.models.pypto_deepseek_v4 import init_pto_runtime
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa import native_adapter as adapter
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.native_adapter import bind_native_inputs
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op()
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
    with native_session(config, 0), torch.inference_mode():
        init_pto_runtime()
        layers = {i: make_layer(config, model, device, i)[0] for i in (2, 4)}
        scenes = []
        for index, (batch, query, layer_index) in enumerate(((16, 1, 2), (16, 6, 2), (24, 6, 4))):
            layer = layers[layer_index]
            fixture = make_fixture(config, layer.self_attn, batch, 131072, 1500 + index, device, query_tokens=query)
            groups = {name: (fixture["metadata"][g["prefix"]], g["views"]) for name, g in fixture["groups"].items()}
            native = bind_native_inputs(
                adapter.prepare_weights(layer.self_attn, groups["indexer"][0].hadamard, layer),
                fixture["hidden"],
                fixture["positions"],
                groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                compact_metadata=fixture["compact"],
                output=torch.empty_like(fixture["hidden"]),
                topk=torch.empty((fixture["tokens"], 512), dtype=torch.int32, device=device),
                topk_scores=torch.empty((fixture["tokens"], 512), dtype=torch.float32, device=device),
            )
            scenes.append((fixture, native))
        report["scenes"] = ["128K/B16/S1/layer2", "128K/B16/S6/layer2", "128K/B24/S6/layer4"]
        save()
        for package, function in (
            ("deepseek_v4_flash_csa_tnd", "decode_csa_tnd"),
            ("deepseek_v4_flash_csa_spmd_tnd", "decode_csa_spmd"),
        ):
            if args.variant == "mpmd_tnd" and package.endswith("spmd_tnd"):
                continue
            if args.variant == "spmd_tnd" and not package.endswith("spmd_tnd"):
                continue
            current_binding = importlib.import_module(f"vllm_ascend.ops.pypto.{package}.binding").PreparedDynamicCSA
            if args.service_binding:
                binding = external_reference(args.reference_source.resolve(), package, function)
            else:
                binding = current_binding
                internal = importlib.import_module(f"vllm_ascend.ops.pypto.{package}.internal_prototype")
                debug = getattr(internal, function + "_internal_debug")
                normal = getattr(internal, function + "_internal")
            controls, expected, call_args = [], [], []
            for fixture, native in scenes:
                control = binding(native, max_query_tokens=6, max_seq_len=131078)
                controls.append(control)
                restore(fixture)
                control()
                torch.npu.synchronize()
                expected.append(snapshot_with_scores(fixture, native))
                aliases = {"q_heads", "idx_query_rope_heads", "idx_query_scale_heads"}
                kwargs = {k: v for k, v in control.buffers.items() if k not in control.allocations and k not in aliases}
                call_args.append((kwargs, control.policy.constexprs()))
            report["stage"] = package + ": external_reference_complete"
            save()
            for fixture, _ in scenes:
                restore(fixture)
            if args.service_binding:
                debug_entries = [
                    current_binding(native, max_query_tokens=6, max_seq_len=131078) for _, native in scenes
                ]
                normal_entries = [
                    current_binding(
                        {key: value for key, value in native.items() if key not in ("idx_topk", "idx_topk_scores")},
                        max_query_tokens=6,
                        max_seq_len=131078,
                    )
                    for _, native in scenes
                ]
                assert all(not entry.allocations for entry in (*debug_entries, *normal_entries))
                assert all("topk_indices" not in entry.buffers for entry in normal_entries)
                external_arguments = len(normal_entries[0].buffers)
            else:
                debug_entries = [partial(debug, **kwargs, **policy) for kwargs, policy in call_args]
                normal_args = [
                    {k: v for k, v in kwargs.items() if k not in ("topk_indices", "topk_scores")}
                    for kwargs, _ in call_args
                ]
                normal_entries = [partial(normal, **kwargs, **p) for kwargs, (_, p) in zip(normal_args, call_args)]
                external_arguments = len(normal_args[0])
            run_entries(debug_entries)
            torch.npu.synchronize()
            debug_checks = [compare(snapshot_with_scores(f, n), e) for (f, n), e in zip(scenes, expected)]
            assert all(bool(torch.isfinite(native["x_out"]).all()) for _, native in scenes)

            for fixture, native in scenes:
                restore(fixture)
                native["idx_topk"].fill_(-123)
                native["idx_topk_scores"].fill_(7)
            run_entries(normal_entries)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run_entries(normal_entries)
            graph_checks = []
            for _ in range(2):
                for fixture, _ in scenes:
                    restore(fixture)
                graph.replay()
                torch.npu.synchronize()
                checks = []
                for (fixture, native), reference in zip(scenes, expected):
                    actual = snapshot(fixture, native)
                    del actual["idx_topk"]
                    checks.append(
                        compare(
                            actual, {k: v for k, v in reference.items() if k not in ("idx_topk", "idx_topk_scores")}
                        )
                    )
                    assert bool(torch.all(native["idx_topk"] == -123))
                    assert bool(torch.all(native["idx_topk_scores"] == 7))
                graph_checks.append(checks)
            guards = [guard_checks(f) for f, _ in scenes]
            assert all(v["status"] == "PASS" for g in guards for v in g.values()), guards
            report["variants"][package] = {
                "status": "PASS",
                "debug_output_topk_cache_bit_equal": debug_checks,
                "normal_graph_output_cache_bit_equal": graph_checks,
                "guard_status": "PASS",
                "normal_ignores_external_debug_outputs": True,
                "normal_external_tensor_arguments": external_arguments,
                "internal_allocation": "pl.create_tensor，普通入口的Top-K也在内部",
            }
            save()
            del graph, controls, expected, call_args, debug_entries, normal_entries
        report["status"] = "PASS"
        report["stage"] = "complete"
        save()
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
