# SPDX-License-Identifier: Apache-2.0
"""单卡验证Simpler内部临时GM：旧外部版本仅用作数值参考。"""

import argparse
import importlib
import json
from pathlib import Path

import torch
import torch_npu
from csa_tnd_model_validation_20261023.workspace_probe import compare, snapshot
from dsv4_csa_native_case import native_session
from dsv4_csa_single_layer import guard_checks, make_fixture, make_layer, restore


def run_entries(function, inputs, policies):
    for kwargs, policy in zip(inputs, policies):
        function(**kwargs, **policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "scope": "内部工作区原型单卡，尚未接入生产服务", "variants": {}}

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
            "pto_csa_ring_config": {"heap_mb": [256, 256, 256, 256], "task_window": 4096},
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
            binding = importlib.import_module(f"vllm_ascend.ops.pypto.{package}.binding").PreparedDynamicCSA
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
                expected.append(snapshot(fixture, native))
                aliases = {"q_heads", "idx_query_rope_heads", "idx_query_scale_heads"}
                kwargs = {k: v for k, v in control.buffers.items() if k not in control.allocations and k not in aliases}
                call_args.append((kwargs, control.policy.constexprs()))
            report["stage"] = package + ": external_reference_complete"
            save()
            for fixture, _ in scenes:
                restore(fixture)
            for kwargs, policy in call_args:
                debug(**kwargs, **policy)
            torch.npu.synchronize()
            debug_checks = [compare(snapshot(f, n), e) for (f, n), e in zip(scenes, expected)]
            normal_args = [
                {k: v for k, v in kwargs.items() if k not in ("topk_indices", "topk_scores")} for kwargs, _ in call_args
            ]

            policies = [p for _, p in call_args]

            for fixture, native in scenes:
                restore(fixture)
                native["idx_topk"].fill_(-123)
                native["idx_topk_scores"].fill_(7)
            run_entries(normal, normal_args, policies)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run_entries(normal, normal_args, policies)
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
                    checks.append(compare(actual, {k: v for k, v in reference.items() if k != "idx_topk"}))
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
                "normal_external_tensor_arguments": len(normal_args[0]),
                "internal_allocation": "pl.create_tensor，普通入口的Top-K也在内部",
            }
            save()
            del graph, controls, expected, call_args
        report["status"] = "PASS"
        save()
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
