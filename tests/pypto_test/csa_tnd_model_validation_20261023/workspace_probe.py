# SPDX-License-Identifier: Apache-2.0
"""单卡对照独占/复用工作区：B16 S1/S6、两层交替、图重放，保持原算术。"""

import importlib
import json
import sys
from pathlib import Path

import torch
import torch_npu
from dsv4_csa_native_case import native_session
from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore


def snapshot(fixture, native):
    return collect_state(fixture, native["x_out"], native["idx_topk"])


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    failures = [
        name
        for name in actual
        if not torch.equal(actual[name].contiguous().view(torch.uint8), expected[name].contiguous().view(torch.uint8))
    ]
    assert not failures, failures
    return list(actual)


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "scope": "单卡工作区生命周期，不替代16卡真实decode", "variants": {}}

    def save():
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_deterministic_level(1)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    from vllm_ascend.models.pypto_deepseek_v4 import init_pto_runtime
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa import native_adapter as adapter
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.native_adapter import bind_native_inputs
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.workspace import TNDWorkspacePool
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
        max_num_seqs=16,
        max_num_batched_tokens=256,
        block_size=32,
        enable_prefix_caching=False,
        enforce_eager=True,
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
    ).create_engine_config()
    device = torch.device("npu:0")
    with native_session(config, 0), torch.inference_mode():
        init_pto_runtime()
        layers = {i: make_layer(config, model, device, i)[0] for i in (2, 4)}
        scenes = []
        for index, (layer_index, query_tokens) in enumerate(((2, 1), (2, 6), (4, 6))):
            layer = layers[layer_index]
            fixture = make_fixture(config, layer.self_attn, 16, 131072, 1024 + index, device, query_tokens=query_tokens)
            groups = {
                name: (fixture["metadata"][group["prefix"]], group["views"])
                for name, group in fixture["groups"].items()
            }
            weights = adapter.prepare_weights(layer.self_attn, groups["indexer"][0].hadamard, layer)
            native = bind_native_inputs(
                weights,
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
        report["stage"] = "native_scenes_ready"
        save()
        for variant in ("deepseek_v4_flash_csa_spmd_tnd", "deepseek_v4_flash_csa_tnd"):
            binding = importlib.import_module("vllm_ascend.ops.pypto." + variant + ".binding").PreparedDynamicCSA
            pool = TNDWorkspacePool()
            controls, candidates, expected = [], [], []
            for fixture, native in scenes:
                options = {"max_query_tokens": 6, "max_seq_len": 131078}
                controls.append(binding(native, **options))
                candidates.append(binding(native, allocator=pool.allocate, **options))
            assert candidates[1].buffers["pair_arena"].data_ptr() == candidates[2].buffers["pair_arena"].data_ptr()
            if "signal" in candidates[0].buffers:
                assert len({c.buffers["signal"].data_ptr() for c in candidates}) == 3
            for (fixture, native), entry in zip(scenes, controls):
                restore(fixture)
                entry()
                torch.npu.synchronize()
                expected.append(snapshot(fixture, native))
            for fixture, _ in scenes:
                restore(fixture)
            # 中间不加设备同步；按模型顺序提交，检验相邻调用复用同一临时地址。
            for entry in candidates:
                entry()
            torch.npu.synchronize()
            eager = [compare(snapshot(f, n), e) for (f, n), e in zip(scenes, expected)]
            for fixture, _ in scenes:
                restore(fixture)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for entry in candidates:
                    entry()
            for fixture, _ in scenes:
                restore(fixture)
            graph.replay()
            torch.npu.synchronize()
            replay = [compare(snapshot(f, n), e) for (f, n), e in zip(scenes, expected)]
            guards = [guard_checks(f) for f, _ in scenes]
            assert all(row["status"] == "PASS" for checks in guards for row in checks.values()), guards
            report["variants"][variant] = {
                "status": "PASS",
                "eager_bit_equal": eager,
                "graph_bit_equal": replay,
                "guard_status": "PASS",
                "shared_bytes": pool.allocated_bytes,
                "independent_bytes": sum(
                    t.numel() * t.element_size() for e in controls for t in e.allocations.values()
                ),
                "signal_independent": "signal" not in candidates[0].buffers
                or len({e.buffers["signal"].data_ptr() for e in candidates}) == 3,
            }
            save()
            del graph, controls, candidates, expected, pool
        report["status"] = "PASS"
        save()
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
