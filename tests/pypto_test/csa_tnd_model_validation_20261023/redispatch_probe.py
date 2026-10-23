# SPDX-License-Identifier: Apache-2.0
"""单卡Native真实半层：图回退容量视图与有效TND输入对照。"""

import argparse
import json
import os
from copy import copy
from pathlib import Path

import torch
import torch_npu
from dsv4_csa_native_case import native_session
from dsv4_csa_single_layer import guard_checks, make_fixture, make_layer, restore
from dsv4_csa_validation import compare_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", type=int, help="task-submit附加的物理卡号；进程内使用可见设备0")
    parser.add_argument("--layers", type=int, nargs="+", choices=(1, 2, 3), default=[1, 3, 2])
    args = parser.parse_args()
    if args.device is not None and args.device != int(os.environ["TASK_DEVICE"]):
        raise ValueError("命令行设备与队列分配不一致")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "cases": [], "scope": "Native半层padding功能，不替代整模型输出验证"}

    def save():
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_deterministic_level(1)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import get_forward_context
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.ops.dsv4_csa import _native_attention_half
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.graph_contract import TND_PADDED_REDISPATCH
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
    histories = [131453, 131483, 131466, 131471, 131454, 131437, 131434]

    def capture_metadata(scene, active):
        # Native图捕获时Python切片固定为容量96；device Q/L仍保留7/4个有效请求。
        # 用原builder生成该捕获形状，不经过待验证的容量适配函数。
        metadata = {}
        common_cache, prefill_cache, decode_cache = {}, {}, {}
        for group in scene["groups"].values():
            common = copy(group["common"])
            common.num_actual_tokens = scene["tokens"]
            metadata[group["prefix"]] = group["builder"].build(
                0,
                common,
                num_reqs_actual=active,
                block_size=group["spec"].block_size,
                common_ratio_to_sas_metadata=common_cache,
                prefill_ratio_to_sas_metadata=prefill_cache,
                decode_ratio_to_sas_metadata=decode_cache,
            )
        return metadata

    def call(layer, scene, adapted, captured=None):
        value = torch.empty_like(scene["hidden"])
        restore(scene)

        def forward():
            _native_attention_half(layer.self_attn.dsa_attn, scene["hidden"], scene["positions"], value)

        with set_ascend_forward_context(
            scene["metadata"] if captured is None else captured,
            config,
            num_tokens=scene["tokens"],
            num_actual_tokens=scene["actual_tokens"],
        ):
            if adapted:
                get_forward_context().additional_kwargs[TND_PADDED_REDISPATCH] = True
            forward()
            if captured is not None:
                torch.npu.synchronize()
                restore(scene)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    forward()
                restore(scene)
                graph.replay()
        torch.npu.synchronize()
        state = {"output": value[: scene["actual_tokens"]].cpu()}
        for name, group in scene["groups"].items():
            for index, view in enumerate(group["views"]):
                state[f"{name}.{index}"] = view.cpu()
        return state

    with native_session(config, 0), torch.inference_mode():
        for layer_index, active in ((1, 7), (3, 4), (2, 7)):
            if layer_index not in args.layers:
                continue
            layer, _ = make_layer(config, model, device, layer_index)
            kwargs = {"query_tokens": 6, "history_lengths": histories[:active], "table_history": 131500}
            padded = make_fixture(
                config, layer.self_attn, active, 131500, 701, device, request_capacity=16, token_capacity=96, **kwargs
            )
            captured = capture_metadata(padded, active)
            assert all(m.num_decode_tokens == 96 for m in captured.values())
            expected = call(layer, padded, False, captured=captured)
            actual = call(layer, padded, True)
            comparisons = {name: compare_tensor(value, expected[name], 0, 0) for name, value in actual.items()}
            guards = guard_checks(padded)
            unchanged = all(
                m.num_actual_tokens == active * 6 and m.num_decode_tokens == active * 6
                for m in padded["metadata"].values()
            )
            row = {
                "layer": layer_index,
                "compress_ratio": layer.self_attn.compress_ratio,
                "active_requests": active,
                "query_rows": active * 6,
                "capacity": 96,
                "reference": "Native原builder的96行捕获及重放，device Q/L含实际请求与padding",
                "comparisons": comparisons,
                "guards": guards,
                "original_metadata_preserved": unchanged,
            }
            report["cases"].append(row)
            save()
            assert unchanged
            assert all(v["status"] == "PASS" for v in guards.values()), guards
            assert all(v["status"] == "PASS" for v in comparisons.values()), comparisons
            del expected, actual, captured, padded, layer
        report["status"] = "PASS"
        save()
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
