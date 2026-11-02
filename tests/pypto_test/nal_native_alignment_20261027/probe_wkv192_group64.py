# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the same private package as the production observation.
"""T192 WKV N64列组假设：先复现首差，再对Native逐bit验收；正式源码不改。"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import pypto.torch
import torch
import torch_npu
from bind_package import bind
from safetensors import safe_open

bind()
from wkv192_kernels import wkv192_arm

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "scope": "T192 WKV N64 grouping single-factor FULL", "arms": {}}
start = time.monotonic()
payload = {}


def metric(a, b):
    a, b = a.cpu().contiguous(), b.cpu().contiguous()
    assert a.shape == b.shape and a.dtype == b.dtype and torch.isfinite(a).all() and torch.isfinite(b).all()
    different = a.view(torch.int16 if a.element_size() == 2 else torch.int32) != b.view(
        torch.int16 if b.element_size() == 2 else torch.int32
    )
    return {
        "bit_equal": not bool(different.any()),
        "different_elements": int(different.sum()),
        "max_abs": float((a.double() - b.double()).abs().max()),
    }


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    previous = Path(os.environ["WKV192_PROBE_REFERENCE"])
    data = torch.load(previous, map_location="cpu", weights_only=True)
    x = data["input_norm_native"].to("npu")
    assert x.shape == (192, 4096) and x.dtype == torch.bfloat16
    root = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    key = "layers.4.attn.wkv.weight"
    index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    with safe_open(root / index[key], framework="pt", device="cpu") as reader:
        weight_cpu = reader.get_tensor(key)
    weight = torch_npu.npu_format_cast(weight_cpu.to("npu"), 29)
    gold = torch.nn.functional.linear(x, weight)
    record["native_saved"] = metric(gold, data["wkv_projection_bf16_native"])
    assert record["native_saved"]["bit_equal"], "canonical linear must reproduce actual Native CV wrapper"
    record.update(reference=str(previous), checkpoint_key=key, sdk_release=os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"])
    pypto.torch.init()
    owner = torch.full((224, 512), -37.0, device="npu")
    projected = owner[16:208]
    for mode, name in ((0, "original_n128_natural"), (3, "n32_group64_order_hypothesis")):

        def run(mode=mode):
            wkv192_arm(x, weight, projected, MODE=mode)

        run()
        torch.npu.synchronize()
        eager = projected.cpu().clone()
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            run()
        projected.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        row = {
            "native_bf16": metric(projected.bfloat16(), gold),
            "full_eager": metric(projected, eager),
            "guard_ok": bool((owner[:16].cpu() == -37).all() and (owner[208:].cpu() == -37).all()),
            "x_readonly": metric(x, data["input_norm_native"]),
            "weight_readonly": metric(weight, weight_cpu),
        }
        if mode == 0:
            row["saved_actual_bf16"] = metric(projected.bfloat16(), data["wkv_projection_bf16_actual"])
            row["saved_actual_fp32"] = metric(projected, data["kv_accumulator"][:192])
        record["arms"][name] = row
        payload[name] = projected.cpu().clone()
        if mode == 0:
            assert row["saved_actual_bf16"]["bit_equal"] and row["saved_actual_fp32"]["bit_equal"], (
                "isolated baseline must reproduce the observed BF16 and FP32 boundary"
            )
        assert row["full_eager"]["bit_equal"] and row["guard_ok"]
        assert row["x_readonly"]["bit_equal"] and row["weight_readonly"]["bit_equal"]
        if mode == 3:
            assert row["native_bf16"]["bit_equal"], "candidate first difference"
    record["heldout_inputs"] = {}
    for seed, amplitude in ((2048, 0.05), (4096, 0.5), (8191, 2.0), (65536, 16.0)):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        input_cpu = (torch.randn((192, 4096), generator=generator) * amplitude).bfloat16()
        other = input_cpu.to("npu")
        expected = torch.nn.functional.linear(other, weight)
        projected.fill_(float("nan"))
        wkv192_arm(other, weight, projected, MODE=3)
        torch.npu.synchronize()
        eager = projected.cpu().clone()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            wkv192_arm(other, weight, projected, MODE=3)
        projected.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        row = {
            "amplitude": amplitude,
            "native_bf16": metric(projected.bfloat16(), expected),
            "full_eager": metric(projected, eager),
            "input_readonly": metric(other, input_cpu),
            "weight_readonly": metric(weight, weight_cpu),
            "guards": bool((owner[:16].cpu() == -37).all() and (owner[208:].cpu() == -37).all()),
        }
        payload[f"seed_{seed}"] = {
            "input": input_cpu.clone(),
            "native": expected.cpu().clone(),
            "eager": eager,
            "full": projected.cpu().clone(),
        }
        record["heldout_inputs"][str(seed)] = row
        assert row["native_bf16"]["bit_equal"], "heldout Native first difference"
        assert row["full_eager"]["bit_equal"] and row["input_readonly"]["bit_equal"]
        assert row["weight_readonly"]["bit_equal"] and row["guards"]
    record.update(complete=True, **{"pass": True})
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    torch.save(payload, out / "wkv192-arms.pt")
