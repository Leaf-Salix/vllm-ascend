# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind private source before importing the diagnostic DSL.
"""同一次第2层T240输入；先复现原系数，再隔离投影K累加顺序。"""

import hashlib
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

source = bind()
from weight240_kernels import WEIGHTS_SCALE, weight240_arm

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "arms": {}, "physical_device": int(os.environ["TASK_DEVICE"])}
payload = {}
start = time.monotonic()


def metric(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    integer = torch.int32 if actual.dtype == torch.float32 else torch.int16
    different = actual.view(integer) != expected.view(integer)
    return {
        "bit_equal": not bool(different.any()),
        "different_elements": int(different.sum()),
        "max_abs": float((actual.double() - expected.double()).abs().max()),
        "first_coordinates": different.nonzero()[:8].tolist(),
    }


try:
    reference = Path(os.environ["WEIGHT240_PROBE_REFERENCE"])
    previous = json.loads(reference.with_name("result.json").read_text())
    assert previous["complete"] and previous["first_difference"] == "qli_weights_fp16"
    assert previous["layer_index"] == 2 and previous["tokens"] == 240
    assert all(row["bit_equal"] for name, row in previous["precision"].items() if name != "qli_weights_fp16")
    for gate in ("input_equivalence", "observation_gate", "native_qli_hook_control"):
        assert all(row["bit_equal"] for row in previous[gate].values()), gate
    saved = torch.load(reference, map_location="cpu", weights_only=True)
    native = saved["native_indexer_weight_stages"]
    assert native["input"].shape == (240, 4096) and native["input"].dtype == torch.bfloat16
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    x = native["input"].to("npu")
    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    key = "layers.2.attn.indexer.weights_proj.weight"
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    with safe_open(checkpoint / index[key], framework="pt", device="cpu") as reader:
        stored = reader.get_tensor(key)
    assert stored.shape == (64, 4096) and stored.dtype == torch.bfloat16
    weight = torch_npu.npu_format_cast(stored.to("npu"), 29)
    projected = torch.nn.functional.linear(x, weight)
    scaled = projected * WEIGHTS_SCALE
    torch.npu.synchronize()
    record["native_repeat"] = {
        "projection": metric(projected, native["projection"]),
        "scaled": metric(scaled, native["scaled"]),
        "consumer": metric(scaled.half(), native["consumer"]),
    }
    assert all(row["bit_equal"] for row in record["native_repeat"].values())
    record.update(
        reference=str(reference),
        layer_index=2,
        tokens=240,
        sdk_release=os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"],
        weight_sha256=hashlib.sha256(stored.view(torch.uint8).numpy().tobytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(__file__).with_name("weight240_kernels.py").read_bytes()).hexdigest(),
        scope="same input/weight/BF16 boundaries; isolate partial reduction vs continuous K accumulation",
    )
    pypto.torch.init()
    for mode in (0, 1, 2, 3):
        owners = [torch.full((rows + 16, 64), -37.0, dtype=torch.float32, device="npu") for rows in (960, 240, 240)]
        partials, values, raw = (owner[8:-8] for owner in owners)
        weight240_arm(x, weight, partials, values, raw, MODE=mode)
        torch.npu.synchronize()
        eager = [value.cpu().clone() for value in (values, raw)]
        payload[f"raw_{mode}"], payload[f"scaled_{mode}"] = eager[1], eager[0]
        row = {
            "projection_bf16": metric(eager[1].bfloat16(), native["projection"]),
            "scaled_bf16": metric(eager[0].bfloat16(), native["scaled"]),
            "consumer_fp16": metric(eager[0].half(), native["consumer"]),
        }
        record["arms"][str(mode)] = row
        if mode == 0:
            row["production_repeat"] = metric(eager[0], saved["observed_head_weights"][:240])
            assert row["production_repeat"]["bit_equal"], "baseline changed producer"
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            weight240_arm(x, weight, partials, values, raw, MODE=mode)
        row["replays"] = []
        for _ in range(3):
            values.fill_(float("nan"))
            raw.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            comparisons = [metric(value, gold) for value, gold in zip((values, raw), eager, strict=True)]
            row["replays"].append(comparisons)
            assert all(value["bit_equal"] for value in comparisons)
        row["guards"] = all(
            bool((owner[:8].cpu() == -37).all() and (owner[-8:].cpu() == -37).all()) for owner in owners
        )
        assert row["guards"]
        assert metric(x, native["input"])["bit_equal"]
        restored_weight = torch_npu.npu_format_cast(weight, 2).cpu()
        assert metric(restored_weight, stored)["bit_equal"]
        if mode and all(row[name]["bit_equal"] for name in ("projection_bf16", "scaled_bf16", "consumer_fp16")):
            record.update(**{"pass": True}, selected_mode=mode)
            break
    record["complete"] = True
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    torch.save(payload, out / "projection.pt")
