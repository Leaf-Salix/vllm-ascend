# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind private source before importing independent DSL modules.
"""同一真实权重输入，先对拍投影，再隔离两处BF16边界；首差即停。"""

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
from weightproj_kernels import WEIGHTS_SCALE, weightproj_arm

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "scope": "saved T24 weights projection/rounding only", "arms": {}}
start = time.monotonic()


class FirstDifference(Exception):
    pass


def metric(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    integer = torch.int32 if actual.dtype == torch.float32 else torch.int16
    diff = actual.view(integer) != expected.view(integer)
    return {
        "bit_equal": not bool(diff.any()),
        "different_elements": int(diff.sum()),
        "max_abs": float((actual.double() - expected.double()).abs().max()),
        "first_coordinates": diff.nonzero()[:8].tolist(),
    }


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    reference = Path(os.environ["WEIGHTPROJ_PROBE_REFERENCE"])
    saved = torch.load(reference, map_location="cpu", weights_only=True)
    native = saved["native_indexer_weight_stages"]
    x = native["input"].to("npu")
    assert x.shape == (24, 4096) and x.dtype == torch.bfloat16
    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    key = "layers.4.attn.indexer.weights_proj.weight"
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    with safe_open(checkpoint / index[key], framework="pt", device="cpu") as reader:
        stored = reader.get_tensor(key)
    assert stored.shape == (64, 4096) and stored.dtype == torch.bfloat16
    record["weight"] = {
        "key": key,
        "shard": index[key],
        "sha256": hashlib.sha256(stored.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }
    weight = torch_npu.npu_format_cast(stored.to("npu"), 29)
    projected = torch.nn.functional.linear(x, weight)
    scaled = projected * WEIGHTS_SCALE
    torch.npu.synchronize()
    record["native_repeat"] = {
        "projection": metric(projected, native["projection"]),
        "scaled": metric(scaled, native["scaled"]),
        "consumer": metric(scaled.half(), native["consumer"]),
    }
    assert all(row["bit_equal"] for row in record["native_repeat"].values()), (
        "Native replay must reproduce actual weight hooks"
    )
    record["reference"] = str(reference)
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    record["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__).with_name("weightproj_kernels.py"),
            source / "deepseek_v4_flash_csa_spmd_tnd/indexer_query_stages.py",
        )
    }
    payload = {"native_projection": projected.cpu(), "native_scaled": scaled.cpu()}
    pypto.torch.init()
    for mode in (0, 1):
        owners = [torch.full((rows + 16, 64), -37.0, dtype=torch.float32, device="npu") for rows in (128, 32, 32)]
        partials, values, raw = (value[8:-8] for value in owners)
        weightproj_arm(x, weight, partials, values, raw, MODE=mode)
        torch.npu.synchronize()
        eager = [value[:24].cpu().clone() for value in (values, raw)]
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            weightproj_arm(x, weight, partials, values, raw, MODE=mode)
        for value in (partials, values, raw):
            value.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        row = {
            "projection_bf16": metric(raw[:24].bfloat16(), native["projection"]),
            "consumer_fp16": metric(values[:24].half(), native["consumer"]),
            "graph_eager": [metric(value[:24], expected) for value, expected in zip((values, raw), eager)],
            "guard_ok": all(
                bool((owner[:8].cpu() == -37).all() and (owner[-8:].cpu() == -37).all()) for owner in owners
            ),
        }
        payload["raw_" + str(mode)] = raw[:24].cpu()
        payload["partial_" + str(mode)] = partials.cpu()
        payload["scaled_" + str(mode)] = values[:24].cpu()
        record["arms"][str(mode)] = row
        assert row["guard_ok"] and all(item["bit_equal"] for item in row["graph_eager"])
        if mode == 0:
            row["production_repeat"] = metric(values[:24], saved["observed_head_weights"][:24])
            assert row["production_repeat"]["bit_equal"]
        if not row["projection_bf16"]["bit_equal"]:
            record["first_difference"] = "projection_bf16"
            raise FirstDifference
        if mode == 1:
            row["scaled_bf16"] = metric(values[:24].bfloat16(), native["scaled"])
            for name in ("scaled_bf16", "consumer_fp16"):
                if not row[name]["bit_equal"]:
                    record["first_difference"] = name
                    raise FirstDifference
    record.update(complete=True, **{"pass": True})
except FirstDifference:
    record["complete"] = True
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    if "payload" in globals():
        torch.save(payload, out / "projection.pt")
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
