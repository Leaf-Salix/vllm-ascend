# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind private DSL source before imports.
"""复现Q首差，先核对反量化边界，再逐项检查Norm/RoPE候选。"""

import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import pypto.torch
import torch
import torch_npu
from bind_package import bind

source = bind()
from qoutput_kernels import qdequant_arm, qoutput_arm

from vllm_ascend.utils import enable_custom_op

out = Path(sys.argv[1])
record = {
    "complete": False,
    "pass": False,
    "scope": "same saved Q accumulator, not new GEMM or whole-model tokens",
    "arms": {},
}
start = time.monotonic()


def metric(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    kind = torch.int16 if actual.element_size() == 2 else torch.int32
    diff = actual.view(kind) != expected.view(kind)
    return {
        "bit_equal": not bool(diff.any()),
        "different_elements": int(diff.sum()),
        "max_abs": float((actual.double() - expected.double()).abs().max()),
        "first_coordinates": diff.nonzero()[:6].tolist(),
    }


def native_norm_rope(projection, cos, sin):
    normalized = operator.apply_dsa_q_rms(projection.reshape(-1, 64, 512), 1e-6)
    result = normalized.clone()
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        result.unsqueeze(1), cos, sin, rotary_mode="interleave", partial_slice=[448, 512]
    )
    return normalized, result.reshape(-1, 32768)


def captured(call, tokens):
    storage = torch.full((tokens + 16, 32768), -37, dtype=torch.bfloat16, device="npu")
    result = storage[8 : 8 + tokens]
    call(result)
    torch.npu.synchronize()
    eager = result.cpu().clone()
    graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(graph, stream=stream):
        call(result)
    result.fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()
    guard = storage.cpu()
    row = {
        "graph_eager": metric(result, eager),
        "guard_ok": bool((guard[:8] == -37).all() and (guard[8 + tokens :] == -37).all()),
    }
    assert row["graph_eager"]["bit_equal"] and row["guard_ok"]
    return result.cpu(), row


def rope_inputs(cos, sin):
    tokens = cos.shape[0]
    signs = torch.where(torch.arange(64) % 2 == 0, -1.0, 1.0).to("npu")
    signed = sin.reshape(tokens, 64) * signs
    swap = ((torch.arange(64) // 2) * 2 + 1 - torch.arange(64) % 2).repeat(tokens, 1).int().to("npu")
    return cos.reshape(tokens, 64), signed, swap


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true" and enable_custom_op()
    # Initialize Native ops before device_op; match the working model-load order.
    importlib.import_module("vllm_ascend.ops")
    operator = importlib.import_module("vllm_ascend.attention.dsa_v1").DeviceOperator
    record["native_operator"] = operator.apply_dsa_q_rms.__qualname__
    record["native_determinism"] = {
        "torch": torch.are_deterministic_algorithms_enabled(),
        "npu_level": torch_npu._C._npu_get_deterministic_level(),
        "HCCL_DETERMINISTIC": os.environ["HCCL_DETERMINISTIC"],
    }
    assert record["native_determinism"] == {"torch": True, "npu_level": 1, "HCCL_DETERMINISTIC": "true"}
    record["source_sha256"] = {
        name: hashlib.sha256((source / "deepseek_v4_flash_csa_spmd_tnd" / name).read_bytes()).hexdigest()
        for name in ("query_projection_stages.py", "q_native_output_stages.py")
    }
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    record["native_source"] = os.environ["SOURCE_ROOT"]
    reference = Path(os.environ["QOUTPUT_PROBE_REFERENCE"])
    record["reference"] = str(reference)
    data = torch.load(reference, map_location="cpu", weights_only=True)
    accumulator = data["q_accumulator"].to("npu")
    native_projection = data["q_projection_native"].to("npu")
    tokens = native_projection.shape[0]
    assert tokens == 24 and accumulator.shape == (128, 32768)
    weight = data["q_weight_scale_prepared"].reshape(32768).to("npu")
    native_weight = data["q_weight_scale_native"].reshape(32768)
    assert native_weight.dtype in (torch.bfloat16, torch.float32)
    record["weight_scale_native_dtype"] = str(native_weight.dtype)
    # prepare_weights widens Native BF16 channel scales to FP32 without changing values.
    record["weight_scale_roundtrip"] = metric(native_weight.float().to(native_weight.dtype), native_weight)
    assert record["weight_scale_roundtrip"]["bit_equal"]
    record["weight_scale_repeat"] = metric(weight, native_weight.float())
    assert record["weight_scale_repeat"]["bit_equal"]
    scale = data["qr_native_scale"].to("npu")
    cos, sin = (data["kv_rope_" + key].to("npu") for key in ("cos", "sin"))
    normalized, gold = native_norm_rope(native_projection, cos, sin)
    record["native_repeat"] = {
        "norm": metric(normalized, data["q_normalized_native"]),
        "rope": metric(gold, data["q_norm_rope_native"]),
    }
    assert all(row["bit_equal"] for row in record["native_repeat"].values())
    pypto.torch.init()
    record["dequant"] = {}
    for combined in (False, True):
        actual, row = captured(partial(qdequant_arm, accumulator, weight, scale, COMBINED=combined), tokens)
        row["native_bf16"] = metric(actual.reshape(tokens, 64, 512), native_projection)
        record["dequant"][str(combined)] = row
        if combined:
            assert row["native_bf16"]["bit_equal"], "Q dequant first difference"
    cosine, signed, swap = rope_inputs(cos, sin)
    for mode in range(6):
        actual, row = captured(
            partial(qoutput_arm, accumulator, weight, scale, cosine, signed, swap, MODE=mode), tokens
        )
        row["native_output"] = metric(actual, gold)
        row["norm_nope"] = metric(actual.reshape(tokens, 64, 512)[:, :, :448], normalized[:, :, :448])
        if mode == 0:
            row["production_repeat"] = metric(actual, data["q_norm_rope_actual"])
            assert row["production_repeat"]["bit_equal"]
        record["arms"][str(mode)] = row
        if mode == 5:
            assert row["native_output"]["bit_equal"], "Q candidate first difference"
    record["shape_cases"] = []
    for count in (1, 17, 23, 32, 33, 48, 63):
        # These are dequant/Norm/RoPE tests using repeated saved rows, not new GEMM qualification.
        acc = accumulator.cpu()[:tokens].repeat(6, 1)[:128].to("npu")
        rows = data["qr_native_scale"].repeat(6, 1)[: (count + 7) // 8 * 8].to("npu")
        projected = data["q_projection_native"].repeat(3, 1, 1)[:count].to("npu")
        c = cos.cpu().repeat(3, 1, 1, 1)[:count].to("npu")
        s = sin.cpu().repeat(3, 1, 1, 1)[:count].to("npu")
        norm, gold = native_norm_rope(projected, c, s)
        cosine, signed, swap = rope_inputs(c, s)
        actual, row = captured(partial(qoutput_arm, acc, weight, rows, cosine, signed, swap, MODE=5), count)
        row.update(tokens=count, native_output=metric(actual, gold))
        record["shape_cases"].append(row)
        if not row["native_output"]["bit_equal"]:
            record["first_difference"] = {"tokens": count, "stage": "Q dequant/Norm/RoPE"}
            raise AssertionError("Q shape first difference")
    record.update(complete=True, **{"pass": True})
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
