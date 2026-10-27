# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the private package before DSL imports.
"""同一WKV/Norm/RoPE输入，区分两个缺失的BF16发布边界。"""

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

source = bind()
from kvnorm_kernels import kvnorm_arm

from vllm_ascend.utils import enable_custom_op

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "scope": "isolated same-input KV Norm/RoPE", "arms": {}}
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
        "first_coordinates": diff.nonzero()[:8].tolist(),
    }


def native_boundary(projected, gamma, cos, sin):
    normalized = torch_npu.npu_rms_norm(projected, gamma, epsilon=1e-6)[0]
    rotated = normalized.clone().reshape(-1, 1, 512)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        rotated.unsqueeze(1), cos, sin, rotary_mode="interleave", partial_slice=[448, 512]
    )
    return normalized, rotated.reshape(-1, 512)


def run_arm(acc, gamma, cos, sin, swap, native):
    tokens = cos.shape[0]
    owner = torch.full((tokens + 16, 512), -37, dtype=torch.bfloat16, device="npu")
    actual = owner[8 : 8 + tokens]
    kvnorm_arm(acc, gamma, cos, sin, swap, actual, NATIVE_BOUNDARIES=native)
    torch.npu.synchronize()
    eager = actual.cpu().clone()
    graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(graph, stream=stream):
        kvnorm_arm(acc, gamma, cos, sin, swap, actual, NATIVE_BOUNDARIES=native)
    actual.fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()
    guard = owner.cpu()
    row = {
        "graph_eager": metric(actual, eager),
        "guard_ok": bool((guard[:8] == -37).all() and (guard[8 + tokens :] == -37).all()),
    }
    assert row["graph_eager"]["bit_equal"] and row["guard_ok"]
    return actual.cpu(), row


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true" and enable_custom_op()
    record["native_determinism"] = {
        "torch": torch.are_deterministic_algorithms_enabled(),
        "npu_level": torch_npu._C._npu_get_deterministic_level(),
        "HCCL_DETERMINISTIC": os.environ["HCCL_DETERMINISTIC"],
    }
    assert record["native_determinism"] == {"torch": True, "npu_level": 1, "HCCL_DETERMINISTIC": "true"}
    record["source_sha256"] = {
        name: hashlib.sha256((source / "deepseek_v4_flash_csa_spmd_tnd" / name).read_bytes()).hexdigest()
        for name in ("kv_stages.py", "kv_native_norm_stages.py")
    }
    record["native_source"] = os.environ["SOURCE_ROOT"]
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    reference = Path(os.environ["KVNORM_PROBE_REFERENCE"])
    record["reference"] = str(reference)
    data = torch.load(reference, map_location="cpu", weights_only=True)
    acc = data["kv_accumulator"].to("npu")
    projected = data["wkv_projection_bf16_native"].to("npu")
    gamma = data["kv_gamma"].to("npu")
    cos, sin = (data["kv_rope_" + key].to("npu") for key in ("cos", "sin"))
    tokens = projected.shape[0]
    assert tokens == 24 and acc.shape == (32, 512)
    assert metric(acc[:tokens].bfloat16(), projected)["bit_equal"], "WKV prerequisite first difference"
    normalized, gold = native_boundary(projected, gamma, cos, sin)
    record["native_repeat"] = {
        "norm": metric(normalized, data["kv_normalized_native"]),
        "rope": metric(gold, data["original_kv_norm_rope_native"]),
    }
    assert all(row["bit_equal"] for row in record["native_repeat"].values())
    cosine = cos.reshape(tokens, 64)
    signs = torch.where(torch.arange(64) % 2 == 0, -1.0, 1.0).to("npu")
    signed = sin.reshape(tokens, 64) * signs
    swap = ((torch.arange(64) // 2) * 2 + 1 - torch.arange(64) % 2).repeat(tokens, 1).int().to("npu")
    pypto.torch.init()
    for name, values, boundaries in (
        ("nal", acc, False),
        ("wkv_bf16", acc.bfloat16().float(), False),
        ("wkv_norm_bf16", acc, True),
    ):
        actual, row = run_arm(values, gamma, cosine, signed, swap, boundaries)
        row.update(norm_nope=metric(actual[:, :448], normalized[:, :448]), norm_rope=metric(actual, gold))
        if name == "nal":
            row["production_repeat"] = metric(actual, data["original_kv_norm_rope_actual"])
            assert row["production_repeat"]["bit_equal"]
        record["arms"][name] = row
        if boundaries:
            assert row["norm_rope"]["bit_equal"], "KV Norm/RoPE candidate first difference"
    record["shape_cases"] = []
    for count in (1, 17, 32, 33, 48, 63):
        # Isolate Norm/RoPE; these cases do not qualify WKV projection for these M.
        padded = (count + 15) // 16 * 16
        values = acc.cpu()[:tokens].repeat(3, 1)[:padded].contiguous().to("npu")
        c = cos.cpu().repeat(3, 1, 1, 1)[:count].to("npu")
        s = sin.cpu().repeat(3, 1, 1, 1)[:count].to("npu")
        norm, gold = native_boundary(values[:count].bfloat16(), gamma, c, s)
        signed = s.reshape(count, 64) * signs
        swap = ((torch.arange(64) // 2) * 2 + 1 - torch.arange(64) % 2).repeat(count, 1).int().to("npu")
        actual, row = run_arm(values, gamma, c.reshape(count, 64), signed, swap, True)
        row.update(tokens=count, norm_nope=metric(actual[:, :448], norm[:, :448]), norm_rope=metric(actual, gold))
        record["shape_cases"].append(row)
        if not row["norm_rope"]["bit_equal"]:
            record["first_difference"] = {"tokens": count, "stage": "KV Norm/RoPE"}
            raise AssertionError("KV Norm/RoPE shape first difference")
    record.update(complete=True, **{"pass": True})
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
