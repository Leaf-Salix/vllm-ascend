# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind independent DSL source before imports.
"""同一Native真实投影保存件，隔离Indexer反量化与RoPE前BF16舍入。"""

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
from idxquery_kernels import idxquery_dequant_arm, idxquery_rope_arm

from vllm_ascend.utils import enable_custom_op

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "scope": "saved T24 indexer dequant/RoPE, not new GEMM shapes", "arms": {}}
start = time.monotonic()


def metric(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == torch.bfloat16
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    diff = actual.view(torch.int16) != expected.view(torch.int16)
    return {
        "bit_equal": not bool(diff.any()),
        "different_elements": int(diff.sum()),
        "max_abs": float((actual.float() - expected.float()).abs().max()),
        "first_coordinates": diff.nonzero()[:8].tolist(),
    }


def run_arm(acc, row_scale, weight, cos, signed, mode):
    tokens = row_scale.shape[0]
    capacity = (tokens + 15) // 16 * 16
    owner = torch.full((capacity + 16, 8192), -37, dtype=torch.bfloat16, device="npu")
    actual = owner[8 : 8 + capacity]
    idxquery_rope_arm(acc, row_scale, weight, cos, signed, actual, MODE=mode)
    torch.npu.synchronize()
    eager = actual[:tokens].cpu().clone()
    graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(graph, stream=stream):
        idxquery_rope_arm(acc, row_scale, weight, cos, signed, actual, MODE=mode)
    actual.fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()
    row = {
        "graph_eager": metric(actual[:tokens], eager),
        "guard_ok": bool((owner[:8].cpu() == -37).all() and (owner[8 + capacity :].cpu() == -37).all()),
    }
    assert row["graph_eager"]["bit_equal"] and row["guard_ok"]
    return actual[:tokens].cpu(), row


def native_rope(projected, cos, sin):
    result = projected.clone().reshape(-1, 64, 128)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        result.unsqueeze(1), cos, sin, rotary_mode="interleave", partial_slice=[64, 128]
    )
    torch.npu.synchronize()
    return result.reshape(-1, 8192)


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true" and enable_custom_op()
    reference = Path(os.environ["IDXQUERY_PROBE_REFERENCE"])
    data = torch.load(reference, map_location="cpu", weights_only=True)
    acc = data["observed_idx_query_acc"].to("npu")
    row_scale = data["qr_native_scale"].reshape(24, 1).to("npu")
    weight_cpu = data["indexer_query_weight_scale_prepared"].reshape(8192)
    assert torch.equal(weight_cpu, data["indexer_query_weight_scale_native"].reshape(8192).float())
    weight = weight_cpu.to("npu")
    projected = data["native_indexer_query_stages"]["projection"].reshape(24, 8192).to("npu")
    cos, sin = (data["kv_rope_" + name].to("npu") for name in ("cos", "sin"))
    assert acc.shape == (64, 8192) and projected.dtype == torch.bfloat16
    record["reference"] = str(reference)
    record["weight_scale_lossless"] = True
    record["source_sha256"] = {
        name: hashlib.sha256((source / "deepseek_v4_flash_csa_spmd_tnd" / name).read_bytes()).hexdigest()
        for name in ("indexer_query_stages.py", "indexer_native_query_stages.py")
    }
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    gold = native_rope(projected, cos, sin)
    record["native_repeat"] = metric(gold, data["native_indexer_query_stages"]["rope"].reshape(24, 8192))
    assert record["native_repeat"]["bit_equal"]
    signs = torch.where(torch.arange(64) % 2 == 0, -1.0, 1.0).to("npu")
    cosine, signed = cos.reshape(24, 64), sin.reshape(24, 64) * signs
    pypto.torch.init()
    record["dequant"] = {}
    for combined in (False, True):
        owner = torch.full((40, 8192), -37, dtype=torch.bfloat16, device="npu")
        value = owner[8:32]
        idxquery_dequant_arm(acc, row_scale, weight, value, COMBINED=combined)
        torch.npu.synchronize()
        eager = value.cpu().clone()
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            idxquery_dequant_arm(acc, row_scale, weight, value, COMBINED=combined)
        value.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        row = {
            "projection": metric(value, projected),
            "graph_eager": metric(value, eager),
            "guard_ok": bool((owner[:8].cpu() == -37).all() and (owner[32:].cpu() == -37).all()),
        }
        record["dequant"][str(combined)] = row
        assert row["graph_eager"]["bit_equal"] and row["guard_ok"]
        if combined:
            assert row["projection"]["bit_equal"]
    for mode in range(3):
        value, row = run_arm(acc, row_scale, weight, cosine, signed, mode)
        row["native_rope"] = metric(value, gold)
        if mode == 0:
            row["production_repeat"] = metric(value, data["indexer_query_rope_actual"].reshape(24, 8192))
            assert row["production_repeat"]["bit_equal"]
        record["arms"][str(mode)] = row
        if mode == 2:
            assert row["native_rope"]["bit_equal"], "Native-boundary indexer RoPE first difference"
    record["shape_cases"] = []
    for tokens in (1, 17, 23, 32, 33, 48, 63):
        padded = (tokens + 63) // 64 * 64
        values = acc.cpu()[:24].repeat(6, 1)[:padded].contiguous().to("npu")
        scales = row_scale.cpu().repeat(3, 1)[:tokens].contiguous().to("npu")
        c, s = (value.cpu().repeat(3, 1, 1, 1)[:tokens].contiguous().to("npu") for value in (cos, sin))
        p = projected.cpu().repeat(3, 1)[:tokens].contiguous().to("npu")
        reference_rope = native_rope(p, c, s)
        value, row = run_arm(values, scales, weight, c.reshape(tokens, 64), s.reshape(tokens, 64) * signs, 2)
        row.update(tokens=tokens, native_rope=metric(value, reference_rope))
        record["shape_cases"].append(row)
        assert row["native_rope"]["bit_equal"], "saved-row tail RoPE first difference"
    record.update(complete=True, **{"pass": True})
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
