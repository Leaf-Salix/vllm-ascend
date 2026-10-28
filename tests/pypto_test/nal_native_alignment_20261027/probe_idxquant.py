# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind private operator source before importing DSL modules.
"""保存的同一Hadamard accumulator，隔离两层BF16边界并复现实际Native量化。"""

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
from idxquant_kernels import idxquant_arm

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "scope": "saved T24 Hadamard to query quant, non-fused path", "arms": {}}
start = time.monotonic()


class FirstDifference(Exception):
    pass


def metric(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    integer = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}.get(actual.dtype)
    diff = actual.view(integer) != expected.view(integer) if integer else actual != expected
    return {
        "bit_equal": not bool(diff.any()),
        "different_elements": int(diff.sum()),
        "max_abs": float((actual.float() - expected.float()).abs().max()),
        "first_coordinates": diff.nonzero()[:8].tolist(),
    }


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    reference = Path(os.environ["IDXQUANT_PROBE_REFERENCE"])
    data = torch.load(reference, map_location="cpu", weights_only=True)
    record["reference"] = str(reference)
    record["source_sha256"] = {
        str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__).with_name("idxquant_kernels.py"),
            source / "deepseek_v4_flash_csa_spmd_tnd/indexer_query_stages.py",
        )
    }
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    raw = data["observed_idx_query_had"].to("npu")
    assert raw.shape == (1536, 128) and raw.dtype == torch.float32
    native_stages, qli = data["native_indexer_query_stages"], data["native_qli_inputs"]
    record["input_hadamard_bf16"] = metric(raw.bfloat16(), native_stages["hadamard_linear"])
    assert record["input_hadamard_bf16"]["bit_equal"]
    scaled = native_stages["scaled"].reshape(24, 64, 128).contiguous().to("npu")
    snapshot = scaled.cpu().clone()
    native_query, native_scale = torch_npu.npu_dynamic_quant(scaled, dst_type=torch.int8)
    torch.npu.synchronize()
    record["native_repeat"] = {
        "query": metric(native_query, qli["query"]),
        "scale_fp16": metric(native_scale.half(), qli["query_dequant_scale"]),
        "input_unchanged": metric(scaled, snapshot),
    }
    assert all(row["bit_equal"] for row in record["native_repeat"].values())
    assert native_scale.dtype == torch.float32
    payload = {
        "raw_hadamard": raw.cpu(),
        "native_scaled": scaled.cpu(),
        "native_query": native_query.cpu(),
        "native_scale_fp32": native_scale.cpu(),
    }
    gold = native_query.reshape(1536, 128).cpu()
    gold_scale = native_scale.reshape(1536, 1).cpu()
    pypto.torch.init()
    for mode in range(3):
        query_owner = torch.full((1552, 128), -37, dtype=torch.int8, device="npu")
        scale_owner = torch.full((1552, 1), -37.0, dtype=torch.float32, device="npu")
        query, scale = query_owner[8:-8], scale_owner[8:-8]
        idxquant_arm(raw, query, scale, MODE=mode)
        torch.npu.synchronize()
        eager_q, eager_s = query.cpu().clone(), scale.cpu().clone()
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            idxquant_arm(raw, query, scale, MODE=mode)
        query.fill_(-123)
        scale.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        row = {
            "query": metric(query, gold),
            "scale_fp16": metric(scale.half(), gold_scale.half()),
            "scale_fp32_diagnostic": metric(scale, gold_scale),
            "query_graph_eager": metric(query, eager_q),
            "scale_graph_eager": metric(scale, eager_s),
            "guard_ok": all(
                bool((value[:8].cpu() == -37).all() and (value[-8:].cpu() == -37).all())
                for value in (query_owner, scale_owner)
            ),
        }
        record["arms"][str(mode)] = row
        payload["mode_" + str(mode) + "_query"] = query.cpu()
        payload["mode_" + str(mode) + "_scale"] = scale.cpu()
        assert row["query_graph_eager"]["bit_equal"] and row["scale_graph_eager"]["bit_equal"] and row["guard_ok"]
        if mode == 0:
            row["production_query_repeat"] = metric(query, data["observed_idx_query_i8"])
            row["production_scale_repeat"] = metric(scale, data["observed_idx_query_scale"])
            assert row["production_query_repeat"]["bit_equal"] and row["production_scale_repeat"]["bit_equal"]
        if mode == 2:
            for name in ("query", "scale_fp16"):
                if not row[name]["bit_equal"]:
                    record["first_difference"] = name
                    raise FirstDifference
    record["pass"] = True
    record["complete"] = True
except FirstDifference:
    record["complete"] = True
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    if "payload" in globals():
        torch.save(payload, out / "quantization.pt")
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
