# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind independent DSL source before imports.
"""复用主state首差的同一输入/权重，隔离N32、K128及循环K顺序。"""

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
from mainproj_kernels import mainproj_arm

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "scope": "saved T24 main projection/state only", "arms": {}}
start = time.monotonic()


def metric(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    diff = actual.view(torch.int32) != expected.view(torch.int32)
    return {
        "bit_equal": not bool(diff.any()),
        "different_elements": int(diff.sum()),
        "max_abs": float((actual.double() - expected.double()).abs().max()),
        "first_coordinates": diff.nonzero()[:6].tolist(),
    }


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    torch.set_num_threads(4)
    reference = Path(os.environ["MAINPROJ_PROBE_REFERENCE"])
    fixture = Path(os.environ["SPMD_PRODUCTION_REFERENCE"])
    data = torch.load(reference, map_location="cpu", weights_only=True)
    initial = torch.load(fixture, map_location="cpu", weights_only=True)["initial"]
    gold = data["main_compressor_state_native"].reshape(data["main_state_native"].shape[0], -1)
    previous = data["main_compressor_state_actual"].reshape_as(gold)
    # The Native allocation can reserve more bytes than its live view. Use the
    # saved actual view, never reinterpret all allocation bytes as logical state.
    state_initial = data["main_state_initial"].reshape_as(gold)
    assert state_initial.dtype == torch.float32 and torch.isfinite(state_initial).all()
    contract = json.loads(reference.with_name("result.json").read_text())["main_compressor_contract"]
    raw = initial["cache_state"]
    assert raw.dtype == torch.int8 and raw.numel() == contract["allocation_bytes"]
    reconstructed = torch.as_strided(
        raw.view(torch.float32),
        size=contract["state_shape"],
        stride=contract["state_stride"],
        storage_offset=contract["state_storage_offset_bytes"] // 4,
    )
    record["saved_initial_view"] = metric(reconstructed.reshape_as(gold), state_initial)
    assert record["saved_initial_view"]["bit_equal"]
    record["state_layout_scope"] = "compact logical state view; production physical layout needs integrated recheck"
    owner_initial = torch.cat((torch.full((32,), -37.0), state_initial.reshape(-1), torch.full((32,), -37.0)))
    x = data["input_norm_native"].to("npu")
    assert x.shape == (24, 4096), "only the saved T24 first difference is qualified"
    names = ("main_wkv_native", "main_wgate_native", "main_ape_native")
    weights = [data[name].to("npu") for name in names]
    record["weights_sha256"] = {
        name: hashlib.sha256(data[name].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for name in names
    }
    metadata = [
        initial[name].to("npu")
        for name in (
            "positions",
            "readonly_compressed.query_start_loc",
            "readonly_compressed.seq_lens",
            "readonly_state.slot_mapping",
        )
    ]
    owner = owner_initial.to("npu")
    state = owner[32:-32].reshape_as(gold)
    storage = [torch.full((80, 1024), -37.0, dtype=torch.float32, device="npu") for _ in range(2)]
    values, scores = (buffer[8:72] for buffer in storage)
    record["reference"] = str(reference)
    record["fixture"] = str(fixture)
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    record["source_sha256"] = {
        name: hashlib.sha256((source / "deepseek_v4_flash_csa_spmd_tnd" / name).read_bytes()).hexdigest()
        for name in ("main_native_projection_stages.py", "compressor_projection_stages.py", "compressor_pool_stages.py")
    }
    pypto.torch.init()
    for mode in range(4):
        owner.copy_(owner_initial)
        mainproj_arm(x, *weights, *metadata, state, values, scores, MODE=mode)
        torch.npu.synchronize()
        eager = state.cpu().clone()
        eager_projected = [value[:24].cpu().clone() for value in (values, scores)]
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            mainproj_arm(x, *weights, *metadata, state, values, scores, MODE=mode)
        owner.copy_(owner_initial)
        values.fill_(float("nan"))
        scores.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        row = {
            "native_state": metric(state, gold),
            "graph_eager": metric(state, eager),
            "projection_graph_eager": [
                metric(value[:24], saved) for value, saved in zip((values, scores), eager_projected)
            ],
            "guard_ok": bool(
                torch.equal(owner[:32].cpu(), owner_initial[:32])
                and torch.equal(owner[-32:].cpu(), owner_initial[-32:])
                and all(
                    bool((buffer[:8].cpu() == -37).all() and (buffer[72:].cpu() == -37).all()) for buffer in storage
                )
            ),
        }
        if mode == 0:
            row["production_repeat"] = metric(state, previous)
            assert row["production_repeat"]["bit_equal"], "isolated projection does not reproduce production state"
        record["arms"][str(mode)] = row
        assert row["graph_eager"]["bit_equal"] and row["guard_ok"]
        assert all(value["bit_equal"] for value in row["projection_graph_eager"])
        if mode == 3:
            assert row["native_state"]["bit_equal"], "Native-order main projection first difference"
        (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    record.update(complete=True, **{"pass": True})
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
