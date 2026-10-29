# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the private DSL namespace before importing the probe.
"""T48首差：用已逐bit相同的state重建窗口，仅改变Indexer pool。"""

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
from indexpool_kernels import indexpool_arm

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "scope": "saved actual T48 Indexer eight-row windows", "arms": {}}
payload = {}
start = time.monotonic()


def metric(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    integer = torch.int32 if actual.element_size() == 4 else torch.int16
    different = actual.view(integer) != expected.view(integer)
    return {
        "bit_equal": not bool(different.any()),
        "different_elements": int(different.sum()),
        "max_abs": float((actual.double() - expected.double()).abs().max()),
        "first_coordinates": different.nonzero()[:8].tolist(),
    }


def save():
    (out / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    torch.save(payload, out / "indexpool.pt")


try:
    reference = Path(os.environ["INDEXPOOL_PROBE_REFERENCE"])
    production = Path(os.environ["SPMD_PRODUCTION_REFERENCE"])
    previous = json.loads(reference.with_name("result.json").read_text())
    assert previous["complete"] and previous["first_difference"] == "indexer_norm_rope"
    assert previous["tokens"] == 48 and previous["lengths"] == [6] * 8
    assert previous["main_compressor_contract"]["epsilon"] == 1e-6, "recompile the micro for another shared epsilon"
    assert len(previous["precision"]) == 14 and all(
        row["bit_equal"] for name, row in previous["precision"].items() if name != "indexer_norm_rope"
    )
    for gate in ("observation_gate", "input_equivalence", "stage_replay_eager", "indexer_native_repeat"):
        assert all(row["bit_equal"] for row in previous[gate].values())
    assert all(row["status"] == "PASS" for row in previous["guards"].values())
    record["stage_source_sha256"] = {
        name: hashlib.sha256((source / "deepseek_v4_flash_csa_spmd_tnd" / name).read_bytes()).hexdigest()
        for name in previous["stage_source_sha256"]
    }
    assert len(record["stage_source_sha256"]) == 27
    assert record["stage_source_sha256"] == previous["stage_source_sha256"]
    data = torch.load(reference, map_location="cpu", weights_only=True)
    snapshot = torch.load(production, map_location="cpu", weights_only=True)
    assert snapshot["source_sha256"] == record["stage_source_sha256"]
    initial = snapshot["initial"]
    contract = previous["indexer_compressor_contract"]
    raw = initial["cache_indexer_state"]
    assert raw.numel() == contract["allocation_bytes"]
    state = torch.as_strided(
        raw.view(torch.float32),
        contract["state_shape"],
        contract["state_stride"],
        storage_offset=contract["state_storage_offset_bytes"] // 4,
    )
    assert metric(state, data["indexer_state_initial"])["bit_equal"]
    old = state.reshape(state.shape[0], -1)
    new = data["indexer_compressor_state_actual"].reshape_as(old)
    assert metric(new, data["indexer_compressor_state_native"].reshape_as(new))["bit_equal"]
    positions = initial["positions"]
    bounds = initial["readonly_compressed.query_start_loc"]
    assert torch.equal(bounds, initial["readonly_indexer.query_start_loc"])
    lengths = initial["readonly_indexer.seq_lens"]
    table = initial["readonly_indexer_state.block_table"]
    slots = initial["readonly_indexer_state.slot_mapping"]
    assert slots.shape == (48, 2) and bool((slots >= 0).all())
    assert len({tuple(slot) for slot in slots.tolist()}) == 48, "later writes must not overwrite this window"
    valid = (data["indexer_compact_slots"] >= 0).all(dim=-1)
    values, scores, tokens = [], [], []
    for request in range(lengths.numel()):
        begin, end = int(bounds[request]), int(bounds[request + 1])
        history = int(lengths[request]) - (end - begin)
        for token in range(begin, end):
            position = int(positions[token])
            assert position == history + token - begin
            if (position + 1) % 4:
                continue
            tokens.append(token)
            for index in range(8):
                logical = position - 7 + index
                assert logical >= 0, "this snapshot has nonnegative history"
                page = int(table[request, logical // 2])
                offset = logical % 2
                assert 0 <= page < old.shape[0]
                selected = old
                if logical >= history:
                    local_token = begin + logical - history
                    assert slots[local_token].tolist() == [page, offset]
                    selected = new
                half = 128 if index >= 4 else 0
                column = offset * 512 + half
                values.append(selected[page, column : column + 128])
                # The committed state score already includes APE; do not add it again.
                scores.append(selected[page, column + 256 : column + 384])
    assert len(tokens) == int(valid.sum()) == 8 and len(values) == 64
    values_cpu, scores_cpu = torch.stack(values), torch.stack(scores)
    gold = data["indexer_compressed_native"][valid].contiguous()
    actual = data["observed_idx_normed"][: valid.numel()][valid].contiguous()
    pooled_gold = data["observed_idx_pooled"][tokens].contiguous()
    assert metric(actual, data["indexer_norm_rope_actual"])["bit_equal"]
    assert metric(gold, data["indexer_norm_rope_native"])["bit_equal"]
    gamma_cpu = data["indexer_gamma_native"].contiguous()
    cos_cpu = initial["readonly_indexer.compact_cos"].reshape(-1, 64)[valid].contiguous()
    sin_cpu = initial["readonly_indexer.compact_sin"].reshape(-1, 64)[valid].contiguous()
    record.update(
        reference=str(reference),
        production=str(production),
        closed_tokens=tokens,
        sdk_release=os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"],
        kernel_sha256=hashlib.sha256(Path(__file__).with_name("indexpool_kernels.py").read_bytes()).hexdigest(),
    )
    payload.update(
        values=values_cpu,
        scores=scores_cpu,
        gamma=gamma_cpu,
        cos=cos_cpu,
        sin=sin_cpu,
        actual_pooled=pooled_gold,
        actual_normed=actual,
        native_normed=gold,
    )
    save()
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    inputs = [tensor.to("npu") for tensor in (values_cpu, scores_cpu, gamma_cpu, cos_cpu, sin_cpu)]
    owners = [torch.full((24, 128), -37, dtype=dtype, device="npu") for dtype in (torch.float32, torch.bfloat16)]
    outputs = [owner[8:-8] for owner in owners]
    pypto.torch.init()
    for mode in range(2):
        indexpool_arm(*inputs, *outputs, MODE=mode)
        torch.npu.synchronize()
        eager = [tensor.cpu().clone() for tensor in outputs]
        payload[f"mode_{mode}_eager"] = eager
        row = {"normed_native": metric(eager[1], gold), "normed_actual": metric(eager[1], actual), "replays": []}
        record["arms"][str(mode)] = row
        if mode == 0:
            row["pooled_actual"] = metric(eager[0], pooled_gold)
            save()
            assert row["pooled_actual"]["bit_equal"] and row["normed_actual"]["bit_equal"], (
                "micro must reproduce first difference"
            )
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            indexpool_arm(*inputs, *outputs, MODE=mode)
        for replay in range(3):
            outputs[0].fill_(float("nan"))
            outputs[1].fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            actual_outputs = [tensor.cpu().clone() for tensor in outputs]
            payload[f"mode_{mode}_replay_{replay}"] = actual_outputs
            metrics = [metric(value, expected) for value, expected in zip(actual_outputs, eager, strict=True)]
            row["replays"].append(metrics)
            save()
            assert all(value["bit_equal"] for value in metrics)
        assert all(bool((owner[:8].cpu() == -37).all() and (owner[-8:].cpu() == -37).all()) for owner in owners)
        assert all(
            metric(value, expected)["bit_equal"]
            for value, expected in zip(inputs, (values_cpu, scores_cpu, gamma_cpu, cos_cpu, sin_cpu), strict=True)
        )
        row["guards_readonly"] = True
    record["complete"] = True
    record["pass"] = record["arms"]["1"]["normed_native"]["bit_equal"]
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    save()
