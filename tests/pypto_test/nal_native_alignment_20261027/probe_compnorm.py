# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the private DSL namespace before importing the probe.
"""固定同次真实pooled，只比较NoPE Norm，不将通用RMS当融合compressor金值。"""

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

bind()
from compressor_norm_kernels import compressor_norm_pair

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "scope": "same actual pooled, NoPE compressor RMS only", "metrics": {}}
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


try:
    reference = Path(os.environ["COMPRESSOR_NORM_REFERENCE"])
    previous = json.loads(reference.with_name("result.json").read_text())
    assert previous["complete"] and previous["first_difference"] == "main_compressed_cache"
    assert previous["precision"]["main_compressor_state"]["bit_equal"]
    for gate in ("observation_gate", "input_equivalence", "stage_replay_eager"):
        assert all(row["bit_equal"] for row in previous[gate].values())
    saved = torch.load(reference, map_location="cpu", weights_only=True)
    closed = saved["main_closed_tokens"]
    valid = (saved["main_compact_slots"] >= 0).all(dim=-1)
    assert int(closed.sum()) == int(valid.sum()) > 0
    pooled_cpu = saved["main_observed_pooled"][closed].contiguous()
    gamma_cpu = saved["main_gamma_native"].contiguous()
    actual_cpu = saved["main_observed_normed"][closed].contiguous()
    gold_cpu = saved["main_compressed_native"][valid].contiguous()
    assert pooled_cpu.dtype == actual_cpu.dtype == torch.float32 and gamma_cpu.dtype == gold_cpu.dtype == torch.bfloat16
    epsilon = previous["main_compressor_contract"]["epsilon"]
    assert epsilon == 1e-6, "CPU qualification used EPS=1e-6; recompile for another epsilon"
    record["closed_compact_pairs"] = [
        {"token_row": int(token), "compact_row": int(compact), "slot": saved["main_compact_slots"][compact].tolist()}
        for token, compact in zip(closed.nonzero().flatten(), valid.nonzero().flatten(), strict=True)
    ]
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    pooled, gamma = pooled_cpu.to("npu"), gamma_cpu.to("npu")
    record["reference"] = str(reference)
    record["kernel_sha256"] = hashlib.sha256(
        Path(__file__).with_name("compressor_norm_kernels.py").read_bytes()
    ).hexdigest()
    record["epsilon"] = epsilon
    record["closed_token_rows"] = closed.nonzero().flatten().tolist()
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    rows = pooled.shape[0]
    owners = [torch.full((rows + 16, 512), -37.0, dtype=torch.float32, device="npu") for _ in range(2)]
    baseline, direct = (value[8:-8] for value in owners)
    pypto.torch.init()
    compressor_norm_pair(pooled, gamma, baseline, direct, EPS=epsilon)
    torch.npu.synchronize()
    eager = [value.cpu().clone() for value in (baseline, direct)]
    payload.update(
        pooled=pooled_cpu,
        gamma=gamma_cpu,
        native_fused=gold_cpu,
        actual_normed=actual_cpu,
        baseline=eager[0],
        direct=eager[1],
    )
    record["metrics"]["baseline_reproduce_actual_nope_fp32"] = metric(eager[0][:, :448], actual_cpu[:, :448])
    assert record["metrics"]["baseline_reproduce_actual_nope_fp32"]["bit_equal"], "micro must reproduce actual producer"
    generic = torch_npu.npu_rms_norm(pooled, gamma.float(), epsilon=epsilon)[0]
    payload["generic_native_rms"] = generic.cpu()
    record["metrics"]["generic_rms_vs_actual_fused_nope"] = metric(generic.bfloat16()[:, :448], gold_cpu[:, :448])
    record["metrics"]["baseline_vs_actual_fused_nope"] = metric(eager[0].bfloat16()[:, :448], gold_cpu[:, :448])
    record["metrics"]["direct_vs_actual_fused_nope"] = metric(eager[1].bfloat16()[:, :448], gold_cpu[:, :448])
    graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(graph, stream=stream):
        compressor_norm_pair(pooled, gamma, baseline, direct, EPS=epsilon)
    for replay in range(3):
        baseline.fill_(float("nan"))
        direct.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        for arm, value in enumerate((baseline, direct)):
            key = f"replay_{replay}_arm_{arm}"
            record["metrics"][key] = metric(value, eager[arm])
            assert record["metrics"][key]["bit_equal"], key
    assert all(bool((owner[:8].cpu() == -37).all() and (owner[-8:].cpu() == -37).all()) for owner in owners)
    assert metric(pooled, pooled_cpu)["bit_equal"] and metric(gamma, gamma_cpu)["bit_equal"]
    record["guards_readonly"] = True
    record["complete"] = True
    record["pass"] = record["metrics"]["direct_vs_actual_fused_nope"]["bit_equal"]
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    torch.save(payload, out / "norm.pt")
