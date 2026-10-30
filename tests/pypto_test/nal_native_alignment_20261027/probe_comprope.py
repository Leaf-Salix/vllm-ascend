# SPDX-License-Identifier: Apache-2.0
"""保存的同次pooled完整512维Norm+RoPE；先复现基线，再验Native融合输出。"""

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
from compressor_rope_kernels import compressor_rope_arm

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "metrics": {}, "physical_device": int(os.environ["TASK_DEVICE"])}
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
    reference = Path(os.environ["COMPRESSOR_ROPE_REFERENCE"])
    previous = json.loads(reference.with_name("result.json").read_text())
    assert previous["complete"] and previous["first_difference"] == "main_compressed_cache"
    assert all(row["bit_equal"] for name, row in previous["precision"].items() if name != "main_compressed_cache")
    for gate in ("observation_gate", "input_equivalence", "stage_replay_eager", "main_native_repeat"):
        assert all(row["bit_equal"] for row in previous[gate].values()), gate
    saved = torch.load(reference, map_location="cpu", weights_only=True)
    closed = saved["main_closed_tokens"]
    valid = (saved["main_compact_slots"] >= 0).all(-1)
    assert int(closed.sum()) == int(valid.sum()) > 0
    pooled_cpu = saved["main_observed_pooled"][closed].contiguous()
    gamma_cpu = saved["main_gamma_native"].contiguous()
    cosine_cpu = saved["main_compact_cos"].reshape(-1, 64)[valid].contiguous()
    sine_cpu = saved["main_compact_sin"].reshape(-1, 64)[valid].contiguous()
    actual_cpu = saved["main_observed_debug"][closed].contiguous()
    gold_cpu = saved["main_compressed_native"][valid].contiguous()
    assert pooled_cpu.dtype == actual_cpu.dtype == torch.float32
    assert gamma_cpu.dtype == gold_cpu.dtype == torch.bfloat16
    epsilon = previous["main_compressor_contract"]["epsilon"]
    assert epsilon == 1e-6
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    inputs = [value.to("npu") for value in (pooled_cpu, gamma_cpu, cosine_cpu, sine_cpu)]
    rows = pooled_cpu.shape[0]
    owners = [torch.full((rows + 16, 512), -37.0, dtype=torch.float32, device="npu") for _ in range(2)]
    outputs = [owner[8:-8] for owner in owners]
    record.update(
        reference=str(reference),
        layer_index=previous.get("layer_index", 4),
        rows=rows,
        epsilon=epsilon,
        scope="saved identical pooled/gamma/RoPE; only direct division changes",
        kernel_sha256=hashlib.sha256(Path(__file__).with_name("compressor_rope_kernels.py").read_bytes()).hexdigest(),
        sdk_release=os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"],
    )
    pypto.torch.init()

    def run():
        for direct, output in zip((False, True), outputs, strict=True):
            compressor_rope_arm(*inputs, output, DIRECT=direct, EPS=epsilon)

    run()
    torch.npu.synchronize()
    eager = [value.cpu().clone() for value in outputs]
    payload.update(
        pooled=pooled_cpu,
        gamma=gamma_cpu,
        cosine=cosine_cpu,
        sine=sine_cpu,
        actual=actual_cpu,
        native=gold_cpu,
        baseline=eager[0],
        direct=eager[1],
    )
    record["metrics"]["baseline_reproduces_saved_fp32_all512"] = metric(eager[0], actual_cpu)
    assert record["metrics"]["baseline_reproduces_saved_fp32_all512"]["bit_equal"], "micro baseline changed producer"
    record["metrics"]["baseline_vs_native_bf16"] = metric(eager[0].bfloat16(), gold_cpu)
    record["metrics"]["direct_vs_native_bf16"] = metric(eager[1].bfloat16(), gold_cpu)
    record["metrics"]["direct_vs_baseline_fp32"] = metric(eager[1], eager[0])
    graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(graph, stream=stream):
        run()
    for replay in range(3):
        for output in outputs:
            output.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        for arm, output in enumerate(outputs):
            key = f"replay_{replay}_arm_{arm}"
            record["metrics"][key] = metric(output, eager[arm])
            assert record["metrics"][key]["bit_equal"]
    assert all(bool((owner[:8].cpu() == -37).all() and (owner[-8:].cpu() == -37).all()) for owner in owners)
    for device, cpu in zip(inputs, (pooled_cpu, gamma_cpu, cosine_cpu, sine_cpu), strict=True):
        assert metric(device, cpu)["bit_equal"]
    record.update(complete=True, guards_readonly=True)
    record["pass"] = record["metrics"]["direct_vs_native_bf16"]["bit_equal"]
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    torch.save(payload, out / "norm_rope.pt")
