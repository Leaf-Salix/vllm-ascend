# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Load the private DSL only after binding its namespace.
"""从实际Native WO-A BF16输入出发，分别验证整行量化与整数WO-B。"""

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
from oproj_suffix_kernels import output_b_arm, output_quant_arm

from vllm_ascend.utils import enable_custom_op

out = Path(sys.argv[1])
record = {
    "complete": False,
    "pass": False,
    "scope": "Native actual WO-A input, T24 quantization then WO-B",
    "precision": {},
}
payload = {}
start = time.monotonic()


def metric(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    kind = {1: torch.int8, 2: torch.int16, 4: torch.int32}[actual.element_size()]
    diff = actual.view(kind) != expected.view(kind)
    return {
        "bit_equal": not bool(diff.any()),
        "different_elements": int(diff.sum()),
        "max_abs": float((actual.double() - expected.double()).abs().max()),
        "first_coordinates": diff.nonzero()[:8].tolist(),
    }


def check(name, actual, expected):
    actual_cpu, expected_cpu = actual.cpu().clone(), expected.cpu().clone()
    payload[name + "_actual"], payload[name + "_native"] = actual_cpu, expected_cpu
    record["precision"][name] = metric(actual_cpu, expected_cpu)
    assert record["precision"][name]["bit_equal"], name


try:
    reference = Path(os.environ["OPROJ_PROBE_REFERENCE"])
    record["reference"] = str(reference)
    previous = json.loads(reference.with_name("result.json").read_text())
    assert previous["complete"] and previous["first_difference"] == "native_full_output"
    assert previous["precision"]["wo_a_bf16"]["bit_equal"]
    for gate in ("observation_gate", "native_qli_hook_control", "input_equivalence"):
        assert all(row["bit_equal"] for row in previous[gate].values())
    data = torch.load(reference, map_location="cpu", weights_only=True)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true" and enable_custom_op()
    value = data["native_output_stages"]["wo_a"].reshape(24, 8192).to("npu")
    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for suffix in ("weight", "weight_scale", "weight_offset"):
        key = "layers.4.attn.wo_b." + suffix
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as reader:
            tensors[suffix] = reader.get_tensor(key)
    assert tensors["weight"].shape == (4096, 8192) and tensors["weight"].dtype == torch.int8
    assert tensors["weight_scale"].dtype == torch.float32 and tensors["weight_scale"].shape == (4096, 1)
    assert not bool(tensors["weight_offset"].any())
    weight = torch_npu.npu_format_cast(tensors["weight"].t().contiguous().to("npu"), 29)
    scale = tensors["weight_scale"].flatten().to("npu")
    native_q, native_scale = torch_npu.npu_dynamic_quant(value, dst_type=torch.int8)
    assert native_q.shape == (24, 8192) and native_q.dtype == torch.int8
    assert native_scale.shape == (24,) and native_scale.dtype == torch.float32
    native_output = torch_npu.npu_quant_matmul(
        native_q, weight, scale, pertoken_scale=native_scale, bias=None, output_dtype=torch.bfloat16
    )
    record["source_sha256"] = {
        "oproj_suffix_kernels.py": hashlib.sha256(
            Path(__file__).with_name("oproj_suffix_kernels.py").read_bytes()
        ).hexdigest(),
        "output_projection_b_stages.py": hashlib.sha256(
            (source / "deepseek_v4_flash_csa_spmd_tnd/output_projection_b_stages.py").read_bytes()
        ).hexdigest(),
    }
    record["weight_sha256"] = {
        key: hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest() for key, value in tensors.items()
    }
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    payload.update(input=value.cpu().clone(), native_q=native_q.cpu().clone(), native_scale=native_scale.cpu().clone())
    check(
        "native_wo_b_repeat_actual_projection",
        native_output,
        data["native_output_stages"]["projection"].reshape(24, 4096),
    )
    before = {"input": value.cpu().clone(), "weight": weight.cpu().clone(), "weight_scale": scale.cpu().clone()}
    q_owner = torch.full((160, 8192), -99, dtype=torch.int8, device="npu")
    quantized = q_owner[16:144]
    scale_owner = torch.full((8 * 128 + 32,), -37.0, dtype=torch.float32, device="npu")
    scales = scale_owner[16:-16].reshape(8, 128)
    pypto.torch.init()
    output_quant_arm(value, quantized, scales)
    torch.npu.synchronize()
    eager_q, eager_scale = quantized.cpu().clone(), scales[:, :24].cpu().clone()
    payload.update(quantized_eager=eager_q[:24], scale_eager=eager_scale)
    check("quantized_eager", quantized[:24], native_q)
    check("scale_eager", scales[:, :24], native_scale.reshape(1, 24).expand(8, 24))
    quant_graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(quant_graph, stream=stream):
        output_quant_arm(value, quantized, scales)
    for replay in range(3):
        quantized.fill_(-99)
        scales.fill_(float("nan"))
        quant_graph.replay()
        torch.npu.synchronize()
        check(f"quantized_replay_{replay}", quantized, eager_q)
        check(f"scale_replay_{replay}", scales[:, :24], eager_scale)
    assert not bool(quantized[24:].cpu().any()), "WO-B padded matrix rows must be zero"
    assert bool((q_owner[:16].cpu() == -99).all() and (q_owner[144:].cpu() == -99).all())
    assert bool((scale_owner[:16].cpu() == -37).all() and (scale_owner[-16:].cpu() == -37).all())
    partials = torch.full((128, 32768), -101, dtype=torch.int32, device="npu")
    output_owner = torch.full((56, 4096), -37.0, dtype=torch.bfloat16, device="npu")
    output = output_owner[16:40]
    output_b_arm(quantized, scales, weight, scale, partials, output)
    torch.npu.synchronize()
    payload["partials_eager"] = partials[:24].cpu().clone()
    check("wo_b_eager", output, native_output)
    eager_output = output.cpu().clone()
    b_graph, b_stream = torch.npu.NPUGraph(), torch.npu.Stream()
    b_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.graph(b_graph, stream=b_stream):
        output_b_arm(quantized, scales, weight, scale, partials, output)
    for replay in range(3):
        partials.fill_(-101)
        output.fill_(float("nan"))
        b_graph.replay()
        torch.npu.synchronize()
        payload["partials_replay"] = partials[:24].cpu().clone()
        check(f"wo_b_replay_{replay}", output, eager_output)
    assert bool((output_owner[:16].cpu() == -37).all() and (output_owner[40:].cpu() == -37).all())
    for name, tensor in (("input", value), ("weight", weight), ("weight_scale", scale)):
        check("readonly_" + name, tensor, before[name])
    payload.update(
        quantized=quantized[:24].cpu(),
        scales=scales[:, :24].cpu(),
        output=output.cpu(),
        native_output=native_output.cpu(),
    )
    record.update(complete=True, **{"pass": True})
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    torch.save(payload, out / "suffix.pt")
