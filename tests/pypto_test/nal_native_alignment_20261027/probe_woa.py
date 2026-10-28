# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the private DSL namespace before importing the root.
"""真实Native heads和wo_b入口：WO-A原实现复现、布局及K顺序单因素对拍。"""

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
from woa_kernels import woa_arm

from vllm_ascend.utils import enable_custom_op

out = Path(sys.argv[1])
record = {"complete": False, "pass": False, "scope": "WO-A only, saved actual Native layer4 heads, T24", "arms": {}}
payload = {}
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


def save():
    (out / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    torch.save(payload, out / "woa-arms.pt")


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true" and enable_custom_op()
    reference = Path(os.environ["WOA_PROBE_REFERENCE"])
    record["reference"] = str(reference)
    data = torch.load(reference, map_location="cpu", weights_only=True)
    observer = json.loads(reference.with_name("result.json").read_text())
    assert observer["complete"] and observer["first_difference"] == "wo_a_bf16"
    for gate in ("observation_gate", "native_qli_hook_control", "input_equivalence"):
        assert all(row["bit_equal"] for row in observer[gate].values())
    assert observer["precision"]["attention_heads_after_inverse_rope"]["bit_equal"]
    record["source_sha256"] = {
        "woa_kernels.py": hashlib.sha256(Path(__file__).with_name("woa_kernels.py").read_bytes()).hexdigest(),
        "output_projection_a_stages.py": hashlib.sha256(
            (source / "deepseek_v4_flash_csa_spmd_tnd/output_projection_a_stages.py").read_bytes()
        ).hexdigest(),
    }
    # v19的23项前缀清单不含WO-A；单独核验已推送a57c3cb4的未改文件，
    # 然后仍要求MODE0逐bit复现v19真实FP32，不能以文件身份替代数值控制。
    assert record["source_sha256"]["output_projection_a_stages.py"] == os.environ["WOA_PROBE_SOURCE_SHA256"]
    record["wo_a_reference_commit"] = "a57c3cb4"
    heads = data["native_output_stages"]["heads"].reshape(24, 32768).to("npu")
    gold = data["native_output_stages"]["wo_a"].reshape(24, 8192)
    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    key = "layers.4.attn.wo_a.weight"
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    with safe_open(checkpoint / index[key], framework="pt", device="cpu") as reader:
        stored = reader.get_tensor(key)
    assert stored.dtype == torch.bfloat16 and stored.shape == (8192, 4096)
    record["weight"] = {
        "key": key,
        "shard": index[key],
        "sha256": hashlib.sha256(stored.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }
    weight = torch_npu.npu_format_cast(stored.reshape(8, 1024, 4096).transpose(1, 2).contiguous().to("npu"), 29)
    native = torch_npu.npu_transpose_batchmatmul(
        heads.view(24, 8, 4096),
        weight,
        bias=None,
        scale=None,
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
        batch_split_factor=1,
    ).reshape(24, 8192)
    record["native_repeat_saved"] = metric(native, gold)
    assert record["native_repeat_saved"]["bit_equal"], "loaded weight/BMM does not reproduce actual Native wo_b entry"
    packed = torch.zeros((1024, 4096), dtype=torch.bfloat16, device="npu")
    for group in range(8):
        packed[group * 128 : group * 128 + 24].copy_(heads[:, group * 4096 : (group + 1) * 4096])
    before = {"packed": packed.cpu().clone(), "weight": weight.cpu().clone()}
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    record["native_source"] = os.environ["SOURCE_ROOT"]
    pypto.torch.init()
    for name, mode in (("production", 0), ("m16_n128_natural", 1), ("historical_half_rotate", 2)):
        owner = torch.full((160, 8192), -37.0, dtype=torch.float32, device="npu")
        projected = owner[16:144]
        woa_arm(packed, weight, projected, MODE=mode)
        torch.npu.synchronize()
        eager = projected[:24].cpu().clone()
        row = {"bf16_native": metric(projected[:24].bfloat16(), gold), "replays": []}
        if mode == 0:
            row["production_accumulator"] = metric(projected[:24], data["observed_o_r"])
            assert row["production_accumulator"]["bit_equal"], (
                "isolated original WO-A changed its production FP32 result"
            )
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            woa_arm(packed, weight, projected, MODE=mode)
        for _ in range(3):
            projected.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            item = {
                "graph_eager": metric(projected[:24], eager),
                "bf16_native": metric(projected[:24].bfloat16(), gold),
            }
            row["replays"].append(item)
            assert item["graph_eager"]["bit_equal"], "WO-A replay drift"
        row["guard_ok"] = bool((owner[:16].cpu() == -37).all() and (owner[144:].cpu() == -37).all())
        assert row["guard_ok"]
        record["arms"][name] = row
        payload[name] = projected[:24].cpu().clone()
        save()
    record["readonly"] = {"packed": metric(packed, before["packed"]), "weight": metric(weight, before["weight"])}
    assert all(row["bit_equal"] for row in record["readonly"].values())
    record["complete"] = True
    record["pass"] = record["arms"]["historical_half_rotate"]["bf16_native"]["bit_equal"]
    if not record["pass"]:
        record["first_difference"] = "wo_a_bf16"
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    save()
