# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the independent package before DSL imports.
"""同一保存输入、真实权重，先重现WKV首差再单因素验证paired K顺序。"""

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
from wkv240_kernels import wkv240_arm
from wkv_kernels import wkv_arm

from vllm_ascend.utils import enable_custom_op

out = Path(sys.argv[1])
record = {
    "complete": False,
    "pass": False,
    "scope": "isolated WKV T24; same Normed input and formal layer 4 weight",
    "arms": {},
}
start = time.monotonic()
payload = {}


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
    record["native_source"] = os.environ["SOURCE_ROOT"]
    record["source_sha256"] = {
        name: hashlib.sha256((source / "deepseek_v4_flash_csa_spmd_tnd" / name).read_bytes()).hexdigest()
        for name in ("kv_stages.py", "kv_native_stages.py", "kv_precision_dispatch.py")
    }
    reference = Path(os.environ["WKV_PROBE_REFERENCE"])
    data = torch.load(reference, map_location="cpu", weights_only=True)
    x = data["input_norm_native"].to("npu")
    tokens = x.shape[0]
    assert tokens in (24, 240), "only observed T24/T240 first differences are qualified"
    record["tokens"] = tokens
    record["scope"] = f"isolated WKV T{tokens}; same Normed input and formal layer 4 weight"
    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    key = "layers.4.attn.wkv.weight"
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    with safe_open(checkpoint / index[key], framework="pt", device="cpu") as reader:
        stored = reader.get_tensor(key)
    assert stored.dtype == torch.bfloat16 and stored.shape == (512, 4096)
    record["weight"] = {
        "key": key,
        "shard": index[key],
        "sha256": hashlib.sha256(stored.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }
    weight = torch_npu.npu_format_cast(stored.to("npu"), 29)
    native = torch.nn.functional.linear(x, weight)
    record["native_repeat_saved"] = metric(native, data["wkv_projection_bf16_native"])
    assert record["native_repeat_saved"]["bit_equal"], "checkpoint linear does not reproduce actual Native CV boundary"
    record["reference"] = str(reference)
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    pypto.torch.init()
    p16 = (tokens + 15) // 16 * 16
    owner = torch.full((p16 + 32, 512), -37.0, dtype=torch.float32, device="npu")
    projected = owner[16 : 16 + p16]
    arms = (
        ("nal", False, False),
        ("n32_k256_natural", True, False),
        ("n32_k256_paired", True, True),
    )
    if tokens == 240:
        arms = (("nal", False, False), ("n32_k256_natural", True, False), ("n32_k256_column_order", True, True))
    for arm_index, (name, narrow, order) in enumerate(arms):

        def run(arm_index=arm_index, narrow=narrow, order=order):
            if tokens == 240:
                wkv240_arm(x, weight, projected, MODE=arm_index)
            else:
                wkv_arm(x, weight, projected, NARROW=narrow, ORDER=order)

        run()
        torch.npu.synchronize()
        eager = projected[:tokens].cpu().clone()
        graph, stream = torch.npu.NPUGraph(), torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, stream=stream):
            run()
        projected.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        row = {
            "bf16_native": metric(projected[:tokens].bfloat16(), native),
            "graph_eager": metric(projected[:tokens], eager),
            "guard_ok": bool((owner[:16].cpu() == -37).all() and (owner[16 + p16 :].cpu() == -37).all()),
            "input_readonly": metric(x, data["input_norm_native"]),
            "weight_readonly": metric(weight, stored),
        }
        assert row["graph_eager"]["bit_equal"] and row["guard_ok"]
        assert row["input_readonly"]["bit_equal"] and row["weight_readonly"]["bit_equal"]
        if name == "nal":
            row["production_accumulator"] = metric(projected[:tokens], data["kv_accumulator"][:tokens])
            assert row["production_accumulator"]["bit_equal"], "isolated Nal does not reproduce production"
        record["arms"][name] = row
        payload[name] = projected[:tokens].cpu().clone()
        if order:
            assert row["bf16_native"]["bit_equal"], "paired WKV first difference"
    if os.environ.get("WKV_PROBE_RANGE") == "small":
        record["small_cases"] = []
        record["small_root"] = "production runtime dispatcher"
        # Native uses natural K order at M16 boundaries, paired order between them.
        # Check every small M rather than selecting a few favorable shapes.
        source_rows = x.cpu().repeat(3, 1)
        for small_t in range(1, 64):
            for variant in ("a", "b"):
                small_cpu = source_rows[:small_t].clone()
                if variant == "b":
                    small_cpu = (small_cpu.float() + 0.125).bfloat16()
                small = small_cpu.to("npu")
                gold = torch.nn.functional.linear(small, weight)
                padded_rows = (small_t + 15) // 16 * 16
                storage = torch.full((padded_rows + 32, 512), -37.0, dtype=torch.float32, device="npu")
                actual = storage[16 : 16 + padded_rows]
                paired = small_t % 16 != 0
                wkv_arm(small, weight, actual, NARROW=True, ORDER=paired, DISPATCH=True)
                torch.npu.synchronize()
                small_eager = actual[:small_t].cpu().clone()
                small_graph, small_stream = torch.npu.NPUGraph(), torch.npu.Stream()
                small_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.graph(small_graph, stream=small_stream):
                    wkv_arm(small, weight, actual, NARROW=True, ORDER=paired, DISPATCH=True)
                actual.fill_(float("nan"))
                small_graph.replay()
                torch.npu.synchronize()
                guard = storage.cpu()
                item = {
                    "tokens": small_t,
                    "input": variant,
                    "paired": paired,
                    "bf16_native": metric(actual[:small_t].bfloat16(), gold),
                    "graph_eager": metric(actual[:small_t], small_eager),
                    "guard_ok": bool((guard[:16] == -37).all() and (guard[16 + padded_rows :] == -37).all()),
                }
                record["small_cases"].append(item)
                (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
                if not (item["bf16_native"]["bit_equal"] and item["graph_eager"]["bit_equal"] and item["guard_ok"]):
                    record["first_difference"] = {"tokens": small_t, "input": variant}
                    raise AssertionError("small-M WKV first difference")
    record.update(complete=True, **{"pass": True})
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    torch.save(payload, out / "wkv-arms.pt")
