# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the private package before importing its DSL modules.
"""真实第4层HC权重，隔离生产helper与Native融合HC同输入对拍。"""

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
from hc_kernels import hc_pair_arm
from layout_kernels import pack_pair

from vllm_ascend.utils import enable_custom_op

out = Path(sys.argv[1])
record = {
    "complete": False,
    "scope": "isolated HC module using production helpers; not complete CSA or SPMD performance",
    "arms": {},
}
start = time.monotonic()


def save():
    (out / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def metric(actual, expected):
    actual, expected = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, (actual.shape, expected.shape)
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    bit_dtype = torch.int16 if actual.element_size() == 2 else torch.int32
    different = actual.view(bit_dtype) != expected.view(bit_dtype)
    delta = actual.double() - expected.double()
    return {
        "bit_equal": torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)),
        "different_elements": int(different.sum()),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
        ),
        "first_coordinates": different.nonzero()[:6].tolist(),
    }


def view_outputs(buffers):
    return {name: buffers[name][:tokens].cpu().clone() for name in ("inv", "raw", "post", "comb", "mixed", "normed")}


try:
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    assert enable_custom_op()
    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((checkpoint / "config.json").read_text())
    weights = {}
    record["weights"] = {}
    for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm.weight"):
        key = f"layers.4.{name}"
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as reader:
            value = reader.get_tensor(key)
        record["weights"][key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
        # Native make_layer在BF16构造的RMSNorm.weight中copy checkpoint。
        loaded = value.to(torch.bfloat16) if name == "attn_norm.weight" else value
        record["weights"][key]["loaded_dtype"] = str(loaded.dtype)
        weights[name] = loaded.to("npu")
    fn, scale, base, norm = (
        weights[name] for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm.weight")
    )
    tokens = 24
    p16, p8 = (tokens + 15) // 16 * 16, (tokens + 7) // 8 * 8
    torch.manual_seed(1024 + tokens)
    hidden_cpu = torch.randn((tokens, 4, 4096), dtype=torch.bfloat16)
    hidden = hidden_cpu.to("npu")
    x = hidden.view(tokens, 16384)
    native_args = (fn, scale, base, 4, 20, config["rms_norm_eps"], config["hc_eps"])
    native = torch.ops._C_ascend.npu_hc_pre_v2(hidden, *native_args)
    native_repeat = torch.ops._C_ascend.npu_hc_pre_v2(hidden, *native_args)
    native_named = {name: value for name, value in zip(("mixed", "post", "comb"), native, strict=True)}
    native_named["comb"] = native_named["comb"].reshape(tokens, 16)
    record["native_repeat"] = [metric(a, b) for a, b in zip(native, native_repeat, strict=True)]
    assert all(row["bit_equal"] for row in record["native_repeat"])
    record["reference_kind"] = "fused_npu_hc_pre_v2_aclnnHcPre"
    record["tokens"] = tokens
    record["native_source"] = os.environ["SOURCE_ROOT"]
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    pypto.torch.init()
    payload = {"hidden": hidden_cpu, **{"native_" + name: value.cpu() for name, value in native_named.items()}}
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_full_precision.hc_pre import hc_pre_test

    reference_mixed = torch.empty((tokens, 4096), dtype=torch.bfloat16, device="npu")
    reference_post = torch.empty((tokens, 4), dtype=torch.float32, device="npu")
    reference_comb = torch.empty((tokens, 16), dtype=torch.float32, device="npu")
    hc_pre_test(hidden, fn, scale, base, reference_mixed, reference_post, reference_comb)
    torch.npu.synchronize()
    record["old_precision_reference_vs_fused_native"] = {
        name: metric(value, native_named[name])
        for name, value in zip(
            ("mixed", "post", "comb"), (reference_mixed, reference_post, reference_comb), strict=True
        )
    }
    save()
    assert all(row["bit_equal"] for row in record["old_precision_reference_vs_fused_native"].values()), (
        "old precision reference does not match fused Native on new SDK"
    )
    # 直接BF16打包只改布局。完整物理P16输出与独立CPU字节构造对照。
    layout_wide = torch.empty((tokens, 16384), dtype=torch.float32, device="npu")
    old_packed = torch.empty((p16, 32768), dtype=torch.bfloat16, device="npu")
    direct_packed = torch.empty_like(old_packed)
    with torch.inference_mode():
        pack_pair(x, layout_wide, old_packed, direct_packed)
        torch.npu.synchronize()
        layout_graph = torch.npu.NPUGraph()
        layout_stream = torch.npu.Stream()
        with torch.npu.graph(layout_graph, stream=layout_stream):
            pack_pair(x, layout_wide, old_packed, direct_packed)
    record["layout_aba"] = []
    for label, layout_hidden in (("A", hidden_cpu), ("B", hidden_cpu + 0.125), ("A", hidden_cpu)):
        x.copy_(layout_hidden.view(tokens, 16384))
        layout_wide.fill_(float("nan"))
        old_packed.fill_(float("nan"))
        direct_packed.fill_(float("nan"))
        layout_graph.replay()
        torch.npu.synchronize()
        padded = torch.zeros((p16, 16384), dtype=torch.bfloat16)
        padded[:tokens] = layout_hidden.view(tokens, 16384)
        expected_packed = padded.reshape(p16, 2048, 8).repeat_interleave(2, dim=1).reshape(p16, 32768)
        checks = {
            "input": label,
            "old_vs_cpu": metric(old_packed, expected_packed),
            "direct_vs_cpu": metric(direct_packed, expected_packed),
        }
        record["layout_aba"].append(checks)
        save()
        assert checks["old_vs_cpu"]["bit_equal"] and checks["direct_vs_cpu"]["bit_equal"], "layout changes packed bytes"
    arms = (
        ("nalinaly_original_hc", False, False, False, False),
        ("nalinaly_hc_rms_only", True, False, False, False),
        ("nalinaly_hc_rms_linear", True, True, False, False),
        ("nalinaly_hc_rms_linear_fold", True, True, True, False),
        ("nalinaly_hc_native_norm", True, True, True, True),
    )
    if len(sys.argv) > 2 and sys.argv[2] == "after-fold":
        arms = arms[-2:]
    for arm, repaired, native_linear, native_fold, native_norm in arms:
        shapes = {
            "wide": (tokens, 16384),
            "inv": (p16, 1),
            "partials": (p16 * (22 if native_linear else 4), 32),
            "squares": (p16 * 22, 16),
            "raw": (p16, 32),
            "post": (tokens, 4),
            "comb": (tokens, 16),
            "mixed": (p8, 4096),
            "normed": (tokens, 4096),
        }
        buffers = {
            name: torch.full(
                shape,
                float("nan"),
                dtype=torch.bfloat16 if name in ("mixed", "normed") else torch.float32,
                device="npu",
            )
            for name, shape in shapes.items()
        }

        def run(
            buffers=buffers,
            repaired=repaired,
            native_linear=native_linear,
            native_fold=native_fold,
            native_norm=native_norm,
        ):
            return hc_pair_arm(
                x,
                fn,
                scale,
                base,
                base.view(1, 24),
                norm,
                **buffers,
                NATIVE_RMS=repaired,
                NATIVE_LINEAR=native_linear,
                NATIVE_FOLD=native_fold,
                NATIVE_NORM=native_norm,
                DIRECT_PACK=native_norm,
            )

        with torch.inference_mode():
            run()
            torch.npu.synchronize()
            eager = view_outputs(buffers)
            graph = torch.npu.NPUGraph()
            stream = torch.npu.Stream()
            with torch.npu.graph(graph, stream=stream):
                run()
            for value in buffers.values():
                value.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
        actual = view_outputs(buffers)
        payload.update({arm + "_" + name: value for name, value in actual.items()})
        torch.save(payload, out / "observation.pt")
        # raw只取真实24列；右侧8列是无效N padding，不构成数值契约。
        controls = {
            name: metric(
                value[:, :24] if name == "raw" else value, eager[name][:, :24] if name == "raw" else eager[name]
            )
            for name, value in actual.items()
        }
        assert all(row["bit_equal"] for row in controls.values()), "diagnostic FULL replay differs from eager"
        assert torch.equal(buffers["wide"].cpu(), hidden_cpu.view(tokens, 16384).float())
        comparisons = {name: metric(actual[name], expected) for name, expected in native_named.items()}
        # 直接把同一组raw/inverse送入Native Sinkhorn，剥离上游误差。
        local = torch.ops._C_ascend.npu_hc_pre_sinkhorn(
            buffers["raw"][:tokens, :24].contiguous(),
            buffers["inv"][:tokens],
            scale,
            base,
            hidden,
            4,
            20,
            config["hc_eps"],
        )
        local_named = {name: value for name, value in zip(("mixed", "post", "comb"), local, strict=True)}
        local_named["mixed"] = local_named["mixed"].to(torch.bfloat16)
        local_named["comb"] = local_named["comb"].reshape(tokens, 16)
        conditional = {name: metric(actual[name], expected) for name, expected in local_named.items()}
        normed_native = torch_npu.npu_rms_norm(buffers["mixed"][:tokens], norm, config["rms_norm_eps"])[0]
        conditional["input_norm_same_mixed"] = metric(actual["normed"], normed_native)
        if native_norm:
            comparisons["normed"] = metric(
                actual["normed"], torch_npu.npu_rms_norm(native_named["mixed"], norm, config["rms_norm_eps"])[0]
            )
        record["arms"][arm] = {
            "full_graph_eager": controls,
            "vs_fused_native_hc": comparisons,
            "same_input_native_submodule": conditional,
        }
        payload.update({arm + "_" + name: value for name, value in actual.items()})
        record["arms"][arm]["aba"] = []
        if native_fold:
            save()
            torch.save(payload, out / "observation.pt")
            assert all(row["bit_equal"] for row in comparisons.values()), "first repaired HC output difference"
            for label, second_hidden_cpu in (("B", hidden_cpu + 0.125), ("A", hidden_cpu)):
                x.copy_(second_hidden_cpu.view(tokens, 16384))
                native_second = torch.ops._C_ascend.npu_hc_pre_v2(hidden, *native_args)
                with torch.inference_mode():
                    run()
                    torch.npu.synchronize()
                    eager_second = view_outputs(buffers)
                    for value in buffers.values():
                        value.fill_(float("nan"))
                    graph.replay()
                    torch.npu.synchronize()
                actual_second = view_outputs(buffers)
                replay_controls = {
                    name: metric(
                        value[:, :24] if name == "raw" else value,
                        eager_second[name][:, :24] if name == "raw" else eager_second[name],
                    )
                    for name, value in actual_second.items()
                }
                native_second_named = dict(zip(("mixed", "post", "comb"), native_second, strict=True))
                native_second_named["comb"] = native_second_named["comb"].reshape(tokens, 16)
                hc_controls = {name: metric(actual_second[name], value) for name, value in native_second_named.items()}
                record["arms"][arm]["aba"].append(
                    {"input": label, "full_graph_eager": replay_controls, "vs_fused_native_hc": hc_controls}
                )
                payload.update({arm + "_" + label + "_" + name: value for name, value in actual_second.items()})
                torch.save(payload, out / "observation.pt")
                save()
                assert all(row["bit_equal"] for row in replay_controls.values()), "stale FULL graph packed input"
                assert all(row["bit_equal"] for row in hc_controls.values()), (
                    "first repaired HC changed-input difference"
                )
        save()
        print(arm, json.dumps(record["arms"][arm]), flush=True)
    torch.save(payload, out / "observation.pt")
    record["complete"] = True
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    save()
