# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind Native control and the independent PTO namespace before imports.
"""8K/B4完整SPMD运行及前缀观测；从HC逐阶段比对Native，首差停止。"""

import hashlib
import importlib
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
sys.path.insert(0, str(Path(os.environ["SOURCE_ROOT"]) / "tests/pypto_test"))
from dsv4_csa_native_case import native_session
from dsv4_csa_single_layer import collect_state, guard_checks, make_layer, restore
from ragged_fixture import make_ragged_fixture
from spmd_probe_entry import spmd_stage_probe
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

from vllm_ascend.utils import enable_custom_op

nal_adapter = importlib.import_module("nal_alignment.deepseek_v4_flash_csa.native_adapter")
nal_native = importlib.import_module("nal_alignment.deepseek_v4_flash_csa_spmd_tnd.native_adapter")
nal_binding = importlib.import_module("nal_alignment.deepseek_v4_flash_csa_spmd_tnd.binding")
nal_entry = importlib.import_module("nal_alignment.deepseek_v4_flash_csa_spmd_tnd.entry")

out = Path(sys.argv[1])
record = {
    "complete": False,
    "pass": False,
    "scope": "complete CSA SPMD execution; Native numerical gates only through QR; not whole-model tokens",
    "history": 8192,
    "lengths": [6, 6, 6, 6],
    "precision": {},
}
payload = {}
start = time.monotonic()


def save():
    (out / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    torch.save(payload, out / "observation.pt")


def metric(actual, expected):
    actual, expected = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    dtype = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[actual.element_size()]
    different = actual.view(dtype) != expected.view(dtype)
    if actual.is_floating_point():
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    delta = actual.double() - expected.double()
    return {
        "bit_equal": not bool(different.any()),
        "different_elements": int(different.sum()),
        "max_abs": float(delta.abs().max()),
        "first_coordinates": different.nonzero()[:6].tolist(),
    }


def check(name, actual, expected):
    payload[name + "_actual"], payload[name + "_native"] = actual.cpu(), expected.cpu()
    row = metric(actual, expected)
    record["precision"][name] = row
    save()
    if not row["bit_equal"]:
        record["first_difference"] = name
        raise FirstDifference(name)


class FirstDifference(Exception):
    pass


try:
    current_platform.pre_register_and_update()
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_deterministic_level(1)
    torch.set_num_threads(4)
    assert os.environ["HCCL_DETERMINISTIC"] == "true"
    assert enable_custom_op()
    checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    config = EngineArgs(
        model=str(checkpoint),
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        quantization="ascend",
        hf_overrides={"sliding_window": 128},
        max_model_len=16384,
        max_num_seqs=40,
        max_num_batched_tokens=256,
        enable_prefix_caching=False,
        enforce_eager=False,
        block_size=32,
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [24],
            "cache_dir": str(out / "vllm-cache"),
        },
        additional_config={
            "ascend_compilation_config": {
                "enable_npugraph_ex": True,
                "enable_static_kernel": True,
                "fuse_norm_quant": True,
                "align_decode_capture_sizes": False,
            },
            "enable_cpu_binding": False,
            "multistream_overlap_shared_expert": True,
            "weight_nz_mode": 2,
            "enable_kv_nz": False,
            "enable_dsa_cp": False,
            "multistream_dsv4_dsa_overlap": True,
        },
    ).create_engine_config()
    record["native_source"] = os.environ["SOURCE_ROOT"]
    record["sdk_release"] = os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"]
    record["stage_source_sha256"] = {
        name: hashlib.sha256((source / "deepseek_v4_flash_csa_spmd_tnd" / name).read_bytes()).hexdigest()
        for name in (
            "entry.py",
            "cube_schedule.py",
            "vector_schedule.py",
            "hc_stages.py",
            "hc_input_norm_stages.py",
            "hc_input_rms_stages.py",
            "hc_native_projection_stages.py",
        )
    }
    with native_session(config, 0), torch.inference_mode():
        layer, details = make_layer(config, checkpoint, torch.device("npu:0"), 4)
        record["weights"] = details
        f = make_ragged_fixture(config, layer.self_attn, 4, 8192, 1024, torch.device("npu:0"), [6] * 4)
        groups = {name: (f["metadata"][g["prefix"]], g["views"]) for name, g in f["groups"].items()}
        weights = nal_adapter.prepare_weights(
            layer.self_attn, groups["indexer"][0].hadamard, layer, root_function=nal_entry.decode_csa_spmd
        )
        record["weight_contract"] = {
            name: {
                "shape": list(weights[name].shape),
                "dtype": str(weights[name].dtype),
                "format": int(torch_npu.get_npu_format(weights[name])),
            }
            for name in ("wq_a", "wq_b", "wo_a", "wo_b", "wkv", "idx_wq_b", "weights_proj")
        }
        output = torch.empty_like(f["hidden"])
        topk = torch.empty((24, 512), dtype=torch.int32, device="npu")
        scores = torch.empty((24, 512), dtype=torch.float32, device="npu")
        native = nal_native.bind_native_inputs(
            weights,
            f["hidden"],
            f["positions"],
            groups,
            layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
            compact_metadata=f["compact"],
            output=output,
            topk=topk,
            topk_scores=scores,
        )
        prepared = nal_binding.PreparedDynamicCSA(native, max_query_tokens=6, max_seq_len=8198)
        record["policy"] = prepared.policy.constexprs()
        shapes = {
            "inv": (32, 1),
            "post": (24, 4),
            "comb": (24, 16),
            "mixed": (24, 4096),
            "normed": (24, 4096),
            "qa": (32, 1024),
            "qr": (24, 1024),
            "qr_scale": (24, 1),
        }
        observed = {
            name: torch.full(
                shape,
                -37,
                dtype=torch.int8 if name == "qr" else torch.bfloat16 if name in ("mixed", "normed") else torch.float32,
                device="npu",
            )
            for name, shape in shapes.items()
        }
        pypto.torch.init()
        initial = {
            "hidden": f["hidden"].cpu(),
            "positions": f["positions"].cpu(),
            **{"cache_" + name: group["initial"] for name, group in f["groups"].items()},
            **{"readonly_" + name: snapshot for name, (_, snapshot) in f["readonly"].items()},
        }
        reference_path = os.environ.get("SPMD_PRODUCTION_REFERENCE")
        if reference_path:
            reference = torch.load(reference_path, map_location="cpu", weights_only=True)
            record["input_equivalence"] = {
                name: metric(value, reference["initial"][name]) for name, value in initial.items()
            }
            assert all(row["bit_equal"] for row in record["input_equivalence"].values()), "cross-task inputs differ"
            assert record["stage_source_sha256"] == reference["source_sha256"], "cross-task sources differ"
            production = reference["state"]
        else:
            restore(f)
            nal_entry.decode_csa_spmd_debug(**prepared.buffers, **prepared.policy.constexprs())
            torch.npu.synchronize()
            eager_production = collect_state(f, output, topk)
            production_graph = torch.npu.NPUGraph()
            production_stream = torch.npu.Stream()
            production_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.graph(production_graph, stream=production_stream):
                nal_entry.decode_csa_spmd_debug(**prepared.buffers, **prepared.policy.constexprs())
            restore(f)
            output.fill_(float("nan"))
            topk.fill_(-12345)
            production_graph.replay()
            torch.npu.synchronize()
            production = collect_state(f, output, topk)
            record["production_replay_eager"] = {
                name: metric(value, eager_production[name]) for name, value in production.items()
            }
            assert all(row["bit_equal"] for row in record["production_replay_eager"].values())
            record["guards"] = guard_checks(f)
            assert all(row["status"] == "PASS" for row in record["guards"].values())
            torch.save(
                {"initial": initial, "state": production, "source_sha256": record["stage_source_sha256"]},
                out / "production.pt",
            )
            record.update(
                complete=True,
                pass_kind="production FULL graph fixture snapshot, not Native precision",
                phase="production",
            )
            raise SystemExit(0)

        def run():
            return spmd_stage_probe(
                **prepared.buffers,
                **{"debug_" + name: value for name, value in observed.items()},
                **prepared.policy.constexprs(),
            )

        restore(f)
        run()
        torch.npu.synchronize()
        diagnostic = collect_state(f, output, topk)
        record["observation_gate"] = {name: metric(value, production[name]) for name, value in diagnostic.items()}
        save()
        assert all(row["bit_equal"] for row in record["observation_gate"].values()), (
            "observed root changes production external outputs"
        )
        eager_stages = {name: value.cpu().clone() for name, value in observed.items()}
        restore(f)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            run()
        restore(f)
        output.fill_(float("nan"))
        topk.fill_(-12345)
        for value in observed.values():
            value.fill_(-37)
        graph.replay()
        torch.npu.synchronize()
        replay = collect_state(f, output, topk)
        record["replay_eager"] = {name: metric(value, diagnostic[name]) for name, value in replay.items()}
        record["stage_replay_eager"] = {name: metric(value, eager_stages[name]) for name, value in observed.items()}
        record["guards"] = guard_checks(f)
        save()
        assert all(row["bit_equal"] for row in record["replay_eager"].values())
        assert all(row["bit_equal"] for row in record["stage_replay_eager"].values())
        assert all(row["status"] == "PASS" for row in record["guards"].values())
        record["native_determinism"] = {
            "torch": torch.are_deterministic_algorithms_enabled(),
            "npu_level": torch_npu._C._npu_get_deterministic_level(),
            "HCCL_DETERMINISTIC": os.environ["HCCL_DETERMINISTIC"],
        }
        assert record["native_determinism"] == {"torch": True, "npu_level": 1, "HCCL_DETERMINISTIC": "true"}
        mixed, post, comb = layer.hc_pre(f["hidden"], layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
        check("hc_mixed", observed["mixed"], mixed)
        check("hc_post_gate", observed["post"], post)
        check("hc_comb", observed["comb"], comb.reshape(24, 16))
        normed = layer.input_layernorm(mixed)
        if isinstance(normed, tuple):
            normed = normed[0]
        check("input_norm", observed["normed"], normed)
        impl = layer.self_attn.dsa_attn.dsa_attn.impl
        # Use the actual multistream Native boundary, including CVLinearWrapper.
        q_quant, q_scale = impl.cv_wq_a.quantize(normed)
        qa = impl.cv_wq_a.matmul(q_quant, q_scale)
        if isinstance(qa, tuple):
            qa = qa[0]
        check("q_a_projection_bf16", observed["qa"][:24].to(torch.bfloat16), qa)
        qr, scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(qa, impl.q_norm.weight, epsilon=impl.eps)
        payload["qr_gamma"] = impl.q_norm.weight.cpu()
        payload["qr_native_scale"] = scale.reshape(24, 1).cpu()
        payload["qa_accumulator"] = observed["qa"].cpu()
        check("q_r_quantized", observed["qr"], qr)
        check("q_r_scale", observed["qr_scale"], scale.reshape(24, 1))
        record["pass"] = True
        record["complete"] = True
except SystemExit as error:
    if error.code:
        raise
except FirstDifference:
    record["complete"] = True
except BaseException:
    record["error"] = traceback.format_exc()
    raise
finally:
    record["elapsed_s"] = time.monotonic() - start
    save()
