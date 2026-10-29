# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind Native control and the independent PTO namespace before imports.
"""8K/B4完整SPMD运行及前缀观测；从HC逐阶段比对Native，首差停止。"""

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--lengths", default="6,6,6,6")
options = parser.parse_args()
request_lengths = [int(value) for value in options.lengths.split(",")]
assert 1 <= len(request_lengths) <= 40 and all(1 <= value <= 6 for value in request_lengths)
tokens, batch = sum(request_lengths), len(request_lengths)
assert tokens <= 256
p8, p16, p64, p128 = ((tokens + unit - 1) // unit * unit for unit in (8, 16, 64, 128))
out = options.output
record = {
    "complete": False,
    "pass": False,
    "scope": "complete SPMD; Native eager prefix/cache/query/weights/TopK/output stages; not whole-model tokens",
    "history": 8192,
    "lengths": request_lengths,
    "tokens": tokens,
    "request_padding_rows": 0,
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
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
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
            "cudagraph_capture_sizes": [tokens],
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
            "q_native_stages.py",
            "qkv_stages.py",
            "kv_stages.py",
            "kv_native_stages.py",
            "kv_precision_dispatch.py",
            "projection_overlap_stages.py",
            "kv_native_norm_stages.py",
            "vector_prefix_stages.py",
            "query_projection_stages.py",
            "q_native_output_stages.py",
            "main_native_projection_stages.py",
            "indexer_native_projection_stages.py",
            "indexer_cache_stages.py",
            "indexer_native_query_stages.py",
            "indexer_score_stages.py",
            "indexer_topk_stages.py",
            "indexer_publish_stages.py",
            "attention_native_stream_stages.py",
            "attention_plan_stages.py",
            "output_projection_a_stages.py",
            "output_projection_b_stages.py",
            "compressor_pool_stages.py",
            "compressor_output_stages.py",
        )
    }
    with native_session(config, 0), torch.inference_mode():
        layer, details = make_layer(config, checkpoint, torch.device("npu:0"), 4)
        record["weights"] = details
        f = make_ragged_fixture(config, layer.self_attn, batch, 8192, 1024, torch.device("npu:0"), request_lengths)
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
        topk = torch.empty((tokens, 512), dtype=torch.int32, device="npu")
        scores = torch.empty((tokens, 512), dtype=torch.float32, device="npu")
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
        prepared = nal_binding.PreparedDynamicCSA(
            native, max_query_tokens=max(request_lengths), max_seq_len=8192 + max(request_lengths)
        )
        record["policy"] = prepared.policy.constexprs()
        shapes = {
            "inv": (p16, 1),
            "post": (tokens, 4),
            "comb": (tokens, 16),
            "mixed": (p8, 4096),
            "normed": (tokens, 4096),
            "qa": (p16, 1024),
            "qr": (tokens, 1024),
            "qr_scale": (tokens, 1),
            "kv_fp32": (p16, 512),
            "kv": (tokens, 512),
            "q_proj": (p128, 32768),
            "q": (tokens, 32768),
            "main_pooled": (tokens, 512),
            "main_normed": (tokens, 512),
            "main_debug": (tokens, 512),
            "idx_pooled": (tokens, 128),
            "idx_normed": (p16, 128),
            "idx_had": (p16, 128),
            "idx_scales": (p16, 1),
            "idx_debug": (tokens, 128),
            "idx_query_i8": (tokens * 64, 128),
            "idx_query_scale": (tokens * 64, 1),
            "idx_query_acc": (p64, 8192),
            "idx_query_rope": (p16, 8192),
            "idx_query_had": (tokens * 64, 128),
            "head_weights": (p16, 64),
            "packed": (8 * p128, 4096),
            "o_r": (
                max(p128, (tokens + prepared.policy.ob_rows - 1) // prepared.policy.ob_rows * prepared.policy.ob_rows),
                8192,
            ),
        }
        observed = {
            name: torch.full(
                shape,
                -37,
                dtype=torch.int8
                if name in ("qr", "idx_query_i8")
                else torch.int32
                if name in ("q_proj", "idx_query_acc")
                else torch.bfloat16
                if name in ("mixed", "normed", "kv", "q", "idx_normed", "idx_query_rope", "packed")
                else torch.float32,
                device="npu",
            )
            for name, shape in shapes.items()
        }
        ring_heap = (256 << 20, 512 << 20, 256 << 20, 256 << 20)
        record["runtime_ring_heap_bytes"] = ring_heap
        pypto.torch.init(ring_heap=ring_heap)
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
            eager_topk_scores = scores.cpu().clone()
            production_graph = torch.npu.NPUGraph()
            production_stream = torch.npu.Stream()
            production_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.graph(production_graph, stream=production_stream):
                nal_entry.decode_csa_spmd_debug(**prepared.buffers, **prepared.policy.constexprs())
            restore(f)
            output.fill_(float("nan"))
            topk.fill_(-12345)
            scores.fill_(float("nan"))
            production_graph.replay()
            torch.npu.synchronize()
            production = collect_state(f, output, topk)
            production_topk_scores = scores.cpu().clone()
            record["production_topk_score_replay_eager"] = metric(production_topk_scores, eager_topk_scores)
            assert record["production_topk_score_replay_eager"]["bit_equal"]
            record["production_replay_eager"] = {
                name: metric(value, eager_production[name]) for name, value in production.items()
            }
            assert all(row["bit_equal"] for row in record["production_replay_eager"].values())
            record["guards"] = guard_checks(f)
            assert all(row["status"] == "PASS" for row in record["guards"].values())
            torch.save(
                {
                    "initial": initial,
                    "state": production,
                    "source_sha256": record["stage_source_sha256"],
                    "topk_scores": production_topk_scores,
                },
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
        if reference_path and "topk_scores" in reference:
            record["observation_gate"]["topk_scores"] = metric(scores, reference["topk_scores"])
            record["topk_score_controlled"] = record["observation_gate"]["topk_scores"]["bit_equal"]
        save()
        assert all(row["bit_equal"] for row in record["observation_gate"].values()), (
            "observed root changes production external outputs"
        )
        # These new compact scratches have unpublished rows. Compare only the
        # real closed-token/valid-slot domain in both eager and FULL; the whole
        # production output/cache and guards are checked separately above/below.
        idx_slots_for_domain = f["compact"]["indexer"][2]
        valid_compact = (idx_slots_for_domain >= 0).all(dim=-1)
        closed_tokens = ((f["positions"] + 1) % 4) == 0
        assert int(closed_tokens.sum()) == int(valid_compact.sum())
        record["indexer_observation_domain"] = {
            "compact_rows": valid_compact.nonzero().flatten().cpu().tolist(),
            "token_rows": closed_tokens.nonzero().flatten().cpu().tolist(),
        }

        def observed_domain(name, value):
            if name == "packed":
                return value.reshape(8, value.shape[0] // 8, 8, 512)[:, :tokens]
            if name in ("mixed", "idx_query_acc", "idx_query_rope", "head_weights", "o_r"):
                return value[:tokens]
            if name in ("idx_normed", "idx_had", "idx_scales"):
                return value[: idx_slots_for_domain.shape[0]][valid_compact]
            if name in ("main_pooled", "main_normed", "main_debug", "idx_pooled", "idx_debug"):
                return value[closed_tokens]
            return value

        eager_stages = {name: observed_domain(name, value).cpu().clone() for name, value in observed.items()}
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
        record["stage_replay_eager"] = {
            name: metric(observed_domain(name, value), eager_stages[name]) for name, value in observed.items()
        }
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
        check("hc_mixed", observed["mixed"][:tokens], mixed)
        check("hc_post_gate", observed["post"], post)
        check("hc_comb", observed["comb"], comb.reshape(tokens, 16))
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
        payload["qa_accumulator"] = observed["qa"].cpu()
        check("q_a_projection_bf16", observed["qa"][:tokens].to(torch.bfloat16), qa)
        qr, scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(qa, impl.q_norm.weight, epsilon=impl.eps)
        payload["qr_gamma"] = impl.q_norm.weight.cpu()
        payload["qr_native_scale"] = scale.reshape(tokens, 1).cpu()
        payload["qa_accumulator"] = observed["qa"].cpu()
        check("q_r_quantized", observed["qr"], qr)
        check("q_r_scale", observed["qr_scale"], scale.reshape(tokens, 1))
        kv_quant, kv_scale = impl.cv_wkv.quantize(normed)
        kv = impl.cv_wkv.matmul(kv_quant, kv_scale)
        if isinstance(kv, tuple):
            kv = kv[0]
        payload["kv_gamma"] = impl.kv_norm.weight.cpu()
        payload["kv_accumulator"] = observed["kv_fp32"].cpu()
        check("wkv_projection_bf16", observed["kv_fp32"][:tokens].bfloat16(), kv)
        kv = impl.kv_norm(kv)
        if isinstance(kv, tuple):
            kv = kv[0]
        # Save before the in-place RoPE mutates this Native RMSNorm boundary.
        payload["kv_normalized_native"] = kv.cpu()
        kv = kv.reshape(tokens, 1, 512)
        cos, sin = (f["readonly"]["rope." + name][0] for name in ("cos", "sin"))
        payload["kv_rope_cos"] = cos.cpu()
        payload["kv_rope_sin"] = sin.cpu()
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1), cos, sin, rotary_mode="interleave", partial_slice=[impl.nope_head_dim, impl.head_dim]
        )
        check("original_kv_norm_rope", observed["kv"], kv.reshape(tokens, 512))
        # Same W8A8 decode branch as Native _forward_qkv_multistream.
        q = torch_npu.npu_quant_matmul(
            qr,
            impl.wq_b.weight,
            impl.wq_b.weight_scale,
            pertoken_scale=scale,
            bias=impl.wq_b.bias,
            output_dtype=normed.dtype,
        ).unflatten(-1, (impl.n_local_heads, impl.head_dim))
        payload["q_projection_native"] = q.cpu()
        payload["q_accumulator"] = observed["q_proj"].cpu()
        payload["q_weight_scale_native"] = impl.wq_b.weight_scale.cpu()
        payload["q_weight_scale_prepared"] = weights["wq_b_scale"].cpu()
        # Use the operator already selected by this Native implementation. A top-level
        # device_op import precedes Native ops initialization and creates an import cycle.
        native_device_operator = sys.modules[type(impl).__module__].DeviceOperator
        q = native_device_operator.apply_dsa_q_rms(q, impl.eps, impl.q_norm_without_weight)
        payload["q_normalized_native"] = q.cpu()
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1), cos, sin, rotary_mode="interleave", partial_slice=[impl.nope_head_dim, impl.head_dim]
        )
        check("q_norm_rope", observed["q"], q.reshape(tokens, 32768))
        # Only advance once every preceding Native module boundary is bit-exact.
        # This calls the frozen Native compressor, not a reconstructed Python golden.
        main_metadata = f["metadata"][f["groups"]["compressed"]["prefix"]].decode
        state_metadata = f["metadata"][f["groups"]["state"]["prefix"]].decode
        main_state = f["groups"]["state"]["views"][0]
        main_cos, main_sin, main_slots = f["compact"]["compressed"]
        record["main_compressor_contract"] = {
            "state_shape": list(main_state.shape),
            "state_stride": list(main_state.stride()),
            "state_storage_offset_bytes": main_state.storage_offset() * main_state.element_size(),
            "allocation_bytes": f["groups"]["state"]["allocation"].numel(),
            "bounds": main_metadata.query_start_loc.cpu().tolist(),
            "start_pos": main_metadata.start_pos.cpu().tolist(),
            "compact_slots_shape": list(main_slots.shape),
            "negative_slot_rows": int((main_slots < 0).any(dim=-1).sum().cpu()),
            "norm_dtype": str(impl.compressor_norm.weight.dtype),
            "coff": 2 if impl.compressor_overlap else 1,
        }
        record["native_compressor_guards"] = []

        def native_main_compressor():
            restore(f)
            torch.npu.synchronize()
            payload["main_state_initial"] = main_state.cpu()
            compressed = torch.ops._C_ascend.compressor(
                normed,
                impl.compressor_wkv.weight,
                impl.compressor_wgate.weight,
                main_state.squeeze(-2),
                impl.compressor_ape,
                impl.compressor_norm.weight,
                main_sin.view(-1, main_sin.shape[-1]),
                main_cos.view(-1, main_cos.shape[-1]),
                state_block_table=state_metadata.block_table,
                cu_seqlens=main_metadata.query_start_loc,
                seqused=None,
                start_pos=main_metadata.start_pos,
                rope_head_dim=impl.rope_head_dim,
                cmp_ratio=impl.compress_ratio,
                coff=2 if impl.compressor_overlap else 1,
                norm_eps=impl.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )
            torch.npu.synchronize()
            guards = guard_checks(f)
            record["native_compressor_guards"].append(guards)
            save()
            assert all(row["status"] == "PASS" for row in guards.values())
            return main_state.cpu(), compressed

        state_a, compressed_a = native_main_compressor()
        state_b, compressed_b = native_main_compressor()
        assert compressed_a.shape == compressed_b.shape == (main_slots.shape[0], 512)
        valid_main_rows = (main_slots >= 0).all(dim=-1)
        # Native only publishes compact rows with valid slots. Keep the full-buffer
        # bit evidence, including unpublished rows, without treating padding as gold.
        output_row_differences = (
            compressed_a.cpu().contiguous().view(torch.int16) != compressed_b.cpu().contiguous().view(torch.int16)
        ).sum(dim=-1)
        record["main_native_output_rows"] = {
            "slots": main_slots.cpu().tolist(),
            "different_elements": output_row_differences.tolist(),
            "valid_rows": valid_main_rows.cpu().tolist(),
        }
        record["main_native_repeat"] = {
            "state": metric(state_a, state_b),
            "published_output": metric(compressed_a[valid_main_rows], compressed_b[valid_main_rows]),
        }
        payload["main_state_native"] = state_b
        payload["main_compressed_native_repeat_a"] = compressed_a.cpu()
        payload["main_compressed_native"] = compressed_b.cpu()
        payload["main_compact_slots"] = main_slots.cpu()
        payload["main_wkv_native"] = impl.compressor_wkv.weight.cpu()
        payload["main_wgate_native"] = impl.compressor_wgate.weight.cpu()
        payload["main_ape_native"] = impl.compressor_ape.cpu()
        payload["main_gamma_native"] = impl.compressor_norm.weight.cpu()
        record["main_compressor_contract"]["output_shape"] = list(compressed_b.shape)
        save()
        assert all(row["bit_equal"] for row in record["main_native_repeat"].values())
        payload["main_observed_pooled"] = observed["main_pooled"].cpu()
        payload["main_observed_normed"] = observed["main_normed"].cpu()
        payload["main_observed_debug"] = observed["main_debug"].cpu()
        payload["main_compact_cos"] = main_cos.cpu()
        payload["main_compact_sin"] = main_sin.cpu()
        payload["main_positions"] = f["positions"].cpu()
        payload["main_closed_tokens"] = closed_tokens.cpu()
        record["main_compressor_contract"]["epsilon"] = impl.compressor_norm_eps
        save()
        check("main_compressor_state", replay["state.0"], state_b)
        # Match Native scatter unchanged. Its two full outputs differ in padding;
        # prove those rows do not affect the entire backing cache or its guards.
        scatter_raw, scatter_views = [], []
        record["native_scatter_guards"] = []
        for compressed in (compressed_a, compressed_b):
            restore(f)
            if compressed.shape[0] > 0:
                native_device_operator.dsa_kv_compress_scatter(
                    f["groups"]["compressed"]["views"][0], compressed, main_slots
                )
            torch.npu.synchronize()
            guards = guard_checks(f)
            record["native_scatter_guards"].append(guards)
            save()
            assert all(row["status"] == "PASS" for row in guards.values())
            scatter_raw.append(f["groups"]["compressed"]["allocation"].cpu())
            scatter_views.append(f["groups"]["compressed"]["views"][0].cpu())
        record["main_native_scatter_repeat"] = metric(scatter_raw[0], scatter_raw[1])
        save()
        assert record["main_native_scatter_repeat"]["bit_equal"]
        check("main_compressed_cache", replay["compressed.0"], scatter_views[1])
        # Indexer is independent of the main compressor. Advance only after both
        # main state and cache pass; do not mix Hadamard/quantization into this gate.
        idx_metadata = f["metadata"][f["groups"]["indexer"]["prefix"]].decode
        idx_state_metadata = f["metadata"][f["groups"]["indexer_state"]["prefix"]].decode
        idx_state = f["groups"]["indexer_state"]["views"][0]
        idx_cos, idx_sin, idx_slots = f["compact"]["indexer"]
        record["indexer_compressor_contract"] = {
            "state_shape": list(idx_state.shape),
            "state_stride": list(idx_state.stride()),
            "state_storage_offset_bytes": idx_state.storage_offset() * idx_state.element_size(),
            "allocation_bytes": f["groups"]["indexer_state"]["allocation"].numel(),
            "bounds_source": "compressed.decode.query_start_loc, as forwarded by Native",
            "bounds": main_metadata.query_start_loc.cpu().tolist(),
            "start_pos": idx_metadata.start_pos.cpu().tolist(),
            "slots": idx_slots.cpu().tolist(),
        }
        record["indexer_native_guards"] = []

        def native_indexer_compressor():
            restore(f)
            torch.npu.synchronize()
            payload["indexer_state_initial"] = idx_state.cpu()
            compressed = torch.ops._C_ascend.compressor(
                normed,
                impl.indexcom_wkv.weight,
                impl.indexcom_wgate.weight,
                idx_state.squeeze(-2),
                impl.indexcom_ape,
                impl.indexcom_norm.weight,
                idx_sin.view(-1, idx_sin.shape[-1]),
                idx_cos.view(-1, idx_cos.shape[-1]),
                state_block_table=idx_state_metadata.block_table,
                cu_seqlens=main_metadata.query_start_loc,
                seqused=None,
                start_pos=idx_metadata.start_pos,
                rope_head_dim=impl.rope_head_dim,
                cmp_ratio=impl.compress_ratio,
                coff=2 if impl.compressor_overlap else 1,
                norm_eps=impl.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )
            torch.npu.synchronize()
            guards = guard_checks(f)
            record["indexer_native_guards"].append(guards)
            save()
            assert all(row["status"] == "PASS" for row in guards.values())
            return idx_state.cpu(), compressed

        idx_state_a, idx_compressed_a = native_indexer_compressor()
        idx_state_b, idx_compressed_b = native_indexer_compressor()
        assert idx_compressed_a.shape == idx_compressed_b.shape == (idx_slots.shape[0], 128)
        valid_idx_rows = (idx_slots >= 0).all(dim=-1)
        record["indexer_native_repeat"] = {
            "state": metric(idx_state_a, idx_state_b),
            "published_output": metric(idx_compressed_a[valid_idx_rows], idx_compressed_b[valid_idx_rows]),
        }
        payload["indexer_compressed_native"] = idx_compressed_b.cpu()
        payload["indexer_wkv_native"] = impl.indexcom_wkv.weight.cpu()
        payload["indexer_wgate_native"] = impl.indexcom_wgate.weight.cpu()
        payload["indexer_ape_native"] = impl.indexcom_ape.cpu()
        payload["indexer_gamma_native"] = impl.indexcom_norm.weight.cpu()
        save()
        assert all(row["bit_equal"] for row in record["indexer_native_repeat"].values())
        check("indexer_compressor_state", replay["indexer_state.0"], idx_state_b)
        # Reuse the actual Native Hadamard and A3 quant/scatter primitives. Keep
        # unpublished/negative slots unchanged; establish their full-allocation
        # repeatability before comparing either PTO cache view.
        idx_outer = f["metadata"][f["groups"]["indexer"]["prefix"]]
        idx_key_cache, idx_scale_cache = f["groups"]["indexer"]["views"]
        native_module = sys.modules[type(impl).__module__]
        assert impl.indexcom_rotate, "this diagnostic qualifies the active Hadamard path"
        idx_linear, idx_shape, idx_dim = native_module.hadamard_linear(idx_compressed_b, idx_outer.hadamard)
        idx_scaled = native_module.hadamard_scale(idx_linear, idx_shape, idx_dim, scale=idx_dim**-0.5)
        idx_rotated = native_module.rotate_activation(idx_compressed_b, idx_outer.hadamard)
        record["indexer_native_hadamard_split_repeat"] = metric(idx_scaled[valid_idx_rows], idx_rotated[valid_idx_rows])
        assert record["indexer_native_hadamard_split_repeat"]["bit_equal"]
        payload["indexer_compact_slots"] = idx_slots.cpu()
        payload["indexer_hadamard_weight_native"] = idx_outer.hadamard.cpu()
        payload["indexer_hadamard_linear_native"] = idx_linear.cpu()
        payload["indexer_hadamard_scaled_native"] = idx_scaled.cpu()
        for name in ("idx_pooled", "idx_normed", "idx_had", "idx_scales", "idx_debug"):
            payload["observed_" + name] = observed[name].cpu()
        save()
        check(
            "indexer_norm_rope", observed_domain("idx_normed", observed["idx_normed"]), idx_compressed_b[valid_idx_rows]
        )
        check(
            "indexer_hadamard_linear_bf16",
            observed_domain("idx_had", observed["idx_had"]).bfloat16(),
            idx_linear[valid_idx_rows],
        )
        check(
            "indexer_hadamard_scaled",
            observed_domain("idx_debug", observed["idx_debug"]),
            idx_scaled[valid_idx_rows].float(),
        )
        idx_raw, idx_views = [], []
        record["indexer_native_scatter_guards"] = []
        for _ in range(2):
            _, idx_kv = native_indexer_compressor()
            assert idx_kv.numel() > 0
            if impl.indexcom_rotate:
                idx_kv = native_module.rotate_activation(idx_kv, idx_outer.hadamard)
            _, idx_scale = native_device_operator.indexer_quant_scatter_part1(idx_kv, idx_key_cache, None, idx_slots)
            assert idx_scale is not None, "this A3 test expects separate scale scatter"
            native_device_operator.dsa_indexer_scatter_scale_part3(idx_scale, idx_scale_cache, idx_slots)
            torch.npu.synchronize()
            guards = guard_checks(f)
            record["indexer_native_scatter_guards"].append(guards)
            save()
            assert all(row["status"] == "PASS" for row in guards.values())
            idx_raw.append(f["groups"]["indexer"]["allocation"].cpu())
            idx_views.append([view.cpu() for view in (idx_key_cache, idx_scale_cache)])
        record["indexer_native_scatter_repeat"] = metric(idx_raw[0], idx_raw[1])
        save()
        assert record["indexer_native_scatter_repeat"]["bit_equal"]
        check("indexer_key_cache", replay["indexer.0"], idx_views[1][0])
        check("indexer_scale_cache", replay["indexer.1"], idx_views[1][1])
        # Native gold now comes from the actual full forward consumer. Keep the
        # diagnostic import after Native initialization, as with device_op above.
        from native_qli_capture import capture_native_qli

        native_states, native_guards, qli_inputs, query_stages, weight_stages, output_stages = capture_native_qli(
            layer, f, config
        )
        record["native_qli_guards"] = native_guards
        record["native_qli_hook_control"] = {
            name: metric(value, native_states[0][name]) for name, value in native_states[1].items()
        }
        record["native_qli_options"] = {
            name: value for name, value in qli_inputs.items() if not isinstance(value, torch.Tensor)
        }
        payload["native_qli_inputs"] = qli_inputs
        payload["native_indexer_query_stages"] = query_stages
        payload["native_indexer_weight_stages"] = weight_stages
        payload["native_output_stages"] = output_stages
        payload["observed_o_r"] = observed["o_r"][:tokens].cpu()
        payload["indexer_query_weight_scale_native"] = impl.inderxer_wq_b.weight_scale.cpu()
        payload["indexer_query_weight_scale_prepared"] = prepared.buffers["idx_wq_b_scale"].cpu()
        for name in ("idx_query_acc", "idx_query_rope", "idx_query_had", "head_weights"):
            payload["observed_" + name] = observed[name].cpu()
        payload["native_full_output"] = native_states[1]["x_out"]
        payload["native_full_topk"] = native_states[1]["idx_topk"]
        payload["observed_topk_scores"] = scores.cpu().clone()
        payload["observed_idx_query_i8"] = observed["idx_query_i8"].cpu()
        payload["observed_idx_query_scale"] = observed["idx_query_scale"].cpu()
        save()
        assert all(row["status"] == "PASS" for group in native_guards for row in group.values())
        assert all(row["bit_equal"] for row in record["native_qli_hook_control"].values())
        for name in ("swa.0", "state.0", "compressed.0", "indexer_state.0", "indexer.0", "indexer.1"):
            check("native_forward_" + name, replay[name], native_states[1][name])
        check(
            "indexer_query_rope",
            observed["idx_query_rope"][:tokens].reshape_as(query_stages["rope"]),
            query_stages["rope"],
        )
        check("indexer_query_hadamard_bf16", observed["idx_query_had"].bfloat16(), query_stages["hadamard_linear"])
        qli_query, qli_scale = qli_inputs["query"], qli_inputs["query_dequant_scale"]
        assert qli_query.dtype == torch.int8 and qli_scale.dtype == torch.float16
        check("qli_query_int8", observed["idx_query_i8"].reshape_as(qli_query), qli_query)
        check("qli_query_scale_fp16", observed["idx_query_scale"].half().reshape_as(qli_scale), qli_scale)
        check("weights_projection_input", observed["normed"], weight_stages["input"])
        check("weights_prepare_repeat", weight_stages["consumer"], qli_inputs["weights"])
        check("qli_weights_fp16", observed["head_weights"][:tokens].half(), qli_inputs["weights"])
        check("qli_topk", replay["idx_topk"].reshape_as(native_states[1]["idx_topk"]), native_states[1]["idx_topk"])
        packed = observed["packed"].reshape(8, p128, 8, 512)[:, :tokens]
        heads = packed.permute(1, 0, 2, 3).reshape(tokens, 64, 512)
        check("attention_heads_after_inverse_rope", heads.reshape_as(output_stages["heads"]), output_stages["heads"])
        check("wo_a_bf16", observed["o_r"][:tokens].bfloat16(), output_stages["wo_a"].reshape(tokens, 8192))
        check("native_full_output", replay["x_out"].reshape_as(native_states[1]["x_out"]), native_states[1]["x_out"])
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
