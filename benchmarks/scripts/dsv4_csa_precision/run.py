# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Deterministic Graph snapshots for one isolated rounding factor."""

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import traceback
from pathlib import Path

import fixture


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
    ).hexdigest()


def weight_snapshot(attention):
    import torch_npu

    return {
        name: torch_npu.npu_format_cast(value.detach(), 2).cpu().clone() for name, value in attention.named_parameters()
    }


def check_weights(attention, initial):
    import torch

    now = weight_snapshot(attention)
    changed = [name for name in initial if not torch.equal(initial[name], now[name])]
    if changed:
        raise AssertionError(f"Weights changed: {changed}")


def capture_once(call, reset):
    import torch

    reset()
    for _ in range(3):
        call()
    torch.npu.synchronize()
    reset()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    torch.npu.synchronize()
    reset()
    graph.replay()
    torch.npu.synchronize()
    return graph


def qkv(args, cfg, attention):
    import torch
    import torch_npu
    from pypto.torch import init, register

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention import pto_attn as p

    impl = attention.dsa_attn.dsa_attn.impl
    layer = attention.dsa_attn.dsa_attn.layer_name
    initial_weights = weight_snapshot(attention)
    data = fixture.build_payload(args)
    md = fixture.build_metadata(args, data, args.start_pos)
    names = [
        layer,
        impl.compressor.state_cache.prefix,
        impl.indexer.compressor.state_cache.prefix,
        impl.indexer.k_cache.prefix,
        impl.swa_cache_layer.prefix,
    ]
    metadata = dict(zip(names, md))
    t = args.batch * args.seq
    saved = {}
    original_prolog = impl._mla_prolog_single_stream
    original_o = impl._forward_o_proj

    def prolog(*a, **kw):
        out = original_prolog(*a, **kw)
        for key, value in zip(("q", "qr", "qr_scale"), out):
            saved[key] = value.clone()
        return out

    def oproj(heads, output):
        saved["heads"] = heads.clone()
        return original_o(heads, output)

    impl._mla_prolog_single_stream = prolog
    impl._forward_o_proj = oproj
    os.environ["VLLM_ASCEND_PYPTO_DSV4_CSA"] = "0"

    def reset(path):
        for dst, src in zip(data[path + "_roots"], data["initial"]):
            dst.copy_(src)
        torch.npu.synchronize()

    def native():
        with set_ascend_forward_context(metadata, cfg, num_tokens=t):
            impl.forward(layer, data["x"], data["native"], metadata, output=data["native_out"])

    native_graph = capture_once(native, lambda: reset("native"))
    check_weights(attention, initial_weights)
    native_cpu = {name: value.cpu() for name, value in saved.items()}
    native_cpu["output"] = data["native_out"].cpu()
    slots = md[4].req_metadata.slot_mapping.long()
    native_cpu["raw_kv"] = data["native"][1][slots[:, 0], slots[:, 1]].reshape(t, 512).cpu()
    kc, _ = p.kernel()
    specs = [
        ("q", (t, 64, 512), torch.bfloat16),
        ("kv", (t, 512), torch.bfloat16),
        ("qr", (t, 1024), torch.int8),
        ("qr_scale", (t, 1), torch.float32),
        ("scores", (t, 512), torch.float32),
        ("topk", (t, 512), torch.int32),
        ("heads", (kc.O_GROUPS * kc.T_PAD, kc.O_GROUP_IN), torch.bfloat16),
    ]
    debug = {n: torch.empty(shape, device=data["x"].device, dtype=dtype) for n, shape, dtype in specs}
    init()
    op = register(kc.decode_csa_attn_tp1_test, "single_factor::csa")
    with set_ascend_forward_context(metadata, cfg, num_tokens=t):
        bound, _ = p.build_args(
            impl, data["x"], data["pto"], impl._get_layer_metadata(layer, metadata), 6, layer, output=data["pto_out"]
        )
    pto_graph = capture_once(lambda: op(*bound, *debug.values()), lambda: reset("pto"))
    check_weights(attention, initial_weights)
    pto_cpu = {name: value.cpu() for name, value in debug.items()}
    pto_cpu["heads"] = pto_cpu["heads"].reshape(8, kc.T_PAD, 4096)[:, :t].permute(1, 0, 2).reshape(t, 64, 512)
    pto_cpu["raw_kv"] = pto_cpu.pop("kv")
    pto_cpu["output"] = data["pto_out"].cpu()
    metrics = {name: fixture.difference(pto_cpu[name].reshape_as(ref), ref) for name, ref in native_cpu.items()}
    cache_dump = {}
    for i, name in enumerate(("compressed", "raw", "main_state", "inner_state", "index_key", "index_scale")):
        group = [0, 4, 1, 2, 3, 3][i]
        req = md[group].req_metadata
        if group in (0, 3):
            comp = impl.compressor if group == 0 else impl.indexer.compressor
            with set_ascend_forward_context(metadata, cfg, num_tokens=t):
                cache_slots = comp._compute_metadata(req)[2]
        else:
            cache_slots = req.slot_mapping
        cache_slots = cache_slots.long()
        cache_slots = cache_slots[(cache_slots[:, 0] >= 0) & (cache_slots[:, 1] >= 0)]
        values = {path: data[path][i][cache_slots[:, 0], cache_slots[:, 1]].cpu() for path in ("native", "pto")}
        cache_dump[name] = dict(slots=cache_slots.cpu(), **values)
        metrics["cache_" + name] = fixture.difference(values["pto"], values["native"])
    # Reuse the exact native Q-B -> Q RMS -> RoPE sequence with CSA's QR/scale.
    from vllm_ascend.device.device_op import DeviceOperator

    same_q = torch_npu.npu_quant_matmul(
        debug["qr"],
        impl.wq_b.weight,
        impl.wq_b.weight_scale,
        pertoken_scale=debug["qr_scale"].reshape(t),
        bias=impl.wq_b.bias,
        output_dtype=torch.bfloat16,
    ).unflatten(-1, (64, 512))
    same_q = DeviceOperator.apply_dsa_q_rms(same_q, impl.eps, impl.q_norm_without_weight)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        same_q.unsqueeze(1),
        md[0].req_metadata.cos[layer],
        md[0].req_metadata.sin[layer],
        rotary_mode="interleave",
        partial_slice=[448, 512],
    )
    same_out = torch.empty_like(data["native_out"])
    original_o(pto_cpu["heads"].to(data["x"].device), same_out)
    torch.npu.synchronize()
    metrics["q_same_qr_input"] = fixture.difference(debug["q"], same_q)
    metrics["output_same_heads"] = fixture.difference(data["pto_out"], same_out)
    assert bool((data["pto_output_guard"] == -12.5).all().cpu())
    check_weights(attention, initial_weights)
    fingerprints = {
        "hidden": digest(data["x"]),
        "initial_cache": [digest(x) for x in data["initial"]],
        "weights": {name: digest(x) for name, x in initial_weights.items()},
    }
    torch.save(
        dict(native=native_cpu, pto=pto_cpu, cache=cache_dump, same_input=dict(q=same_q.cpu(), output=same_out.cpu())),
        args.out_dir / "stages.pt",
    )
    return dict(
        metrics=metrics,
        inputs=fingerprints,
        native_hashes={k: digest(v) for k, v in native_cpu.items()},
        weight_unchanged=True,
        graph_replay=True,
        guard=True,
        cache_scope="all six caches, valid write slots; initial backing hashed",
        graphs_retained=bool(native_graph and pto_graph),
    )


def oproj(args, cfg, attention):
    import torch
    import torch_npu
    from pypto.torch import init, register

    from vllm_ascend.attention import pto_attn as p

    impl = attention.dsa_attn.dsa_attn.impl
    weights = weight_snapshot(attention)
    kc, _ = p.kernel()
    module = importlib.import_module("vllm_ascend.attention.pto_kernels.dspark.oproj_bench")
    init()
    op = register(module.oproj_bench, "single_factor::oproj")
    frozen = torch.load(args.frozen_stages, map_location="cpu", weights_only=True)
    # All factors consume byte-identical heads, including the native reference.
    heads = frozen["native"]["heads"].to("npu").contiguous()
    t = heads.shape[0]
    packed = torch.zeros((8, kc.T_PAD, 4096), dtype=torch.bfloat16, device="npu")
    packed[:, :t].copy_(heads.reshape(t, 8, 4096).transpose(0, 1))
    packed = packed.reshape(8 * kc.T_PAD, 4096)
    wa = impl.wo_a.weight.transpose(1, 2).contiguous()
    wb, scale = p._int8(impl.wo_b, (4096, 8192), 4096)
    guard = torch.full((t + 1, 4096), -12.5, dtype=torch.bfloat16, device="npu")
    output = guard[:t]
    ref = torch.empty_like(output)
    oa = torch.zeros((kc.T_PAD, 8192), dtype=torch.float32, device="npu")
    quant = torch.zeros((kc.T_PAD, 8192), dtype=torch.int8, device="npu")
    dq = torch.zeros((8, kc.T_PAD), dtype=torch.float32, device="npu")
    ng = capture_once(lambda: impl._forward_o_proj(heads, ref), lambda: None)
    pg = capture_once(lambda: op(packed, wa, wb, scale, output, oa, quant, dq), lambda: None)
    native_oa = torch_npu.npu_transpose_batchmatmul(
        heads.reshape(t, 8, 4096),
        impl.wo_a.weight,
        bias=None,
        scale=None,
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
        batch_split_factor=1,
    ).reshape(t, 8192)
    native_quant, native_scale = torch_npu.npu_dynamic_quant(native_oa)
    effective_oa = oa[:t].bfloat16() if "16" in args.variant_dir.name else oa[:t]
    metrics = dict(
        output=fixture.difference(output, ref),
        oa=fixture.difference(effective_oa, native_oa),
        quant=fixture.difference(quant[:t], native_quant),
        scale=fixture.difference(dq[:, :t].T, native_scale.reshape(t, 1).expand(t, 8)),
    )
    # Upstream QR/Q/KV/heads/cache are frozen artifacts, not recomputed per factor.
    for key in ("qr", "qr_scale", "q", "raw_kv", "heads"):
        metrics[key] = fixture.difference(frozen["native"][key], frozen["native"][key])
    for name, values in frozen["cache"].items():
        metrics["cache_" + name] = fixture.difference(values["native"], values["native"])
    check_weights(attention, weights)
    assert bool((guard[t:] == -12.5).all().cpu())
    torch.save(
        dict(
            native=dict(oa=native_oa.cpu(), quant=native_quant.cpu(), scale=native_scale.cpu(), output=ref.cpu()),
            pto=dict(oa=oa[:t].cpu(), quant=quant[:t].cpu(), scale=dq[:, :t].cpu(), output=output.cpu()),
            frozen_stages=str(args.frozen_stages),
            heads_sha256=digest(heads),
        ),
        args.out_dir / "stages.pt",
    )
    return dict(
        metrics=metrics,
        heads_sha256=digest(heads),
        weights_sha256={name: digest(value) for name, value in weights.items()},
        frozen_stages=str(args.frozen_stages),
        frozen_sha256=hashlib.sha256(args.frozen_stages.read_bytes()).hexdigest(),
        weight_unchanged=True,
        graph_replay=True,
        guard=True,
        upstream_status="frozen native QR/Q/raw KV/heads/cache referenced from control artifact",
        graphs_retained=bool(ng and pg),
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "extension-dir", "sdk-root", "variant-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frozen-stages", type=Path)
    parser.add_argument("--start-pos", type=int, default=8186)
    parser.add_argument("--mode", choices=("qkv", "oproj"), default="qkv")
    args = parser.parse_args()
    args.variant_dir = args.variant_dir.resolve()
    args.device = 0
    args.batch = 4
    args.seq = 6
    args.steps = 1
    args.layer = 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(variant=args.variant_dir.name, mode=args.mode, start_pos=args.start_pos, batch=4, seq=6, layer=2)
    record["source_sha256"] = {
        str(path.relative_to(args.variant_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.variant_dir.rglob("*.py"))
    }
    record["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        with contextlib.ExitStack() as stack:
            cfg = fixture.initialize(args, stack)
            import pypto
            import simpler
            import torch
            import torch_npu
            import vllm
            from pypto.torch import shutdown

            original = shutdown.require_supported_framework

            def gate(framework):
                if framework.__version__.split("+")[0] != "2.10.0.post4":
                    return original(framework)
                assert callable(framework._C._npu_shutdown_synchronize) and callable(framework._C._npu_shutdown)

            shutdown.require_supported_framework = gate
            stack.enter_context(torch.inference_mode())
            attention, record["weights"] = fixture.load_attention(args, cfg)
            record["runtime"] = dict(
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                vllm=vllm.__version__,
                pypto=pypto.__file__,
                simpler=simpler.__file__,
                cann=os.environ.get("ASCEND_HOME_PATH"),
                deterministic_level=torch_npu.npu._get_deterministic_level(),
                HCCL_DETERMINISTIC=os.environ.get("HCCL_DETERMINISTIC"),
            )
            record["result"] = (qkv if args.mode == "qkv" else oproj)(args, cfg, attention)
    except Exception:
        record["error"] = traceback.format_exc()
    (args.out_dir / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    return int("error" in record)


if __name__ == "__main__":
    raise SystemExit(main())
