# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Uninstrumented, paired Graph latency and accuracy for merged QKV rounding."""

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import statistics
import sys
import traceback
import types
from pathlib import Path

import fixture
from run import check_weights, digest, weight_snapshot


def run_case(args, cfg, attention):
    # Shared inputs, separate native/CSA cache roots, and serial graphs isolate
    # the five rounding factors. Snapshot collection never enters timed graphs.
    import torch
    from pypto.torch import init, register

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention import pto_attn as p

    impl = attention.dsa_attn.dsa_attn.impl
    layer = attention.dsa_attn.dsa_attn.layer_name
    initial_weights = weight_snapshot(attention)
    data = fixture.build_payload(args)
    md = fixture.build_metadata(args, data, args.start_pos)
    keys = [
        layer,
        impl.compressor.state_cache.prefix,
        impl.indexer.compressor.state_cache.prefix,
        impl.indexer.k_cache.prefix,
        impl.swa_cache_layer.prefix,
    ]
    metadata = dict(zip(keys, md))
    kc, _ = p.kernel()
    package = types.ModuleType("rounding_graph_aligned")
    package.__path__ = [str(args.variants / "aligned/pto_kernels/dspark")]
    sys.modules[package.__name__] = package
    candidate = importlib.import_module(package.__name__ + ".decode_csa")
    init()
    ops = {
        "baseline": register(kc.decode_csa_attn_tp1_test, "rounding_old::csa"),
        "aligned": register(candidate.decode_csa_attn_tp1_test, "rounding_new::csa"),
    }
    paths = ("native", "baseline", "aligned")

    def slot(path):
        return "native" if path == "native" else "pto"

    def reset(path):
        for dst, src in zip(data[slot(path) + "_roots"], data["initial"]):
            dst.copy_(src)
        torch.npu.synchronize()

    def invoke(path):
        os.environ["VLLM_ASCEND_PYPTO_DSV4_CSA"] = "0" if path == "native" else "1"
        if path != "native":
            p._registered = lambda: ops[path]
        before = p._RAN[0]
        with set_ascend_forward_context(metadata, cfg, num_tokens=args.batch * args.seq):
            impl.forward(layer, data["x"], data[slot(path)], metadata, output=data[slot(path) + "_out"])
        assert p._RAN[0] == before + (path != "native"), "CSA fallback or contaminated native"

    graphs, outputs, cache = {}, {}, {}
    cache_names = ("compressed", "raw", "main_state", "inner_state", "index_key", "index_scale")
    for path in paths:
        reset(path)
        for _ in range(3):
            invoke(path)
        torch.npu.synchronize()
        reset(path)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            invoke(path)
        torch.npu.synchronize()
        reset(path)
        data[slot(path) + "_out"].fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        outputs[path] = data[slot(path) + "_out"].cpu().clone()
        assert torch.isfinite(outputs[path]).all(), "Graph replay failed to write finite output"
        graphs[path] = graph
        check_weights(attention, initial_weights)
        cache[path] = {}
        for i, name in enumerate(cache_names):
            group = (0, 4, 1, 2, 3, 3)[i]
            req = md[group].req_metadata
            if group in (0, 3):
                comp = impl.compressor if group == 0 else impl.indexer.compressor
                with set_ascend_forward_context(metadata, cfg, num_tokens=args.batch * args.seq):
                    slots = comp._compute_metadata(req)[2]
            else:
                slots = req.slot_mapping
            slots = slots.long()
            slots = slots[(slots[:, 0] >= 0) & (slots[:, 1] >= 0)]
            cache[path][name] = dict(slots=slots.cpu(), values=data[slot(path)][i][slots[:, 0], slots[:, 1]].cpu())
        assert bool((data["pto_output_guard"].cpu() == -12.5).all()), "Output guard changed"
    samples = {name: [] for name in paths}
    for rep in range(args.rounds):
        order = paths[rep % 3 :] + paths[: rep % 3]
        if rep % 2:
            order = tuple(reversed(order))
        for path in order:
            reset(path)
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin.record()
            for _ in range(args.iterations):
                graphs[path].replay()
            end.record()
            torch.npu.synchronize()
            samples[path].append(begin.elapsed_time(end) / args.iterations)
    check_weights(attention, initial_weights)
    assert bool((data["pto_output_guard"].cpu() == -12.5).all()), "Output guard changed after timing"
    accuracy = {}
    for path in ("baseline", "aligned"):
        accuracy[path] = {"output": fixture.difference(outputs[path], outputs["native"])}
        for name in cache_names:
            assert torch.equal(cache[path][name]["slots"], cache["native"][name]["slots"])
            accuracy[path][name] = fixture.difference(cache[path][name]["values"], cache["native"][name]["values"])
        assert all(value["finite"] for value in accuracy[path].values()), path
    torch.save(dict(outputs=outputs, cache=cache), args.out_dir / "outputs.pt")
    return dict(
        accuracy=accuracy,
        graph={name: dict(median_ms=statistics.median(v), round_ms=v) for name, v in samples.items()},
        aligned_vs_baseline_round_pct=[100 * (a / b - 1) for a, b in zip(samples["aligned"], samples["baseline"])],
        weight_unchanged=True,
        guard=True,
        inputs=dict(
            hidden=digest(data["x"]),
            cache=[digest(t) for t in data["initial"]],
            weights={name: digest(value) for name, value in initial_weights.items()},
        ),
        output_hashes={name: digest(value) for name, value in outputs.items()},
        scope="one C4 attention layer, real weights, synthetic hidden/history, fixed metadata Graph replay",
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "extension-dir", "sdk-root", "variants", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start-pos", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    args.variants = args.variants.resolve()
    args.variant_dir = args.variants / "baseline"
    args.device, args.batch, args.seq, args.steps, args.layer = 0, 4, 6, 1, 2
    args.out_dir.mkdir(parents=True, exist_ok=False)
    record = dict(start_pos=args.start_pos, batch=4, seq=6, layer=2, rounds=args.rounds, iterations=args.iterations)
    record["sources"] = {
        str(p.relative_to(args.variants)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.variants.rglob("*.py"))
    }
    record["scripts"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("graph.py", "fixture.py", "run.py", "generate_graph.py", "factors.json")
    }
    try:
        with contextlib.ExitStack() as stack:
            cfg = fixture.initialize(args, stack)
            import pypto
            import simpler
            import torch
            import torch_npu
            import vllm
            from pypto.torch import shutdown

            original_gate = shutdown.require_supported_framework

            def gate(framework):
                if framework.__version__.split("+")[0] != "2.10.0.post4":
                    return original_gate(framework)
                assert callable(framework._C._npu_shutdown_synchronize) and callable(framework._C._npu_shutdown)

            shutdown.require_supported_framework = gate
            stack.enter_context(torch.inference_mode())
            record["runtime"] = dict(
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                vllm=vllm.__version__,
                pypto=pypto.__file__,
                simpler=simpler.__file__,
                cann=os.environ["ASCEND_HOME_PATH"],
                deterministic_level=torch_npu.npu._get_deterministic_level(),
                HCCL_DETERMINISTIC=os.environ["HCCL_DETERMINISTIC"],
            )
            attention, record["weights"] = fixture.load_attention(args, cfg)
            record["result"] = run_case(args, cfg, attention)
    except Exception:
        record["error"] = traceback.format_exc()
    (args.out_dir / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    return int("error" in record)


if __name__ == "__main__":
    raise SystemExit(main())
