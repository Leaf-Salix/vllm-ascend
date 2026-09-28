"""Actual native-interface graph replay with changing inputs at fixed addresses."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import traceback
from pathlib import Path

from . import fixture
from .bit_metrics import bit_metrics
from .cache_metrics import cache_dump
from .canonical_payload import canonicalize

cache_views = fixture.cache_views


def run(args, cfg, attention):
    import torch

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention import pto_attn
    from vllm_ascend.attention.dsa_v1 import AscendDSAImpl

    impl = attention.dsa_attn.dsa_attn.impl
    layer = attention.dsa_attn.dsa_attn.layer_name
    if not isinstance(impl, pto_attn.PyptoDSAImpl):
        raise AssertionError(f"Unexpected attention implementation: {type(impl)}")
    kernel, _ = pto_attn.kernel()
    expected_adapter = args.variant_dir / "vllm_ascend/attention/pto_attn.py"
    expected_kernel = args.variant_dir / "vllm_ascend/attention/pto_kernels/dspark/decode_csa.py"
    if Path(pto_attn.__file__).resolve() != expected_adapter.resolve():
        raise AssertionError(f"Wrong adapter import: {pto_attn.__file__}")
    if Path(kernel.__file__).resolve() != expected_kernel.resolve():
        raise AssertionError(f"Wrong kernel import: {kernel.__file__}")
    abi = list(kernel.decode_csa_attn_tp1_test.param_names)
    if any(name.startswith("debug_") for name in abi):
        raise AssertionError("Graph test requires the uninstrumented production kernel")
    impl.process_weights_after_loading(torch.bfloat16)
    impl._pto_calls = 0
    payload = fixture.build_payload(args)
    logical = canonicalize(payload, args, 128)
    metadata = fixture.build_metadata(args, payload, args.start_pos)
    hidden_a = payload["x"].clone()
    hidden_b = (hidden_a.float() * -0.375).to(torch.bfloat16)
    t = args.batch * args.seq

    def reset(path, hidden):
        payload["x"].copy_(hidden)
        for dst, src in zip(payload[path + "_roots"], payload["initial"]):
            dst.copy_(src)
        payload[path + "_out"].fill_(float("nan"))

    def invoke(path):
        if path == "native":
            return AscendDSAImpl.forward(
                impl, layer, payload["x"], payload[path], metadata, output=payload[path + "_out"]
            )
        before = impl._pto_calls
        result = impl.forward(layer, payload["x"], payload[path], metadata, output=payload[path + "_out"])
        if impl._pto_calls != before + 1:
            raise AssertionError("CSA fell back to native")
        if result is not payload[path + "_out"]:
            raise AssertionError("CSA did not return the caller's output buffer")
        return result

    checks, graphs, references = {}, {}, {}
    with set_ascend_forward_context({layer: metadata}, cfg, num_tokens=t):
        for path in ("native", "pto"):
            for label, hidden in (("A", hidden_a), ("B", hidden_b)):
                reset(path, hidden)
                invoke(path)
                torch.npu.synchronize()
                references[path, label] = (
                    payload[path + "_out"].clone(),
                    tuple(x.clone() for x in payload[path + "_roots"]),
                )
            if torch.equal(references[path, "A"][0], references[path, "B"][0]):
                raise AssertionError(f"A/B hidden stimulus did not change {path} output")
            reset(path, hidden_a)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                invoke(path)
            torch.npu.synchronize()
            graphs[path] = graph
            for step, (label, hidden) in enumerate((("A", hidden_a), ("B", hidden_b), ("A", hidden_a))):
                reset(path, hidden)
                before_replay = impl._pto_calls
                graph.replay()
                torch.npu.synchronize()
                expected_out, expected_cache = references[path, label]
                check = {
                    "output_exact": bool(torch.equal(payload[path + "_out"], expected_out)),
                    "cache_exact": all(
                        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                        for a, b in zip(payload[path + "_roots"], expected_cache)
                    ),
                    "python_forward_not_called": impl._pto_calls == before_replay,
                }
                checks[f"{path}-{step}-{label}"] = check
                if not all(check.values()):
                    raise AssertionError(f"A/B/A replay changed numerical result: {checks}")
        samples = {path: [] for path in graphs}
        calls_before_timing = impl._pto_calls
        for repetition in range(args.iterations + 5):
            for path in ("native", "pto") if repetition % 2 == 0 else ("pto", "native"):
                reset(path, hidden_a)
                torch.npu.synchronize()
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                begin.record()
                graphs[path].replay()
                end.record()
                torch.npu.synchronize()
                if repetition >= 5:
                    samples[path].append(begin.elapsed_time(end))
        if impl._pto_calls != calls_before_timing:
            raise AssertionError("Timed graph replay unexpectedly called Python forward")
        output_metrics = {}
        cache_metrics = {}
        for label in ("A", "B"):
            a, e = references["pto", label][0], references["native", label][0]
            output_metrics[label] = {**fixture.difference(a, e), "bits": bit_metrics(a, e)}
            label_payload = {path: cache_views(references[path, label][1]) for path in ("native", "pto")}
            cache_metrics[label] = {
                name: {
                    **fixture.difference(values["pto"], values["native"]),
                    "bits": bit_metrics(values["pto"], values["native"]),
                }
                for name, values in cache_dump(impl, metadata, label_payload).items()
            }
        if not bool((payload["pto_output_guard"] == -12.5).all().cpu()):
            raise AssertionError("Output guard changed")
        # Retain graphs through process shutdown, as in the established runner.
        globals()["retained_graphs"] = graphs
        return {
            "logical_input_sha256": logical,
            "batch": args.batch,
            "seq": args.seq,
            "start_pos": args.start_pos,
            "mode": "native forward graph A/B/A replay",
            "scope": "single real-weight layer; hidden changes at fixed addresses; metadata and initial history fixed",
            "replay": checks,
            "output_metrics": output_metrics,
            "cache_metrics": cache_metrics,
            "kernel_abi": abi,
            "imported_adapter": pto_attn.__file__,
            "imported_kernel": kernel.__file__,
            "event_ms": {p: {"median": statistics.median(s), "samples": s} for p, s in samples.items()},
            "guard_unchanged": True,
            "pto_calls": impl._pto_calls,
        }


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "extension-dir", "variant-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=6)
    parser.add_argument("--start-pos", type=int, default=8191)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    args.steps = 1
    if args.batch < 1 or args.batch > 64 or args.seq not in range(1, 7):
        parser.error("Require 1 <= batch <= 64 and 1 <= seq <= 6")
    if args.start_pos < 8 or args.layer < 0:
        parser.error("Require start-pos >= 8 and layer >= 0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "result.json").exists():
        parser.error("Refusing to overwrite an existing result.json; use a fresh --out-dir")
    record = {"complete": False}
    try:
        with contextlib.ExitStack() as stack:
            cfg = fixture.initialize(args, stack)
            import pypto
            import simpler
            import torch
            import torch_npu
            import vllm

            record["runtime"] = {
                "python": os.sys.version,
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "vllm": vllm.__version__,
                "pypto": pypto.__file__,
                "simpler": simpler.__file__,
                "cann": os.environ.get("ASCEND_HOME_PATH"),
                "ptoas": os.environ.get("PTOAS_ROOT"),
            }
            record["source_sha256"] = {
                str(path.relative_to(args.variant_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((args.variant_dir / "vllm_ascend").rglob("*.py"))
            }
            stack.enter_context(torch.inference_mode())
            attention, record["weights"] = fixture.load_attention(args, cfg)
            record["result"] = run(args, cfg, attention)
            record["complete"] = True
    except Exception:
        record["error"] = traceback.format_exc()
    (args.out_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    return 0 if record["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
