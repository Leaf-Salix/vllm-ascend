# SPDX-License-Identifier: Apache-2.0
"""Compare TMR Scalar-overlap boundaries on the frozen, dynamic-shape CSA."""

import argparse
import importlib
import json
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import pypto
import torch
from dsv4_csa_native_case import native_session
from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
from pypto.ir._kernel_compile import kernel_abi_for_program
from pypto.runtime._kernel_artifact import KernelArtifact
from tmr_csa_full_128k_b16_compile import BATCH, HISTORY, PACKAGES, TRACE_PACKAGE, shape_only_args


def load_artifact(name, root, batch=BATCH, history=HISTORY):
    package = TRACE_PACKAGE if name == "trace" else PACKAGES[name]
    module = importlib.import_module(f"vllm_ascend.ops.pypto.{package}.decode_csa")
    arguments, _ = shape_only_args(module._decode_csa_tp1_layer, batch, history)
    program = module.decode_csa_tp1_layer_test.specialize(*arguments)
    abi = kernel_abi_for_program(program, platform="a2a3", runtime="tensormap_and_ringbuffer")
    artifact = KernelArtifact(root / name / "build", abi)
    artifact.load()
    return artifact


def exact(actual, expected):
    assert actual.keys() == expected.keys()
    result = {}
    for name, value in actual.items():
        assert torch.equal(value, expected[name]), f"full CSA mismatch: {name}"
        result[name] = "exact"
    return result


def run(args, report):
    # Use only the private env already sourced by task-submit. Do not call
    # dsv4_csa_env.activate(), which deliberately chooses the shared SDK.
    assert os.environ.get("TASK_DEVICE") == str(args.device), "must use queue-assigned NPU"
    import torch_npu  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import init, register  # noqa: PLC0415
    from pypto.torch.launch import enqueue  # noqa: PLC0415
    from vllm.engine.arg_utils import EngineArgs  # noqa: PLC0415
    from vllm.platforms import current_platform  # noqa: PLC0415

    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_scalar_overlap_demo import native_adapter  # noqa: PLC0415
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_scalar_overlap_demo.decode_csa import (  # noqa: PLC0415
        decode_csa_tp1_layer_test,
    )
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_scalar_overlap_demo.reduction import ATOMIC_ADD  # noqa: PLC0415

    assert ATOMIC_ADD == 0, "the bit-exact comparison requires the existing non-atomic path"
    torch.set_num_threads(2)
    current_platform.pre_register_and_update()
    torch_npu.npu.set_device(args.device)
    torch_npu.npu.config.allow_internal_format = True
    torch_npu.npu.set_deterministic_level(0)
    # The frozen Python checkout has no installed Native extension. Load the
    # release's existing metadata/operator binding read-only, as a fixture
    # dependency; never redirect PyPTO or Simpler to their shared installs.
    extension_path = args.native_extension.resolve(strict=True)
    spec = spec_from_file_location("vllm_ascend.vllm_ascend_C", extension_path)
    assert spec is not None and spec.loader is not None
    extension = module_from_spec(spec)
    sys.modules[spec.name] = extension
    spec.loader.exec_module(extension)
    from vllm_ascend.utils import enable_custom_op  # noqa: PLC0415

    assert enable_custom_op(), "Native fixture custom operators must be registered"
    report["native_fixture_extension"] = str(extension_path)
    import _task_interface  # noqa: PLC0415
    import simpler  # noqa: PLC0415

    report["sdk"] = {
        "pypto_source": pypto.__file__,
        "simpler_source": simpler.__file__,
        "simpler_binding": _task_interface.__file__,
        "simpler_revision": _task_interface.__build_commit__,
        "ptoas_root": os.environ.get("PTOAS_ROOT"),
        "cann_root": os.environ.get("ASCEND_HOME_PATH"),
    }
    config = EngineArgs(
        model=str(args.checkpoint),
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        quantization="ascend",
        hf_overrides={"sliding_window": 128},
        max_model_len=max(16384, args.history + 128),
        max_num_seqs=40,
        max_num_batched_tokens=256,
        enable_prefix_caching=False,
        enforce_eager=True,
        block_size=32,
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
    ).create_engine_config()
    device = torch.device(f"npu:{args.device}")
    print("STAGE: load precompiled private baseline/candidate", flush=True)
    names = args.variants or tuple(PACKAGES)
    artifacts = [load_artifact(name, args.compiled, args.batch, args.history) for name in names]
    if args.trace_compiled:
        artifacts.append(load_artifact("trace", args.trace_compiled, args.batch, args.history))
    with native_session(config, args.device), torch.inference_mode():
        print("STAGE: layer 2 formal weights and Native fixture", flush=True)
        layer, weights_description = make_layer(config, args.checkpoint, device, layer_index=2)
        fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 1024, device)
        groups = {name: (fixture["metadata"][g["prefix"]], g["views"]) for name, g in fixture["groups"].items()}
        hadamard = groups["indexer"][0].hadamard
        weights = native_adapter.prepare_weights(layer.self_attn, hadamard, layer)
        # Public registration supplies the exact external schema; execution below
        # uses the artifacts compiled before acquiring the NPU.
        schema_op = register(decode_csa_tp1_layer_test, "scalar_overlap_demo::csa_baseline")
        call = native_adapter.NativeCSACall(
            SimpleNamespace(attention=schema_op),
            weights,
            fixture["hidden"],
            fixture["positions"],
            groups,
            layer_name=fixture["groups"]["compressed"]["prefix"],
            compact_metadata=fixture["compact"],
        )
        report.update(
            weights=weights_description,
            cache_layout={name: g["layout"] for name, g in fixture["groups"].items()},
            input_shapes={name: list(value.shape) for name, value in call.args.items()},
            weight_formats={name: int(torch_npu.get_npu_format(value)) for name, value in weights.items()},
        )
        init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer")
        runtime = get_process_kernel_state()
        graphs = []
        try:
            registrations = [runtime.ensure_callable(a, runtime.require_config()) for a in artifacts]
            initial = {name: g["initial"].to(device) for name, g in fixture["groups"].items()}

            def reset():
                for name, group in fixture["groups"].items():
                    group["allocation"].copy_(initial[name])
                call.args["x_out"].fill_(float("nan"))
                call.args["idx_topk"].fill_(-777)
                call.args["idx_topk_scores"].fill_(float("nan"))

            def state():
                values = collect_state(fixture, call.args["x_out"], call.args["idx_topk"])
                values["idx_topk_scores"] = call.args["idx_topk_scores"].cpu()
                assert torch.isfinite(values["x_out"]).all()
                return values

            if args.trace_compiled:
                from tmr_csa_o_projection_scalar_overlap_demo import _inspect_traces  # noqa: PLC0415

                print("STAGE: full CSA trace versus ordinary path, two inputs", flush=True)
                tail = torch.full((72, 8), -1, dtype=torch.int64, device=device)
                prefix = torch.full_like(tail, -1)
                assert tail.data_ptr() % 64 == prefix.data_ptr() % 64 == 0
                report["trace_samples"] = []
                for repeat in range(2):
                    if repeat:
                        fixture["hidden"].neg_()
                    reset()
                    enqueue(registrations[0], call.core_args)
                    torch_npu.npu.synchronize()
                    reference = state()
                    reset()
                    tail.fill_(-1)
                    prefix.fill_(-1)
                    enqueue(registrations[-1], (*call.core_args, tail, prefix))
                    torch_npu.npu.synchronize()
                    equality = exact(state(), reference)
                    guards = guard_checks(fixture)
                    assert all(v["status"] == "PASS" for v in guards.values()), guards
                    tail_cpu, prefix_cpu = tail.cpu(), prefix.cpu()
                    pairs = _inspect_traces(tail_cpu, prefix_cpu)
                    report["trace_samples"].append(
                        {
                            "repeat": repeat,
                            "exact": equality,
                            "guards": guards,
                            "handoff_pairs": pairs,
                            "tail_trace": tail_cpu.tolist(),
                            "prefix_trace": prefix_cpu.tolist(),
                        }
                    )
                    print(
                        f"TRACE round {repeat}: {len(pairs)} same-core handoffs, "
                        f"{sum(p['remaining_cube_ticks'] > 5 for p in pairs)} Cube overlaps",
                        flush=True,
                    )
                report["status"] = (
                    "FULL_CSA_COMPUTE_OVERLAP_OBSERVED"
                    if any(p["remaining_cube_ticks"] > 5 for s in report["trace_samples"] for p in s["handoff_pairs"])
                    else "FULL_CSA_FUNCTIONAL_NO_COMPUTE_OVERLAP"
                )
                return

            print("STAGE: eager outputs, full cache state and guards", flush=True)
            reference = None
            report["guards"] = {}
            report["eager_exact_by_variant"] = {}
            for name, registration in zip(names, registrations):
                reset()
                enqueue(registration, call.core_args)
                torch_npu.npu.synchronize()
                current = state()
                if reference is None:
                    reference = current
                    torch.save(
                        {key: current[key] for key in ("x_out", "idx_topk", "idx_topk_scores")},
                        args.output / "reference.pt",
                    )
                else:
                    equality = exact(current, reference)
                    report["eager_exact"] = equality
                    report["eager_exact_by_variant"][name] = equality
                guards = guard_checks(fixture)
                assert all(v["status"] == "PASS" for v in guards.values()), guards
                report["guards"][name] = guards
                reset()
                graph = torch_npu.npu.NPUGraph()
                graphs.append(graph)
                with torch_npu.npu.graph(graph):
                    enqueue(registration, call.core_args)
                for _ in range(5):
                    reset()
                    graph.replay()
                torch_npu.npu.synchronize()
                exact(state(), reference)
            print(f"STAGE: 30 single-CSA replays for {len(names)} path(s), rotating order", flush=True)
            samples = []
            start = torch_npu.npu.Event(enable_timing=True)
            end = torch_npu.npu.Event(enable_timing=True)
            start.record()
            end.record()
            end.synchronize()
            for repeat in range(30):
                for offset in range(len(names)):
                    mode = (repeat + offset) % len(names)
                    reset()
                    start.record()
                    graphs[mode].replay()
                    end.record()
                    end.synchronize()
                    samples.append({"repeat": repeat, "mode": mode, "us": start.elapsed_time(end) * 1000})
            print("STAGE: changed-input graph replay", flush=True)
            fixture["hidden"].neg_()
            changed = []
            for graph in graphs:
                reset()
                graph.replay()
                torch_npu.npu.synchronize()
                changed.append(state())
            assert not torch.equal(changed[0]["x_out"], reference["x_out"]), "input change did not affect output"
            report["changed_input_exact_by_variant"] = {}
            for name, current in zip(names[1:], changed[1:]):
                equality = exact(current, changed[0])
                report["changed_input_exact"] = equality
                report["changed_input_exact_by_variant"][name] = equality
            report["samples"] = samples
            report["summary"] = []
            for mode, name in enumerate(names):
                values = [s["us"] for s in samples if s["mode"] == mode]
                report["summary"].append(
                    {"path": name, "min_us": min(values), "mean_us": mean(values), "max_us": max(values)}
                )
            report["status"] = "FUNCTIONAL_AND_TIMING_COMPLETE"
        finally:
            for graph in reversed(graphs):
                graph.reset()
            runtime.drain()
            runtime.close()
            restore(fixture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(PACKAGES))
    parser.add_argument("--trace-compiled", type=Path, help="run two mechanism traces instead of timing")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/model/DeepSeek-V4-Flash-0731-w8a8"))
    parser.add_argument(
        "--native-extension",
        type=Path,
        default=Path(os.environ["PTO_EAGER_ROOT"]) / "vllm-ascend-dsv4-pto-0251rc1/.cache/csa/native-install/"
        "vllm_ascend_C.cpython-310-aarch64-linux-gnu.so",
    )
    args = parser.parse_args()
    if not 1 <= args.batch <= 40 or args.history < 0:
        parser.error("batch must be 1..40 and history must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "RUNNING",
        "variants": args.variants or list(PACKAGES),
        "batch": args.batch,
        "history": args.history,
        "device": args.device,
        "scope": "single layer 2 HC_pre+norm+CSA+HC_post; not full model",
        "runtime": "TMR",
        "timing": (
            "disabled: mechanism trace only"
            if args.trace_compiled
            else "single complete NPUGraph replay, external NPU events, resets outside interval"
        ),
        "atomic_add": False,
        "deterministic_level": 0,
        "profiler": False,
    }
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report.get("summary", report["status"]), indent=2))


if __name__ == "__main__":
    main()
