# SPDX-License-Identifier: Apache-2.0
"""Observe actual private CSA prefix handoffs; diagnostic runs are not timing runs."""

import argparse
import json
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pypto
import torch
from dsv4_csa_native_case import native_session
from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
from tmr_csa_full_128k_b16_compile import compile_variant
from tmr_csa_full_128k_b16_demo import exact, load_artifact
from tmr_csa_prefix_coverage_instrument import (
    MAX_RAW_CORES,
    RECORDS_PER_CORE,
    WORDS_PER_RECORD,
    analyze_trace,
    instrument_wrappers,
)


def run(args, report, *, reset_extra_trace=None):
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
    print("STAGE: precompiled baseline and process-local diagnostic build", flush=True)
    arena = torch.full((MAX_RAW_CORES, RECORDS_PER_CORE, WORDS_PER_RECORD), -1, dtype=torch.int64, device=device)
    functions = []
    ordinary = load_artifact("baseline", args.compiled)
    with instrument_wrappers(arena.data_ptr(), functions):
        candidate = compile_variant("mixed", args.output / "instrumented")
    report["functions"] = functions
    artifacts = [ordinary, candidate]
    with native_session(config, args.device), torch.inference_mode():
        print("STAGE: layer 2 formal weights and Native fixture", flush=True)
        layer, weights_description = make_layer(config, args.checkpoint, device, layer_index=2)
        fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 1024, device)
        groups = {name: (fixture["metadata"][g["prefix"]], g["views"]) for name, g in fixture["groups"].items()}
        hadamard = groups["indexer"][0].hadamard
        weights = native_adapter.prepare_weights(layer.self_attn, hadamard, layer)
        # Public registration supplies the exact external schema; execution below
        # uses the ordinary artifact and this process-local diagnostic artifact.
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

            report["samples"] = []
            for repeat in range(2):
                if repeat:
                    fixture["hidden"].neg_()
                reset()
                enqueue(registrations[0], call.core_args)
                torch_npu.npu.synchronize()
                reference = state()
                reset()
                arena.fill_(-1)
                if reset_extra_trace is not None:
                    reset_extra_trace()
                enqueue(registrations[1], call.core_args)
                torch_npu.npu.synchronize()
                equality = exact(state(), reference)
                guards = guard_checks(fixture)
                assert all(v["status"] == "PASS" for v in guards.values()), guards
                trace = arena.cpu()
                # Save before assertions, so a failed timing invariant is inspectable.
                torch.save(trace, args.output / f"trace_round{repeat}.pt")
                sample = analyze_trace(trace, functions)
                sample.update(repeat=repeat, exact=equality, guards={k: v["status"] for k, v in guards.items()})
                report["samples"].append(sample)
                print("CSA coverage", repeat, json.dumps(sample["totals"]), flush=True)
            report["status"] = "FUNCTIONAL_AND_COVERAGE_COMPLETE"
        finally:
            runtime.drain()
            runtime.close()
            restore(fixture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    args.batch, args.history = 16, 131072
    args.checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
    args.native_extension = (
        Path(os.environ["PTO_EAGER_ROOT"])
        / "vllm-ascend-dsv4-pto-0251rc1"
        / ".cache/csa/native-install/vllm_ascend_C.cpython-310-aarch64-linux-gnu.so"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "RUNNING",
        "device": args.device,
        "batch": args.batch,
        "history": args.history,
        "scope": "single layer 2 HC_pre+norm+CSA+HC_post",
        "runtime": "TMR",
        "timing": "disabled; diagnostic inserts stores and clocks",
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
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
