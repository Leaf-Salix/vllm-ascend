# SPDX-License-Identifier: Apache-2.0
# 复用10月12日整层PTO外部profiling用例，仅补缺失的128K/B28含AICPU跨度。
# 保留正式第4层权重、Native设备metadata、npugraph_ex整图及全部输出/state检查。
"""Pair actual Native/PTO compiled attention halves with the decode template config."""
import argparse
import contextlib
import importlib
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

from dsv4_csa_env import activate

SOURCE = activate()
import torch  # noqa: E402
import torch_npu  # noqa: E402
from dsv4_csa_native_case import native_session  # noqa: E402
from dsv4_csa_single_layer import (  # noqa: E402
    collect_state,
    guard_checks,
    make_fixture,
    make_layer,
    restore,
)
from csa_measure import layer_device_interval, measure_graph_interval  # noqa: E402
from offline_pd.run import decode_additional_config  # noqa: E402
from vllm.compilation.decorators import support_torch_compile  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402


class CompiledNativeHalf(torch.nn.Module):
    def __init__(self, *, vllm_config, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor):
        layer = self.layer
        residual = hidden_states.clone()
        mixed, post, comb = layer.hc_pre(
            hidden_states, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
        normed = layer.input_layernorm(mixed)
        attended = layer.self_attn(positions=positions, hidden_states=normed, llama_4_scaling=None)
        output.copy_(layer.hc_post(attended, residual, post, comb))
        return output


@support_torch_compile(dynamic_arg_dims={"hidden_states": 0, "positions": 0, "output": 0})
class CompiledPTOHalf(torch.nn.Module):
    def __init__(self, *, vllm_config, layer):
        super().__init__()
        self.prefix = layer.self_attn.dsa_attn.prefix

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor):
        torch.ops.vllm.dsv4_csa_forward(hidden_states, positions, output, self.prefix)
        return output


class RequirePTORuntime:
    """Fail on any Native fallback; the counter runs during capture, outside replay."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = 0

    def eligible(self, context, hidden, positions):
        if not self.delegate.eligible(context, hidden, positions):
            raise RuntimeError("PTO fixture unexpectedly falls back to Native")
        return True

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.delegate(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--schedule-only", action="store_true",
                        help="只补采派发映射；保留重放校验，不执行性能 profiler")
    parser.add_argument("--super-kernel", type=int, choices=(0, 1), required=True)
    parser.add_argument("--side", choices=("pto",), required=True)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    if "TASK_DEVICE" not in os.environ or args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("必须使用task-submit --device auto分配的设备")
    args.output.mkdir(parents=True, exist_ok=True)
    preexisting = list((Path(os.environ["ASCEND_OPP_PATH"]) / "static_kernel").glob("**/binary_info_config.json"))
    if preexisting:
        raise RuntimeError("This pair requires a fresh per-side OPP static_kernel directory")
    report = {
        "super_kernel": bool(args.super_kernel),
        "diagnostic_only": args.profile_only or args.schedule_only,
        "compile_entry": "torch.compile(backend=npugraph_ex)",
        "status": "RUNNING", "side": args.side, "batch": args.batch, "history": args.history,
        "device": args.device, "layer_index": 4, "source": str(SOURCE),
        "variant": os.environ["PTO_CSA_VARIANT"],
        "cann": os.environ["ASCEND_HOME_PATH"], "opp": os.environ["ASCEND_OPP_PATH"],
        "custom_opp": os.environ["ASCEND_CUSTOM_OPP_PATH"],
        "scope": "Single attention half, real layer4 weights, synthetic history; no MoE/EP16 acceptance",
        "compact_metadata": "PTO second-CSA reuse; Native retains its per-layer metadata calls",
    }
    original_qli = None
    try:
        current_platform.pre_register_and_update()
        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("Native custom operators unavailable")
        tokens = args.batch * 6
        additional = decode_additional_config(SimpleNamespace(
            graph_mode="full_decode_only", capture_sizes=[tokens], recompute_scheduler=False))
        additional.update(weight_nz_mode=2, enable_kv_nz=False, enable_dsa_cp=False,
                          pto_csa_ring_config={"heap_mb": [256, 128, 256, 32], "task_window": 4096})
        config = EngineArgs(
            model="/data/model/DeepSeek-V4-Flash-0731-w8a8", tokenizer_mode="deepseek_v4",
            trust_remote_code=True, dtype="bfloat16", quantization="ascend", tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128}, max_model_len=max(16384, args.history + 128),
            max_num_seqs=40, max_num_batched_tokens=256, block_size=32, enable_prefix_caching=False,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [tokens],
                                "cache_dir": str(args.output / "compile_cache")},
            additional_config=additional,
        ).create_engine_config()
        report["requested"] = additional
        report["compile_config"] = str(config.compilation_config)
        from vllm.forward_context import get_forward_context

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_csa import _native_attention_half

        with native_session(config, args.device), torch.inference_mode():
            device = torch.device(f"npu:{args.device}")
            layer, weights = make_layer(config, Path(config.model_config.model), device, 4)
            report["weights"] = weights
            report["multistream"] = {
                "dsa_overlap": layer.self_attn.dsa_attn.dsa_attn.impl.multistream_dsv4_dsa_overlap,
                "expression": "torch.npu.stream + record_event/wait_event/wait_stream",
                "boundary": "Native dsa_forward custom op; stream operations execute during graph capture",
                "graph_owner": "npugraph_ex (force_eager=False)",
                "shared_expert": "configured but outside the attention-half measurement",
            }
            if not report["multistream"]["dsa_overlap"]:
                raise RuntimeError("Native DSA multistream overlap must remain enabled")
            fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 1024, device)
            group = fixture["groups"]["indexer"]
            scale = group["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            group["initial"] = group["allocation"].cpu()
            output = torch.empty_like(fixture["hidden"])
            topk = {}
            service = runtime = observed = None
            if args.side == "pto":
                import pypto.torch
                from pypto import pypto_core, _torch_npu
                import _task_interface
                expected_root = Path(os.environ["CSA_STATIC_PYPTO_ROOT"]).resolve()
                for loaded in (pypto_core, _torch_npu):
                    if expected_root not in Path(loaded.__file__).resolve().parents:
                        raise RuntimeError("PyPTO未使用指定独立worktree")
                report["dependency_environment"] = {
                    "pypto_core": pypto_core.__file__, "adapter": _torch_npu.__file__,
                    "simpler": _task_interface.__file__, "torch": torch.__version__,
                    "torch_npu": torch_npu.__version__,
                }

                from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs, selected_variant, variant_package

                service = importlib.import_module(f"{variant_package()}.service")
                adapter = importlib.import_module(f"{variant_package()}.native_adapter")
                pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer",
                                 **ring_sizing_kwargs(additional))
                runtime = service.CSAServiceRuntime(layer.self_attn, adapter.CSAOperators.register(), 40, layer)
                observed = RequirePTORuntime(runtime)
                layer.self_attn.dsa_attn._pto_csa_runtime = observed
                report["variant_kind"] = selected_variant()
                report["implementation_package"] = service.__package__
                report["implementation_source"] = str(Path(service.__file__).resolve())
                report["native_weight_formats"] = {
                    name: int(torch_npu.get_npu_format(getattr(layer.self_attn, name).weight))
                    for name in ("wq_a", "wq_b", "wo_a", "wo_b")}
            else:
                original_qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer

                def record_qli(*inputs, **kwargs):
                    value = original_qli(*inputs, **kwargs)
                    topk["value"] = value[0]
                    return value

                torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = record_qli

            @contextlib.contextmanager
            def context():
                with set_ascend_forward_context(
                        fixture["metadata"], config, num_tokens=tokens, num_actual_tokens=tokens):
                    if runtime is not None:
                        current = get_forward_context()
                        cache = current.additional_kwargs.setdefault(service._COMPACT_METADATA_CACHE, {})
                        for name in ("compressed", "indexer"):
                            req = fixture["metadata"][runtime.prefixes[name]].decode
                            cache[id(req)] = (req, fixture["compact"][name])
                    yield

            def topk_value():
                return runtime.topk[:tokens] if runtime is not None else topk["value"]

            def eager_call():
                with context():
                    if runtime is None:
                        _native_attention_half(layer.self_attn.dsa_attn,
                                               fixture["hidden"], fixture["positions"], output)
                    else:
                        torch.ops.vllm.dsv4_csa_forward(
                            fixture["hidden"], fixture["positions"], output, layer.self_attn.dsa_attn.prefix)

            restore(fixture)
            eager_call()
            torch.npu.synchronize()
            reference = collect_state(fixture, output, topk_value())
            report["initial_guards"] = guard_checks(fixture)
            from npugraph_ex._acl_concrete_graph import static_kernel
            from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph

            graph_instances = {}
            original_graph_run = AclGraph.run

            def record_graph_run(owner, key, *inputs, **kwargs):
                graph_instances[(id(owner), key)] = owner
                return original_graph_run(owner, key, *inputs, **kwargs)

            AclGraph.run = record_graph_run
            static_results = []
            static_super_flags = []
            report["super_kernel_graph_calls"] = []
            original_super = torch.npu.NPUGraph.super_kernel_optimize

            def observe_super(graph, *inputs, **kwargs):
                record = {"status": "STARTED"}
                report["super_kernel_graph_calls"].append(record)
                result = original_super(graph, *inputs, **kwargs)
                record["status"] = "RETURNED_SUCCESSFULLY"
                return result

            torch.npu.NPUGraph.super_kernel_optimize = observe_super
            original_compile = static_kernel.static_compile

            def observe_static(*inputs, **kwargs):
                import inspect

                bound = inspect.signature(original_compile).bind_partial(*inputs, **kwargs)
                flag = bool(kwargs.get("super_kernel_optimize",
                                       bound.arguments.get("super_kernel_optimize", False)))
                static_super_flags.append(flag)
                if flag != bool(args.super_kernel):
                    raise RuntimeError("Static compiler did not receive the requested superkernel flag")
                success = original_compile(*inputs, **kwargs)
                static_results.append(success)
                return success

            static_kernel.static_compile = observe_static
            cls = CompiledNativeHalf if runtime is None else CompiledPTOHalf
            module = cls(vllm_config=config, layer=layer)
            options = {
                "force_eager": False, "inplace_pass": True,
                "clone_input": False, "clone_output": False,
                "static_kernel_compile": True,
                "super_kernel_optimize": bool(args.super_kernel),
            }
            report["backend_options"] = options
            report["torch_compile_options"] = {"dynamic": False, "fullgraph": True}
            report["compilation_scope"] = {
                "vllm_ascend_fx_passes": False,
                "note": "Direct npugraph_ex passes; EngineArgs alone does not apply vLLM FX passes",
            }
            # The named backend is mandatory; no vLLM compilation wrapper or fallback.
            torch.npu.set_compile_mode(jit_compile=False)
            compiled = torch.compile(module, backend="npugraph_ex", fullgraph=True,
                                     dynamic=False, options=options)

            def compiled_call():
                with context():
                    compiled(fixture["hidden"], fixture["positions"], output)

            restore(fixture)
            compiled_call()
            torch.npu.synchronize()
            report["compiler"] = {
                "wrapper_compiled": bool(static_results),
                "entry": "torch.compile(backend=npugraph_ex)",
                "static_super_flags": static_super_flags,
                "static_compile_results": static_results,
                "installed_static_packages": len(static_kernel._installed_run_pkgs),
                "pto_dispatch_calls": observed.calls if observed is not None else 0,
            }
            if (runtime is None and not report["compiler"]["wrapper_compiled"]) or any(not result for result in static_results):
                raise RuntimeError(f"Compilation failed: {report['compiler']}")
            if runtime is None and (not static_results or not static_kernel._installed_run_pkgs):
                raise RuntimeError("Native requires actual static compilation and installation")
            if observed is not None and observed.calls == 0:
                raise RuntimeError("No PTO service invocation was observed")
            if len(graph_instances) != 1:
                raise RuntimeError("Expected one complete npugraph_ex graph for isolated device timing")
            (_, graph_key), owner = next(iter(graph_instances.items()))
            if owner._updated_node_infos:
                raise RuntimeError("Host-updated graph nodes require the backend replay wrapper")
            graph = owner.graph[graph_key]
            # Addresses/shapes are fixed, clone_input/output=False. Retain the graph owner,
            # tensor fixture and module. Check raw replay against this compiled-call result.
            # The first compiled invocation includes backend warmups that mutate cache/state.
            restore(fixture)
            compiled_call()
            torch.npu.synchronize()
            if len(graph_instances) != 1:
                raise RuntimeError("Compiled-call reference unexpectedly created another graph")
            reference = collect_state(fixture, output, topk_value())
            report["measurement"] = {
                "capture_owner": "npugraph_ex",
                "entry": "torch.compile(backend=npugraph_ex)",
                "timed_call": "backend-created NPUGraph.replay",
                "graph_count": len(graph_instances), "host_updated_nodes": 0,
                "reference": "Same captured graph via the compiled callable, identical initial state",
                "scope": "Fixed-address single-CSA device replay; not changing-input model execution",
            }
            report["timing"] = measure_graph_interval(
                fixture, graph.replay, output, topk_value, reference,
                iters=1 if args.profile_only or args.schedule_only else 20,
                warmup=1 if args.profile_only or args.schedule_only else 5,
                require_exact=runtime is not None,
                profile_dir=None if args.schedule_only else args.output / "profile")
            calls = report["super_kernel_graph_calls"]
            if bool(calls) != bool(args.super_kernel):
                raise RuntimeError("npugraph_ex did not apply the requested graph optimization")
            if any(call["status"] != "RETURNED_SUCCESSFULLY" for call in calls):
                raise RuntimeError("Superkernel graph optimization did not finish")
            if any(value["status"] != "PASS"
                   for value in report["timing"]["eager_comparison"].values()):
                raise RuntimeError("Direct replay differs from the same graph through the compiled callable")
            if args.save_state:
                torch.save(collect_state(fixture, output, topk_value()), args.output / "states.pt")
            report["mean_us"] = (None if args.profile_only or args.schedule_only else
                                 statistics.mean(report["timing"]["samples_us"]))
            if args.schedule_only:
                if report["timing"].get("diagnostic_dfx", {}).get("windows") != 1:
                    raise RuntimeError("派发补采必须由诊断入口生成一个 DFX 窗口")
                report["device_profile"] = {"available": False,
                                            "reason": "仅补采调度映射，不产生性能结论"}
                report["device_span_us"] = None
            else:
                # 按Stream/Model ID剔除图外reset，保留整层全部流及AICPU调度。
                report["device_profile"] = layer_device_interval(args.output / "profile")
                (args.output / "device_profile_layer_only.json").write_text(
                    json.dumps(report["device_profile"], ensure_ascii=False, indent=2) + "\n")
                if not report["device_profile"]["available"]:
                    raise RuntimeError("缺少 profiler 设备跨度，不能以 Host Event 代替")
                report["device_span_us"] = report["device_profile"]["span_us"]
            report["operator_source"] = os.environ["HC_PRE_TEST_OPERATOR_SOURCE"]
            report["status"] = ("SCHEDULE_CAPTURED" if args.schedule_only else
                                "PROFILED" if args.profile_only else "MEASURED")
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        if original_qli is not None:
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        fields = ("status", "mean_us", "compiler", "error")
        print(json.dumps({k: report[k] for k in fields if k in report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
