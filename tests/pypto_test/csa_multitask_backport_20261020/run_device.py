# SPDX-License-Identifier: Apache-2.0
"""同根Tensor的原多task/迁移候选/正式单SPMD对照，复用既有fixture和L1导出。"""

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import statistics
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HELPERS = (
    HERE.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
    "/parallel_20261018/mgather_physical_plan_retest_20261019"
)
sys.path[:0] = [str(HELPERS), str(HELPERS.parents[1])]


def load_multitask(source, label):
    importlib.import_module("vllm_ascend.ops.pypto")
    name = f"vllm_ascend.ops.pypto._csa_backport_{label}"
    spec = importlib.util.spec_from_file_location(
        name, source / "__init__.py", submodule_search_locations=[str(source)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(name + ".native_adapter")


def save(output, report):
    temporary = output / "result.tmp.json"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output / "result.json")
    print(json.dumps({"stage": report["stage"], "passed": report.get("passed", False)}, ensure_ascii=False), flush=True)


def profile(entries, reset, verify, output, configs, windows):
    import pypto.torch
    import torch
    from l1_compare import export_window

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graphs, checks, records = {}, {}, []
    replay_counts = {side: 0 for side in entries}

    def order(index):
        names = list(entries)
        names = names[index % len(names) :] + names[: index % len(names)]
        return names[::-1] if index % 2 else names

    with torch.npu.stream(stream):
        for side, entry in entries.items():
            reset()
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                entry()
            graphs[side] = graph
            reset()
            graph.replay()
            stream.synchronize()
            replay_counts[side] += 1
            checks[side] = verify(side)
        for index in range(3 + windows):
            for side in order(index):
                reset()
                stream.synchronize()
                if index >= 3:
                    pypto.torch.begin_dfx()
                try:
                    graphs[side].replay()
                finally:
                    if index >= 3:
                        pypto.torch.end_dfx()
                stream.synchronize()
                replay_counts[side] += 1
                if index >= 3:
                    window = len(records)
                    source = output / "dfx"
                    if window:
                        source = source / f"window_{window}"
                    records.append(
                        export_window(
                            source,
                            output / f"l1_{index - 3:02d}_{side}",
                            side,
                            kernel_config=configs[side],
                        )
                    )
                if index == 2 + windows:
                    checks[side + "_after_l1"] = verify(side)
    samples = {side: [r["incore_span_us"] for r in records if r["side"] == side] for side in entries}
    stats = {
        side: dict(
            samples_us=values,
            min_us=min(values),
            mean_us=statistics.mean(values),
            max_us=max(values),
            median_us=statistics.median(values),
        )
        for side, values in samples.items()
    }
    original, candidate, spmd = (stats[s]["mean_us"] for s in entries)
    return {
        "scope": "完整CSA最早incore开始至最晚结束，包含内部等待与调度间隙",
        "statistics_us": stats,
        "windows": records,
        "graph_checks": checks,
        "replay_counts": replay_counts,
        "order": [order(i + 3) for i in range(windows)],
        "candidate_reduction_vs_original_pct": 100 * (original - candidate) / original,
        "candidate_reduction_vs_spmd_pct": 100 * (spmd - candidate) / spmd,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="score_split")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--attempt", default="v1")
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    if not args.candidate.isidentifier() or not args.attempt.isidentifier():
        raise ValueError("候选及attempt必须是标识符")
    output = HERE / "runs" / args.candidate / f"h{args.history}_b{args.batch}_{args.attempt}"
    if args.parse_only or args.compile_only:
        output = HERE / "runs" / args.candidate / f"cpu_{args.attempt}"
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        PYPTO_PROG_BUILD_DIR=str(output / "build_output"),
        PYPTO_CACHE_DIR=str(output / "cache"),
        ASCEND_PROCESS_LOG_PATH=str(output / "ascend"),
    )
    report = dict(
        stage="initialize",
        passed=False,
        batch=args.batch,
        history=args.history,
        candidate=args.candidate,
        query_tokens=6,
        provenance=json.loads((HERE / "PROVENANCE.json").read_text()),
    )
    try:
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import pypto
        import pypto.torch
        import torch
        import torch_npu
        from compile_diagnostics import compilation_probe
        from compiler_context import record_invocations
        from pypto import passes
        from pypto.runtime import CompileOptions
        from signal_checks import signal_padding_ok, signal_view
        from source_loader import load_binding
        from wave_signal_checks import WaveSignalVerifier

        original = load_multitask(HERE / "sources/baseline", "original")
        candidate_adapter = load_multitask(HERE / "sources" / args.candidate, "candidate")
        donor = load_binding(HERE / "sources/spmd", "backport_reference")
        report["environment"] = {
            "cann": os.environ.get("ASCEND_HOME_PATH"),
            "pypto": pypto.__file__,
            "ptoas": os.environ["PTOAS_ROOT"],
            "simpler": os.environ["HCA_SIMPLER_ROOT"],
            "atomic_add": os.environ["VLLM_ASCEND_PTO_CSA_ATOMIC_ADD"],
        }
        if args.parse_only or args.compile_only:
            candidate_adapter.decode_csa_tp1_layer_test._get_dep_graph()
            if args.compile_only:
                artifact = candidate_adapter.decode_csa_tp1_layer_test._resolve_kernel_artifact(
                    (),
                    {
                        "config": CompileOptions(
                            platform="a2a3",
                            output_dir=str(output / "kernel"),
                            memory_planner=passes.MemoryPlanner.PYPTO,
                        )
                    },
                    allow_signature_mode=True,
                )
                report["compiled_artifact"] = str(artifact.output_dir)
            report.update(stage="dependency_graph_parsed", passed=True, device_execution=False)
            if args.compile_only:
                report["stage"] = "kernel_compiled_on_cpu"
            return
        if "TASK_DEVICE" not in os.environ:
            raise RuntimeError("必须经task-submit --device auto执行")
        device_id = int(os.environ["TASK_DEVICE"])
        if args.device is not None and args.device != device_id:
            raise ValueError("队列设备不一致")
        if args.windows < 3:
            raise ValueError("至少三个轮换窗口")
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
        from dsv4_csa_validation import compare_tensor
        from offline_pd.run import decode_additional_config
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

        current_platform.pre_register_and_update()
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("Native fixture需要已有自定义算子")
        tokens = args.batch * 6
        additional = decode_additional_config(
            SimpleNamespace(graph_mode="full_decode_only", capture_sizes=[tokens], recompute_scheduler=False)
        )
        additional.update(
            weight_nz_mode=2,
            enable_kv_nz=False,
            enable_dsa_cp=False,
            pto_csa_ring_config={"heap_mb": [256, 128, 256, 32], "task_window": 4096},
        )
        config = EngineArgs(
            model="/data/model/DeepSeek-V4-Flash-0731-w8a8",
            tokenizer_mode="deepseek_v4",
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="ascend",
            tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128},
            max_model_len=max(16384, args.history + 128),
            max_num_seqs=40,
            max_num_batched_tokens=256,
            block_size=32,
            enable_prefix_caching=False,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [tokens],
                "cache_dir": str(output / "compile_cache"),
            },
            additional_config=additional,
        ).create_engine_config()
        record_invocations(output / "COMPILER_INVOCATIONS.jsonl")
        with native_session(config, device_id), torch.inference_mode():
            device = torch.device(f"npu:{device_id}")
            layer, report["weights"] = make_layer(config, Path(config.model_config.model), device, 4)
            fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 1024, device)
            idx = fixture["groups"]["indexer"]
            scale = idx["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            idx["initial"] = idx["allocation"].cpu()
            pypto.torch.init(
                device=device_id,
                platform="a2a3",
                runtime="tensormap_and_ringbuffer",
                enable_chip_swimlane=1,
                enable_dep_gen=True,
                output_dir=str(output / "dfx"),
                **ring_sizing_kwargs(additional),
            )
            groups = {name: (fixture["metadata"][g["prefix"]], g["views"]) for name, g in fixture["groups"].items()}
            weights = original.prepare_weights(layer.self_attn, groups["indexer"][0].hadamard, layer)
            report["nz_formats"] = {
                n: int(torch_npu.get_npu_format(weights[n])) for n in ("wq_a", "wq_b", "wo_a", "wo_b")
            }
            if set(report["nz_formats"].values()) != {29}:
                raise AssertionError("根投影权重不是NZ29")
            out = torch.empty_like(fixture["hidden"])
            call = original.NativeCSACall(
                original.CSAOperators.register(),
                weights,
                fixture["hidden"],
                fixture["positions"],
                groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                compact_metadata=fixture["compact"],
                buffers={
                    "idx_topk_scores": torch.empty((tokens, 512), device=device),
                    "idx_topk": torch.empty((tokens, 512), dtype=torch.int32, device=device),
                    "x_out": out,
                },
            )
            allocations = {}

            def allocate(name, shape, dtype, device):
                count = 1
                for size in shape:
                    count *= size
                guard = 512 // torch.empty((), dtype=dtype).element_size()
                storage = torch.full((count + 2 * guard,), 37, dtype=dtype, device=device)
                allocations[name] = (storage, guard)
                return storage[guard:-guard].view(shape)

            spmd = donor.prepared(call.args, max_query_tokens=6, max_seq_len=args.history + 6, allocator=allocate)
            protocol = WaveSignalVerifier(spmd)
            options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
            entries = {
                "multitask_original": lambda: original.decode_csa_tp1_layer_test(*call.core_args, config=options),
                "multitask_candidate": lambda: candidate_adapter.decode_csa_tp1_layer_test(
                    *call.core_args, config=options
                ),
                "single_spmd": lambda: spmd(config=options),
            }
            input_a = fixture["hidden"].clone()
            reference, configs = {}, {}
            spmd_calls = 0

            def reset(hidden=input_a):
                restore(fixture)
                fixture["hidden"].copy_(hidden)
                out.fill_(float("nan"))
                call.args["idx_topk"].fill_(-777)
                call.args["idx_topk_scores"].fill_(float("nan"))

            def state():
                result = collect_state(fixture, out, call.args["idx_topk"])
                result["idx_topk_scores"] = call.args["idx_topk_scores"].cpu()
                return result

            def verify(side, label="A"):
                current = state()
                compared = {n: compare_tensor(v, reference[label][n], 0, 0) for n, v in current.items()}
                guards = guard_checks(fixture)
                checked = dict(against_original=compared, guards=guards)
                checked["passed"] = all(c["status"] == "PASS" for c in [*compared.values(), *guards.values()])
                if not checked["passed"]:
                    report.setdefault("failures", {})[side + label] = checked
                    save(output, report)
                    raise AssertionError(f"{side}/{label}数值、状态或保护检查失败")
                return checked

            for label, hidden in (("A", input_a), ("B", -input_a)):
                for side, entry in entries.items():
                    reset(hidden)
                    before = set((output / "build_output").glob("*/kernel_config.py"))
                    report["stage"] = f"compile_call_{side}_{label}"
                    save(output, report)
                    probe = (
                        compilation_probe(output / "spmd_codegen", wait_backoff=True, phase_clock=False)
                        if side == "single_spmd" and label == "A"
                        else contextlib.nullcontext()
                    )
                    with probe:
                        entry()
                    torch.npu.synchronize()
                    if label == "A":
                        new = set((output / "build_output").glob("*/kernel_config.py")) - before
                        if len(new) != 1:
                            raise AssertionError(f"{side}新编译入口不唯一：{new}")
                        configs[side] = next(iter(new))
                    if side == "multitask_original":
                        reference[label] = state()
                    report.setdefault("eager", {})[side + "_" + label] = verify(side, label)
                    if side == "single_spmd":
                        spmd_calls += 1
                        signals = signal_view(spmd)
                        if not (
                            bool((signals[0, :, 0] == spmd_calls).all())
                            and signal_padding_ok(signals)
                            and protocol.check(signals, spmd_calls)["passed"]
                        ):
                            raise AssertionError("单SPMD参考协议检查失败")
            report["stage"] = "paired_l1"
            save(output, report)
            report["l1"] = profile(entries, reset, verify, output, configs, args.windows)
            signals = signal_view(spmd)
            expected = spmd_calls + report["l1"]["replay_counts"]["single_spmd"]
            report["spmd_protocol"] = bool(
                (signals[0, :, 0] == expected).all()
                and signal_padding_ok(signals)
                and protocol.check(signals, expected)["passed"]
            )
            report["scratch_guards"] = {
                n: bool((base[:g].cpu() == 37).all() and (base[-g:].cpu() == 37).all())
                for n, (base, g) in allocations.items()
            }
            if not report["spmd_protocol"] or not all(report["scratch_guards"].values()):
                raise AssertionError("计时后参考协议或scratch保护区失败")
            report.update(
                stage="complete",
                passed=True,
                device_execution=True,
                scope="单卡S6合成历史；不代表整模型token/DSpark或全部动态边界",
            )
    except Exception as exc:
        # verifier可能为每项诊断重复打印整个IR；只保留首因和末尾位置，避免GB级重复日志。
        message = str(exc)
        excerpt = message if len(message) <= 18000 else message[:14000] + "\n...[重复IR省略]...\n" + message[-4000:]
        stack = "".join(traceback.format_list(traceback.extract_tb(exc.__traceback__)))
        report.update(
            passed=False, error=stack + type(exc).__name__ + ": " + excerpt, exception_characters=len(message)
        )
        raise RuntimeError(excerpt) from None
    finally:
        save(output, report)


if __name__ == "__main__":
    main()
