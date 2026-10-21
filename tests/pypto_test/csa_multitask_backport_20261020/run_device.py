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


def functional_replay(entries, reset, verify, input_a):
    """边界用例只比较原多task与候选，不重编无关SPMD参考或重复采性能。"""
    import torch

    checked = {}
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for side, entry in entries.items():
            if side == "multitask_original":
                continue
            reset(input_a)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                entry()
            stream.synchronize()
            for number, (label, hidden) in enumerate((("A", input_a), ("B", -input_a), ("A", input_a))):
                reset(hidden)
                graph.replay()
                stream.synchronize()
                checked[f"{side}_{number}_{label}"] = verify(side, label)
    return checked


def profile(entries, reset, verify, output, configs, windows, measure_outer=False):
    import pypto.torch
    import torch
    from l1_compare import export_window

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graphs, checks, records, outer_samples = {}, {}, [], {side: [] for side in entries}
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
        if measure_outer:
            for index in range(windows):
                for side in order(index):
                    reset()
                    stream.synchronize()
                    begin = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    begin.record(stream)
                    graphs[side].replay()
                    end.record(stream)
                    stream.synchronize()
                    replay_counts[side] += 1
                    outer_samples[side].append(begin.elapsed_time(end) * 1000)
                    if index == windows - 1:
                        checks[side + "_after_outer"] = verify(side)
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
    original = stats["multitask_original"]["mean_us"]
    candidate = stats["multitask_candidate"]["mean_us"]
    spmd = stats["single_spmd"]["mean_us"]
    result = {
        "scope": "完整CSA最早incore开始至最晚结束，包含内部等待与调度间隙",
        "statistics_us": stats,
        "windows": records,
        "graph_checks": checks,
        "replay_counts": replay_counts,
        "order": [order(i + 3) for i in range(windows)],
        "candidate_reduction_vs_original_pct": 100 * (original - candidate) / original,
        "candidate_reduction_vs_spmd_pct": 100 * (spmd - candidate) / spmd,
    }
    if "multitask_current" in stats:
        current = stats["multitask_current"]["mean_us"]
        result["candidate_reduction_vs_current_pct"] = 100 * (current - candidate) / current
    if measure_outer:
        result["outer"] = {
            "scope": "DFX窗口外NPU event包围单次graph.replay；含AICPU/运行时，不与L1相减认领纯AICPU",
            "samples_per_side": windows,
            "statistics_us": {
                side: dict(samples_us=values, min_us=min(values), mean_us=statistics.mean(values), max_us=max(values))
                for side, values in outer_samples.items()
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="score_split")
    parser.add_argument("--current", help="可选的已保留私有包；与原基线、候选和单SPMD同轮比较")
    parser.add_argument(
        "--spmd-reference",
        choices=("spmd", "spmd_7061c112", "spmd_a241fa61", "spmd_a241fa61_half_shape_fix"),
        default="spmd",
        help="固定13259f92/100tick、7061c112/150tick或a241fa61/B1驻留UB；不混合汇总",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--attempt", default="v1")
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--compile-spmd-reference", action="store_true", help="CPU核对所选参考的策略类型与代码生成")
    parser.add_argument("--dump-final-ir", action="store_true", help="仅CPU失败诊断，保存一份最终IR")
    parser.add_argument("--compile-legacy-qr", action="store_true", help="仅CPU编译HCA使用的旧QR入口")
    parser.add_argument("--compile-layout-probe", action="store_true")
    parser.add_argument("--layout-probe", action="store_true", help="新页证明helper需测；相同helper已通过时不重复")
    parser.add_argument("--boundary-replay", action="store_true", help="单图metadata变化及Native补位，独立于性能")
    parser.add_argument(
        "--boundary-context-history",
        type=int,
        help="边界重放中另切换到此history，再恢复原值；不改冻结源码清单",
    )
    parser.add_argument("--measure-outer", action="store_true", help="另采NPU event外层耗时，保持L1口径独立")
    parser.add_argument(
        "--dep-gen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认保留历史依赖图采集；--no-dep-gen仅用于单独核对采集开销，仍保留L1时间戳",
    )
    parser.add_argument("--functional-only", action="store_true", help="只做原多task/候选精确与A/B/A图检查")
    parser.add_argument("--constant-indexer-history", action="store_true", help="仅功能模式：历史key为零制造严格同分")
    parser.add_argument("--capture-level4", action="store_true", help="功能模式后补采一个L4诊断窗口，不进入性能表")
    parser.add_argument(
        "--candidate-wait-backoff",
        choices=("off", "cached100"),
        help="默认继承候选清单的来源等待策略；显式off仅用于默认TWAIT消融",
    )
    parser.add_argument("--candidate-wait-units", choices=("score", "output", "hc", "attention"))
    parser.add_argument(
        "--indexer-page-pattern", choices=("original", "ascending_history", "fragmented_history"), default="original"
    )
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    if args.compile_spmd_reference and not args.compile_only:
        raise ValueError("参考CPU编译必须同时使用--compile-only")
    if not args.candidate.isidentifier():
        raise ValueError("候选必须是标识符")
    from candidate_wait_probe import resolve_wait_policy

    candidate_manifest = HERE / "sources" / args.candidate / "candidate.json"
    manifest = json.loads(candidate_manifest.read_text()) if candidate_manifest.exists() else {}
    args.candidate_wait_backoff, args.candidate_wait_units = resolve_wait_policy(
        manifest, args.candidate_wait_backoff, args.candidate_wait_units
    )
    if args.compile_legacy_qr and not args.compile_only:
        raise ValueError("旧QR入口探针必须使用--compile-only")
    if args.dump_final_ir and not args.compile_only:
        raise ValueError("最终IR诊断仅支持--compile-only")
    if args.candidate_wait_backoff == "off" and args.candidate_wait_units != "score":
        raise ValueError("指定等待单元须同时启用--candidate-wait-backoff cached100")
    if args.functional_only and args.measure_outer:
        raise ValueError("功能用例不采外层或L1性能")
    if args.constant_indexer_history and not args.functional_only:
        raise ValueError("同分边界只用于功能模式，不混入主性能数据")
    if args.capture_level4 and (not args.functional_only or args.current or args.compile_only or args.parse_only):
        raise ValueError("L4补采仅支持单一候选的functional-only设备模式")
    if args.boundary_replay and (args.batch < 1 or args.history < 2):
        raise ValueError("metadata用例要求batch>=1且history>=2；B1只验证长度/页表变化")
    if args.boundary_context_history is not None:
        if not args.boundary_replay or not 0 < args.boundary_context_history < args.history:
            raise ValueError("--boundary-context-history要求边界重放且0<history<原history")
    if not args.candidate.isidentifier() or not args.attempt.isidentifier():
        raise ValueError("候选及attempt必须是标识符")
    if args.current and (not args.current.isidentifier() or args.current == args.candidate):
        raise ValueError("current必须是不同于candidate的私有包名")
    output = HERE / "runs" / args.candidate / f"h{args.history}_b{args.batch}_{args.attempt}"
    if args.parse_only or args.compile_only or args.compile_layout_probe:
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
        current=args.current,
        candidate_wait_backoff=args.candidate_wait_backoff,
        candidate_wait_units=args.candidate_wait_units,
        performance_measured=False,
        query_tokens=6,
        indexer_page_pattern=args.indexer_page_pattern,
        constant_indexer_history=args.constant_indexer_history,
        provenance=json.loads((HERE / "PROVENANCE.json").read_text()),
    )
    # 初始来源清单保持原样；用户后来增加的测试范围单独记录。
    report["requested_test_standard"] = json.loads((HERE / "CASE_STANDARD.json").read_text())
    report["profiling"] = {
        "chip_swimlane_level": 4 if args.capture_level4 else 1,
        "dep_gen": args.dep_gen,
    }
    spmd_source = HERE / "sources" / args.spmd_reference
    spmd_revision, spmd_wait_ticks = {
        "spmd": ("13259f92", 100),
        "spmd_7061c112": ("7061c112", 150),
        "spmd_a241fa61": ("a241fa61", 150),
        "spmd_a241fa61_half_shape_fix": ("a241fa61", 150),
    }[args.spmd_reference]
    report["spmd_reference"] = {
        "source": str(spmd_source),
        "source_commit": spmd_revision,
        "backoff_ticks": spmd_wait_ticks,
    }
    if args.spmd_reference == "spmd_a241fa61_half_shape_fix":
        report["spmd_reference"]["private_compatibility_patch"] = json.loads(
            (spmd_source / "compatibility_patch.json").read_text()
        )
    # 公共清单描述最初冻结，不能把其score_split标签当成后续候选的来源。
    report["provenance"]["initial_candidate"] = report["provenance"].pop("candidate", None)
    report["provenance"]["candidate"] = args.candidate
    report["provenance"]["initial_changed_files"] = report["provenance"].pop("changed_files", [])
    report["provenance"]["initial_status"] = report["provenance"].pop("status", None)
    if manifest:
        report["provenance"]["candidate_manifest"] = manifest
    try:
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import pypto
        import pypto.torch
        import torch
        import torch_npu
        from candidate_wait_probe import backoff_probe
        from compiler_context import record_invocations
        from pypto import passes
        from pypto.runtime import CompileOptions
        from signal_checks import signal_padding_ok, signal_view
        from source_loader import load_binding
        from toolchain_identity import check_ptoas_invocations, check_same, record
        from wave_signal_checks import WaveSignalVerifier

        original = load_multitask(HERE / "sources/baseline", "original")
        candidate_adapter = load_multitask(HERE / "sources" / args.candidate, "candidate")
        current_adapter = load_multitask(HERE / "sources" / args.current, "current") if args.current else None
        donor = load_binding(spmd_source, "backport_reference")

        def candidate_compile_scope():
            if args.candidate_wait_backoff == "off":
                return contextlib.nullcontext()
            return backoff_probe(
                output / "candidate_wait_codegen",
                spmd_source / "wait_codegen.py",
                args.candidate_wait_units,
            )

        report["environment"] = {
            "cann": os.environ.get("ASCEND_HOME_PATH"),
            "pypto": pypto.__file__,
            "ptoas": os.environ["PTOAS_ROOT"],
            "simpler": os.environ["HCA_SIMPLER_ROOT"],
            "atomic_add": os.environ["VLLM_ASCEND_PTO_CSA_ATOMIC_ADD"],
        }
        report["toolchain_identity"] = record()
        report["shared_wait_source"] = str(spmd_source / "wait_codegen.py")
        if args.compile_layout_probe:
            from layout_probe import make_probe

            helper = importlib.import_module(candidate_adapter.__package__ + ".indexer_scale_layout").prove_scale_layout
            probe = make_probe(helper)
            artifact = probe._resolve_kernel_artifact(
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
            report.update(
                stage="layout_probe_compiled",
                passed=True,
                device_execution=False,
                compiled_artifact=str(artifact.output_dir),
            )
            return
        if args.parse_only or args.compile_only:
            compile_entry = candidate_adapter.decode_csa_tp1_layer_test
            select_constants = getattr(candidate_adapter, "csa_compile_constants", None)
            compile_keywords = {} if select_constants is None else select_constants(args.batch * 6)
            if select_constants is not None and not (args.compile_spmd_reference or args.compile_legacy_qr):
                report["candidate_controls"] = compile_keywords
                # 注册只解析前端IR，不初始化NPU；检查实际入口选择及TMR/HBG运行时ABI。
                cpu_ops = candidate_adapter.CSAOperators.register(namespace="csa_shape_gate_cpu")
                selected_op = cpu_ops.for_tokens(args.batch * 6)
                default_names = [p.name for p in cpu_ops.attention._schema.arguments]
                selected_names = [p.name for p in selected_op._schema.arguments]
                hbg_op = candidate_adapter.HBGCSAOperators.register().attention
                hbg_names = [p.name for p in hbg_op._schema.arguments]
                if selected_names != default_names or hbg_names != [*default_names, "host_max_seq_len"]:
                    raise AssertionError("入口分档不应改变TMR Tensor ABI或HBG Host scalar ABI")
                report["registered_entry_abi"] = {
                    "selected": str(selected_op),
                    "tmr_parameters": default_names,
                    "hbg_parameters": hbg_names,
                    "device_execution": False,
                }
            if args.compile_spmd_reference:
                compile_entry = importlib.import_module(donor.name + ".entry").decode_csa_spmd
                compile_keywords = donor.select_policy(
                    tokens=args.batch * 6,
                    requests=args.batch,
                    max_query_tokens=6,
                    max_seq_len=args.history + 6,
                ).constexprs()
                report["compile_scope"] = "仅所选单SPMD参考的CPU编译，不构造设备输入或执行NPU"
                report["reference_controls"] = compile_keywords
            if args.compile_legacy_qr:
                from qr_compat_probe import make_probe

                helper = importlib.import_module(candidate_adapter.__package__ + ".qkv_proj_rope").q_proj_qr
                compile_entry = make_probe(helper)
                compile_keywords = {}
                report["compile_scope"] = "仅HCA旧七参数QR入口，未运行HCA设备计算"
            compile_entry._get_dep_graph()
            if args.compile_only:
                debug_scope = contextlib.nullcontext()
                if args.dump_final_ir:
                    from codegen_debug import final_ir_capture

                    report["final_ir"] = str(output / "final_program.py")
                    debug_scope = final_ir_capture(report["final_ir"])
                with candidate_compile_scope(), debug_scope:
                    artifact = compile_entry._resolve_kernel_artifact(
                        (),
                        {
                            **compile_keywords,
                            "config": CompileOptions(
                                platform="a2a3",
                                output_dir=str(output / "kernel"),
                                memory_planner=passes.MemoryPlanner.PYPTO,
                            ),
                        },
                        allow_signature_mode=True,
                    )
                report["compiled_artifact"] = str(artifact.output_dir)
                from pypto.runtime.kernel_compiler import KernelCompiler

                sources = list(Path(artifact.output_dir).glob("orchestration/*.cpp"))
                if len(sources) != 1:
                    raise AssertionError("CPU编译需要唯一AICPU编排入口")
                record_invocations(output / "COMPILER_INVOCATIONS.jsonl")
                binary = KernelCompiler(platform="a2a3").compile_orchestration(
                    "tensormap_and_ringbuffer", str(sources[0])
                )
                report["orchestration_shared_object_bytes"] = len(binary)
                report["incore_binary_compiled"] = False
            report.update(stage="dependency_graph_parsed", passed=True, device_execution=False)
            if args.compile_only:
                report["stage"] = "kernel_codegen_and_orchestration_compiled_on_cpu"
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
            fixture = make_fixture(
                config,
                layer.self_attn,
                args.batch,
                args.history,
                1024,
                device,
                indexer_page_pattern=args.indexer_page_pattern,
            )
            idx = fixture["groups"]["indexer"]
            scale = idx["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            if args.constant_indexer_history:
                # 与来源的同分回归一致；当前key仍由完整CSA写回，guard继续检查。
                idx["views"][0].zero_()
                scale.fill_(0.00390625)
            idx["initial"] = idx["allocation"].cpu()
            pypto.torch.init(
                device=device_id,
                platform="a2a3",
                runtime="tensormap_and_ringbuffer",
                enable_chip_swimlane=4 if args.capture_level4 else 1,
                enable_dep_gen=args.dep_gen,
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

            spmd = (
                None
                if args.functional_only
                else donor.prepared(call.args, max_query_tokens=6, max_seq_len=args.history + 6, allocator=allocate)
            )
            protocol = None if args.functional_only else WaveSignalVerifier(spmd)
            options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
            has_scale_layout = (HERE / "sources" / args.candidate / "indexer_scale_layout.py").exists()
            if has_scale_layout and args.layout_probe:
                from layout_probe import validate as validate_layout

                report["stage"] = "layout_proof_validation"
                save(output, report)
                report["layout_probe"] = validate_layout(candidate_adapter, device, options)

            def multitask_entry(adapter, role):
                select_constants = getattr(adapter, "csa_compile_constants", None)
                if select_constants is None:
                    return lambda: adapter.decode_csa_tp1_layer_test(*call.core_args, config=options)
                ops = adapter.CSAOperators.register(namespace=f"csa_backport_{role}", config=options)
                bound_call = adapter.NativeCSACall(
                    ops,
                    weights,
                    fixture["hidden"],
                    fixture["positions"],
                    groups,
                    layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                    compact_metadata=fixture["compact"],
                    buffers={name: call.args[name] for name in ("idx_topk_scores", "idx_topk", "x_out")},
                )
                if bound_call.param_names != call.param_names:
                    raise AssertionError("入口分档不应改变原CSA运行时Tensor参数顺序")
                report.setdefault("entry_dispatch", {})[role] = {
                    "route": "CSAOperators.register + NativeCSACall",
                    "constexpr": select_constants(tokens),
                    "operator": str(ops.for_tokens(tokens)),
                }
                return bound_call

            entries = {
                "multitask_original": multitask_entry(original, "multitask_original"),
                "multitask_candidate": multitask_entry(candidate_adapter, "multitask_candidate"),
                "single_spmd": lambda: spmd(config=options),
            }
            if current_adapter is not None:
                entries = {
                    "multitask_original": entries["multitask_original"],
                    "multitask_current": multitask_entry(current_adapter, "multitask_current"),
                    "multitask_candidate": entries["multitask_candidate"],
                    "single_spmd": entries["single_spmd"],
                }
            if args.functional_only:
                entries.pop("single_spmd")
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
                    if label == "A":
                        check_same(report["toolchain_identity"])
                        report.setdefault("same_toolchain_roles", []).append(side)
                    reset(hidden)
                    before = set((output / "build_output").glob("*/kernel_config.py"))
                    report["stage"] = f"compile_call_{side}_{label}"
                    save(output, report)
                    probe = (
                        backoff_probe(
                            output / "spmd_codegen",
                            spmd_source / "wait_codegen.py",
                            "single_spmd",
                            expected_ticks=spmd_wait_ticks,
                        )
                        if side == "single_spmd" and label == "A"
                        else contextlib.nullcontext()
                    )
                    if side == "multitask_candidate" and label == "A":
                        probe = candidate_compile_scope()
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
            if args.functional_only:
                report["stage"] = "functional_graph_replay"
                save(output, report)
                report["functional_graph"] = functional_replay(entries, reset, verify, input_a)
                report["spmd_protocol"] = None
                report["spmd_reference_scope"] = "未运行SPMD；功能边界只对照迁移前多task"
                if args.capture_level4:
                    from capture_level4 import capture

                    report["stage"] = "capture_level4"
                    save(output, report)
                    report["diagnostic_swimlane"] = capture(
                        entry=entries["multitask_candidate"],
                        reset=reset,
                        verify=lambda: verify("multitask_candidate"),
                        output=output,
                        kernel_config=configs["multitask_candidate"],
                    )
            else:
                report["stage"] = "paired_l1"
                save(output, report)
                report["l1"] = profile(entries, reset, verify, output, configs, args.windows, args.measure_outer)
                report["performance_measured"] = True
                if has_scale_layout and args.dep_gen:
                    from layout_probe import verify_edges

                    first = next(w for w in report["l1"]["windows"] if w["side"] == "multitask_candidate")
                    report["layout_dependency"] = verify_edges(json.loads(Path(first["trace"]).read_text()))
                    if (
                        (args.history + 6) // 4 > 8192
                        and tokens >= 24
                        and not report["layout_dependency"]["applicable"]
                    ):
                        raise AssertionError("长G6实际未执行布局证明task")
                elif has_scale_layout:
                    report["layout_dependency"] = {
                        "collected": False,
                        "reason": "本诊断关闭依赖图快照；不据本轮声称核对过实际fanin",
                    }
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
            if args.boundary_replay:
                from boundary_replay import validate as validate_boundary

                report["stage"] = "boundary_replay"
                save(output, report)
                report["boundary_replay"] = validate_boundary(
                    config,
                    layer,
                    fixture,
                    call,
                    original,
                    entries["multitask_original"],
                    entries["multitask_candidate"],
                    output / "boundary_replay.json",
                    args.history,
                    args.batch,
                    context_switch_history=(
                        args.boundary_context_history
                        if args.boundary_context_history is not None
                        else manifest.get("context_switch_history")
                    ),
                )
            report["actual_ptoas_invocations"] = check_ptoas_invocations(
                output / "COMPILER_INVOCATIONS.jsonl", report["toolchain_identity"]
            )
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
