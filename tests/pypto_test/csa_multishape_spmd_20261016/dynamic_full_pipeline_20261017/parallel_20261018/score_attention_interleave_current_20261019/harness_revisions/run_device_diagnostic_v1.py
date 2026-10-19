# SPDX-License-Identifier: Apache-2.0
"""正式动态单SPMD与原动态分task CSA的单卡对照；功能通过后可选配对L1计时。

复用既有正式权重加载、Native cache/metadata fixture、保护区与状态检查，
候选只引用ops/pypto正式动态源码，不导入静态排程生成器。首轮失败即停止，
保存完整差异及中间量供定位；通过后再做同址A→B→A代际验证。
"""

import argparse
import contextlib
import json
import os
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BASE_D = HERE.parents[1]
REPO = BASE_D.parents[3]

from signal_checks import signal_padding_ok, signal_padding_status, signal_view


def save(output, report):
    temporary = output / "result.tmp.json"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output / "result.json")
    print(
        json.dumps({k: report.get(k) for k in ("stage", "passed", "device_execution")}, ensure_ascii=False), flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--indexer-page-pattern", choices=("original", "ascending_history", "fragmented_history"), default="original")
    parser.add_argument("--launches", type=int, choices=(1, 3), default=3)
    parser.add_argument("--l1-windows", type=int, default=0)
    parser.add_argument("--wait-backoff", action="store_true")
    parser.add_argument("--phase-clock", action="store_true")
    parser.add_argument("--plan-probe", action="store_true")
    parser.add_argument("--address-probe", action="store_true")
    parser.add_argument("--group-probe", action="store_true")
    parser.add_argument("--scale-page-probe", action="store_true")
    parser.add_argument("--metadata-replay", action="store_true")
    parser.add_argument("--padding-replay", action="store_true")
    parser.add_argument("--oa-cols", type=int, choices=(128, 256), default=None)
    parser.add_argument("--output-group-pipeline", choices=("on", "off"), default=None)
    parser.add_argument("--output-policy-matrix", action="store_true")
    parser.add_argument("--hc-row-dependencies", choices=("on", "off"), default=None)
    parser.add_argument("--hc-policy-matrix", action="store_true")
    parser.add_argument("--hc-consumer-reduce", choices=("on", "off"), default=None)
    parser.add_argument("--hc-consumer-matrix", action="store_true")
    parser.add_argument("--attention-query-metadata", choices=("on", "off"), default=None)
    parser.add_argument("--attention-query-matrix", action="store_true")
    parser.add_argument("--candidate-ptoas-root", type=Path, required=True)
    parser.add_argument("--spmd-source", type=Path, default=HERE / "source_candidate")
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--source-matrix-reference", type=Path)
    parser.add_argument("--static-reference", action="store_true",
                        help="加入历史selected静态SPMD，同Native根/独立暂存/原冻结调度")
    parser.add_argument("--failure-snapshot", action="store_true",
                        help="失败后仅尝试ACL读取信号/组metadata，不继续执行任务")
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    if args.phase_clock:
        raise ValueError("交错候选runner保持扩展padding全零；本入口不支持phase-clock插桩")
    if "TASK_DEVICE" not in os.environ:
        raise RuntimeError("必须通过task-submit --device auto取得设备")
    if args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("命令行设备必须与队列分配一致")
    if args.l1_windows and args.l1_windows < 3:
        raise ValueError("性能对比至少三个配对采样窗口")
    if args.phase_clock and (args.l1_windows or args.launches != 1):
        raise ValueError("阶段插桩只执行一次候选，不与正式性能采样混用")
    if args.metadata_replay and (args.phase_clock or args.history < 2):
        raise ValueError("metadata重放要求无阶段插桩、history>=2")
    if args.padding_replay and (args.phase_clock or args.batch < 2):
        raise ValueError("padding重放要求无阶段插桩、batch>=2")
    if sum((args.output_policy_matrix, args.hc_policy_matrix, args.attention_query_matrix, args.hc_consumer_matrix, bool(args.source_matrix_reference))) > 1:
        raise ValueError("每轮只比较一个策略轴，避免组合过多掩盖收益来源")
    if (args.output_policy_matrix or args.hc_policy_matrix or args.attention_query_matrix or args.hc_consumer_matrix or args.source_matrix_reference) and not args.l1_windows:
        raise ValueError("输出策略矩阵需要显式指定配对L1窗口数")
    if args.static_reference and any((args.output_policy_matrix, args.hc_policy_matrix,
                                    args.attention_query_matrix, args.hc_consumer_matrix,
                                    args.source_matrix_reference)):
        raise ValueError("静态对照本轮使用三角色入口，暂不与其他策略矩阵叠加")
    if args.static_reference:
        from static_baseline_adapter import resolve_selected_case

        # 在加载权重和上卡前拒绝缺少历史验收的规格，不能默默借用邻档。
        resolve_selected_case(BASE_D.parent / "selected_shape_policies.json",
                              history=args.history, batch=args.batch)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        stage="initializing",
        passed=False,
        device_execution=False,
        full_csa_validated=False,
        performance_measured=False,
        native_attention_run=False,
        source="ops/pypto/deepseek_v4_flash_csa_spmd源级动态入口",
        comparison="当前原动态分task PTO；同输入/正式第4层权重/合成历史",
        batch=args.batch,
        history=args.history,
        query_tokens=6,
        launches=[],
        wait_backoff=args.wait_backoff,
        phase_clock=args.phase_clock,
    )
    try:
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
        from dsv4_csa_validation import compare_tensor
        from offline_pd.run import decode_additional_config
        from pypto import passes
        from pypto.runtime import CompileOptions
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        current_platform.pre_register_and_update()
        device_id = int(os.environ["TASK_DEVICE"])
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("构造Native fixture需要既有自定义算子包")
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
                "cache_dir": str(args.output / "compile_cache"),
            },
            additional_config=additional,
        ).create_engine_config()
        import _task_interface
        import simpler
        from pypto import _torch_npu, pypto_core

        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa import native_adapter as adapter
        from source_loader import load_binding
        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

        isolated = load_binding(args.spmd_source, "candidate")
        PreparedDynamicCSA, select_policy, PHASE_COUNT = isolated.prepared, isolated.select_policy, isolated.phase_count
        # 本轮只比较预取源码；三方共用同一冻结v3，保留调用记录确认未混用。
        assert args.candidate_ptoas_root.resolve() == Path(os.environ["PTOAS_ROOT"]).resolve()
        from compiler_context import bind_compiler, record_invocations
        record_invocations(args.output / "COMPILER_INVOCATIONS.jsonl")
        PreparedDynamicCSA = bind_compiler(PreparedDynamicCSA, args.candidate_ptoas_root)
        candidate_files = {path.name: path.read_bytes() for path in isolated.source.glob("*.py")}
        reference_files = {path.name: path.read_bytes() for path in args.source_matrix_reference.glob("*.py")}
        source_differences = sorted(name for name in candidate_files.keys() | reference_files.keys()
                                    if candidate_files.get(name) != reference_files.get(name))
        report["compiler_comparison"] = {
            "candidate": str(args.candidate_ptoas_root),
            "reference": os.environ["PTOAS_ROOT"],
            "same_operator_source": not source_differences,
            "operator_source_differences": source_differences,
        }
        report["source_package"] = str(isolated.source)

        pypto_root = Path(os.environ["CSA_STATIC_PYPTO_ROOT"]).resolve()
        simpler_root = Path(os.environ["HCA_SIMPLER_ROOT"]).resolve()
        assert pypto_root in Path(pypto.__file__).resolve().parents
        for module in (pypto_core, _torch_npu):
            # overlay只复用原二进制符号链接，不能要求其resolve目标位于overlay。
            expected_binary = pypto_root / "python/pypto" / Path(module.__file__).name
            assert Path(module.__file__).resolve() == expected_binary.resolve()
        for module in (simpler, _task_interface):
            assert simpler_root in Path(module.__file__).resolve().parents
        report["environment"] = {
            "pypto": pypto_core.__file__,
            "pypto_python": pypto.__file__,
            "pypto_binary_resolved": str(Path(pypto_core.__file__).resolve()),
            "adapter": _torch_npu.__file__,
            "simpler": _task_interface.__file__,
            "device": device_id,
        }
        (args.output / "source").mkdir()
        package = isolated.source
        for path in package.glob("*.py"):
            (args.output / "source" / path.name).write_text(path.read_text())
        for script in (
            Path(__file__),
            HERE / "l1_compare.py",
            HERE / "compiler_context.py",
            HERE / "compile_diagnostics.py",
            HERE / "plan_device_probe.py",
            HERE / "address_device_probe.py",
            HERE / "group_device_probe.py",
            HERE / "scale_page_device_probe.py",
            HERE / "metadata_replay.py",
            HERE / "padding_replay.py",
            HERE / "output_policy_matrix.py",
            HERE / "check_output_phase_signals.py",
            HERE / "source_loader.py",
            HERE / "signal_checks.py",
            HERE / "static_baseline_adapter.py",
            HERE / "wave_signal_checks.py",
            HERE / "failure_snapshot.py",
            HERE / "run_device.sh",
        ):
            (args.output / script.name).write_text(script.read_text())
        with native_session(config, device_id), torch.inference_mode():
            device = torch.device(f"npu:{device_id}")
            layer, report["weights"] = make_layer(config, Path(config.model_config.model), device, 4)
            report["stage"] = "formal_weights_loaded"
            save(args.output, report)
            fixture = make_fixture(
                config, layer.self_attn, args.batch, args.history, 1024, device,
                indexer_page_pattern=args.indexer_page_pattern,
            )
            report["indexer_page_pattern"] = {
                "pattern": args.indexer_page_pattern,
                "full_history_pages": args.history // (4 * 32),
                "current_slot_mapping_unchanged": True,
                "canonical_seven_input": args.indexer_page_pattern == "original",
                "native_metadata_generated_after_page_layout": True,
            }
            # 保留原对照用非均匀scale，避免常量scale掩盖页/行索引错误。
            idx_group = fixture["groups"]["indexer"]
            scale = idx_group["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            idx_group["initial"] = idx_group["allocation"].cpu()
            pypto.torch.init(
                device=device_id,
                platform="a2a3",
                runtime="tensormap_and_ringbuffer",
                enable_chip_swimlane=1 if args.l1_windows else 0,
                enable_dep_gen=bool(args.l1_windows),
                output_dir=str(args.output / "dfx") if args.l1_windows else None,
                **ring_sizing_kwargs(additional),
            )
            groups = {
                name: (fixture["metadata"][group["prefix"]], group["views"])
                for name, group in fixture["groups"].items()
            }
            weights = adapter.prepare_weights(layer.self_attn, groups["indexer"][0].hadamard, layer)
            output = torch.empty_like(fixture["hidden"])
            call = adapter.NativeCSACall(
                adapter.CSAOperators.register(),
                weights,
                fixture["hidden"],
                fixture["positions"],
                groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                compact_metadata=fixture["compact"],
                buffers={
                    "idx_topk_scores": torch.empty((tokens, 512), dtype=torch.float32, device=device),
                    "idx_topk": torch.empty((tokens, 512), dtype=torch.int32, device=device),
                    "x_out": output,
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

            policy = select_policy(tokens=tokens, requests=args.batch, max_query_tokens=6, max_seq_len=args.history + 6)
            # 仅研究脚本显式覆盖数学/编排旋钮；正式入口仍使用工作量区间策略。
            # 实际选择随result保存，不能把N128结果标作默认N256或隐去消融配置。
            if args.oa_cols is not None:
                policy = replace(policy, oa_cols=args.oa_cols)
            if args.output_group_pipeline is not None:
                policy = replace(policy, output_group_pipeline=args.output_group_pipeline == "on")
            if args.hc_row_dependencies is not None:
                policy = replace(policy, hc_row_dependencies=args.hc_row_dependencies == "on")
            if args.hc_consumer_reduce is not None:
                policy = replace(policy, hc_consumer_reduce=args.hc_consumer_reduce == "on")
            if args.attention_query_metadata is not None:
                policy = replace(policy, attention_query_metadata=args.attention_query_metadata == "on")
            if args.policy_json:
                overrides = json.loads(args.policy_json.read_text())
                if not isinstance(overrides, dict) or any(type(v) not in (bool, int) for v in overrides.values()):
                    raise ValueError("策略覆盖只允许字段到int/bool的JSON对象")
                policy = replace(policy, **overrides)
            candidate = PreparedDynamicCSA(
                call.args, max_query_tokens=6, max_seq_len=args.history + 6, allocator=allocate, policy=policy
            )
            from wave_signal_checks import WaveSignalVerifier

            wave_verifier = WaveSignalVerifier(candidate)
            contract = json.loads((HERE / "source_interface.json").read_text())
            assert set(candidate.buffers) == set(contract["runtime_tensors"])
            report.update(
                stage="bound",
                policy=candidate.policy.constexprs(),
                shapes={k: list(v.shape) for k, v in candidate.buffers.items()},
                tensor_count=len(candidate.buffers),
                workspace_bytes=sum(v.numel() * v.element_size() for v in candidate.allocations.values()),
            )
            save(args.output, report)
            compile_options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
            static_reference = None
            if args.static_reference:
                from static_baseline_adapter import prepare_static_baseline

                static_reference = prepare_static_baseline(
                    call.args, registry=BASE_D.parent / "selected_shape_policies.json",
                    history=args.history, batch=args.batch, output=args.output / "static_reference")
                report["static_reference"] = dict(source=static_reference.source_report, launches=[])
            if args.scale_page_probe:
                from scale_page_device_probe import validate as validate_scale_pages

                report["scale_page_probe"] = validate_scale_pages(
                    device, compile_options, args.output / "scale_page_device_probe.json"
                )
            input_a = fixture["hidden"].clone()
            input_b = -input_a
            baseline = {}
            for label, hidden in (("A", input_a), ("B", input_b))[: 1 if args.launches == 1 else 2]:
                restore(fixture)
                fixture["hidden"].copy_(hidden)
                report["stage"] = f"reference_{label}_enqueue"
                save(args.output, report)
                adapter.decode_csa_tp1_layer_test(*call.core_args, config=compile_options)
                torch.npu.synchronize()
                baseline[label] = collect_state(fixture, output, call.args["idx_topk"])
                baseline[label]["idx_topk_scores"] = call.args["idx_topk_scores"].cpu()
                reference_guards = guard_checks(fixture)
                report.setdefault("reference_guards", {})[label] = reference_guards
                save(args.output, report)
                assert all(c["status"] == "PASS" for c in reference_guards.values())
            report["stage"] = "reference_complete"
            save(args.output, report)
            for generation, label in enumerate(["A", "B", "A"][: args.launches], 1):
                restore(fixture)
                fixture["hidden"].copy_(input_a if label == "A" else input_b)
                output.fill_(float("nan"))
                call.args["idx_topk"].fill_(-777)
                # signal必须跨调用保留；暂存填毒让漏写暴露，不让上次内容掩盖问题。
                for name, value in candidate.allocations.items():
                    if name not in ("signal", "ffts"):
                        value.fill_(37)
                report["stage"] = f"candidate_{generation}_{label}_compile_and_call"
                save(args.output, report)
                compilation = contextlib.nullcontext()
                if generation == 1 and (args.wait_backoff or args.phase_clock):
                    from compile_diagnostics import compilation_probe

                    compilation = compilation_probe(
                        args.output / "codegen_probe", wait_backoff=args.wait_backoff, phase_clock=args.phase_clock
                    )
                with compilation:
                    candidate(config=compile_options)
                torch.npu.synchronize()
                if generation == 1 and (args.wait_backoff or args.phase_clock):
                    report["codegen_probe"] = json.loads((args.output / "codegen_probe/audit.json").read_text())
                    # 不把命令行开关当作编译实际生效的证据；检查最终CCEC输入。
                    artifacts = list((args.output / "build_output").glob("decode_csa_spmd_*/kernels/*/*.cpp"))
                    if len(artifacts) != 2:
                        raise AssertionError("无法唯一定位候选实际AIC/AIV生成源码")
                    for artifact in artifacts:
                        source = artifact.read_text()
                        required = "CSA_PROFILE_NOTIFY(" if args.phase_clock else "CSA_WAIT_LOCAL_GM("
                        if source.count(required) < 2:
                            raise AssertionError("实际kernel没有采用请求的源码变换")
                report["device_execution"] = True
                report["stage"] = f"candidate_{generation}_{label}_readback_and_compare"
                save(args.output, report)
                if generation == 1:
                    # 只在数值验证阶段读回计划，性能窗口内不执行。确认实际命中
                    # 连续快路径/碎片回退，避免仅凭源码分支存在推断设备执行路径。
                    from layout_plan_audit import validate as validate_layout_plan
                    report["layout_plan_audit"] = validate_layout_plan(candidate)
                    save(args.output, report)
                    if not report["layout_plan_audit"]["passed"]:
                        raise AssertionError("前置scale页布局计划与Native实际页表不一致")
                actual = collect_state(fixture, output, call.args["idx_topk"])
                actual["idx_topk_scores"] = call.args["idx_topk_scores"].cpu()
                comparisons = {
                    name: compare_tensor(value, baseline[label][name], 0, 0) for name, value in actual.items()
                }
                guards = guard_checks(fixture)
                scratch_guards = {
                    name: bool((base[:g].cpu() == 37).all() and (base[-g:].cpu() == 37).all())
                    for name, (base, g) in allocations.items()
                }
                signal = signal_view(candidate)
                padding_status = signal_padding_status(signal)
                padding_ok = all(padding_status.values())
                # 协议检查自身报错时也保留已经完成的独立数值/保护区结果，
                # 避免一次测试断言遮蔽上卡计算是否完成、有哪些实际差异。
                report["pending_checks"] = dict(
                    generation=generation, label=label, against_dynamic=comparisons,
                    guards=guards, scratch_guards=scratch_guards,
                    signal_padding=padding_status,
                )
                save(args.output, report)
                wave_protocol = wave_verifier.check(signal, generation)
                protocol = bool((signal[0, :, 0] == generation).all() and padding_ok and wave_protocol["passed"])
                if args.phase_clock:
                    from compile_diagnostics import export_clock

                    report["phase_clock_result"] = export_clock(
                        signal, args.output / "phase_clock", {v: k for k, v in contract["phases"].items()}
                    )
                checks = dict(
                    generation=generation,
                    label=label,
                    against_dynamic=comparisons,
                    guards=guards,
                    scratch_guards=scratch_guards,
                    protocol=protocol,
                    # 原输出owner检查继续只接收完全相同的60-phase前缀。
                    observed_phase_slots=signal[:PHASE_COUNT, :, 0].tolist(),
                    observed_wave_phase_slots=signal[PHASE_COUNT:, :, 0].tolist(),
                    signal_phase_count=signal.shape[0],
                    signal_padding=padding_status,
                    wave_protocol=wave_protocol,
                )
                checks["passed"] = (
                    all(c["status"] == "PASS" for c in comparisons.values())
                    and all(c["status"] == "PASS" for c in guards.values())
                    and all(scratch_guards.values())
                    and protocol
                )
                report["launches"].append(checks)
                report.pop("pending_checks", None)
                report["stage"] = f"candidate_{generation}_checked"
                save(args.output, report)
                if not checks["passed"]:
                    torch.save({"candidate": actual, "reference": baseline[label]}, args.output / "failure_states.pt")
                    torch.save(
                        {name: value.cpu() for name, value in candidate.allocations.items()},
                        args.output / "failure_workspace.pt",
                    )
                    raise AssertionError("完整动态SPMD对照/保护区/代际检查失败；保留诊断，不继续重放")
            report["full_csa_validated"] = True
            if static_reference is not None:
                report["stage"] = "static_reference_compile"
                save(args.output, report)
                report["static_reference"]["compile"] = static_reference.compile(config=compile_options)
                for generation, label in enumerate(["A", "B", "A"][:args.launches], 1):
                    restore(fixture)
                    fixture["hidden"].copy_(input_a if label == "A" else input_b)
                    output.fill_(float("nan"))
                    call.args["idx_topk"].fill_(-777)
                    call.args["idx_topk_scores"].fill_(float("nan"))
                    static_reference.poison_temporaries()
                    static_reference(config=compile_options)
                    torch.npu.synchronize()
                    state = collect_state(fixture, output, call.args["idx_topk"])
                    state["idx_topk_scores"] = call.args["idx_topk_scores"].cpu()
                    compared = {name: compare_tensor(value, baseline[label][name], 0, 0)
                                for name, value in state.items()}
                    guards = guard_checks(fixture)
                    checked = dict(generation=generation, label=label, against_dynamic=compared,
                                   guards=guards, protocol=static_reference.readback(generation))
                    checked["passed"] = (all(c["status"] == "PASS" for c in compared.values())
                                         and all(c["status"] == "PASS" for c in guards.values())
                                         and checked["protocol"]["passed"])
                    report["static_reference"]["launches"].append(checked)
                    report["stage"] = f"static_reference_{generation}_checked"
                    save(args.output, report)
                    if not checked["passed"]:
                        torch.save({"static": state, "reference": baseline[label]},
                                   args.output / "static_failure_states.pt")
                        raise AssertionError("历史静态同根对照失败，禁止继续采性能")
            if args.metadata_replay:
                from metadata_replay import validate as validate_metadata_replay

                report["stage"] = "metadata_replay"
                save(args.output, report)
                report["metadata_replay"] = validate_metadata_replay(
                    config,
                    layer,
                    fixture,
                    call,
                    candidate,
                    allocations,
                    compile_options,
                    args.output / "metadata_replay.json",
                    args.history,
                    args.batch,
                )
                report["graph_replay_validated"] = True
            if args.padding_replay:
                from padding_replay import validate as validate_padding_replay

                report["stage"] = "padding_replay"
                save(args.output, report)
                report["padding_replay"] = validate_padding_replay(
                    layer,
                    fixture,
                    call,
                    candidate,
                    allocations,
                    compile_options,
                    args.output / "padding_replay.json",
                    args.batch,
                )
                report["graph_replay_validated"] = True
            if args.plan_probe:
                from plan_device_probe import validate as validate_plan

                report["plan_probe"] = validate_plan(device, compile_options, args.output / "plan_device_probe.json")
            if args.address_probe:
                from address_device_probe import validate as validate_addresses

                report["address_probe"] = validate_addresses(
                    device, compile_options, args.output / "address_device_probe.json"
                )
            if args.group_probe:
                from group_device_probe import validate as validate_groups

                report["group_probe"] = validate_groups(
                    device, compile_options, args.output / "group_device_probe.json"
                )
            if args.l1_windows:
                from l1_compare import profile_pair

                def reset():
                    restore(fixture)
                    fixture["hidden"].copy_(input_a)

                def verify():
                    state = collect_state(fixture, output, call.args["idx_topk"])
                    state["idx_topk_scores"] = call.args["idx_topk_scores"].cpu()
                    compared = {name: compare_tensor(value, baseline["A"][name], 0, 0) for name, value in state.items()}
                    if not all(c["status"] == "PASS" for c in compared.values()):
                        raise AssertionError("图重放输出/state与eager参考不一致：" + str(compared))
                    return compared

                def update_stage(stage):
                    report["stage"] = stage
                    save(args.output, report)

                before_l1 = signal_view(candidate)[0, :, 0]
                if not bool((before_l1 == before_l1[0]).all()):
                    raise AssertionError("L1前参与核代际不一致")
                if args.output_policy_matrix or args.hc_policy_matrix or args.attention_query_matrix or args.hc_consumer_matrix or args.source_matrix_reference:
                    from output_policy_matrix import run_matrix

                    matrix_axis = "source" if args.source_matrix_reference else "hc_consumer" if args.hc_consumer_matrix else "attention_query" if args.attention_query_matrix else "hc_rows" if args.hc_policy_matrix else "output"
                    matrix_key = {
                        "source": "source_matrix",
                        "attention_query": "attention_query_policy_matrix",
                        "hc_consumer": "hc_consumer_policy_matrix",
                        "hc_rows": "hc_row_policy_matrix",
                        "output": "output_policy_matrix",
                    }[matrix_axis]
                    report[matrix_key] = run_matrix(
                        existing=candidate,
                        max_seq_len=args.history + 6,
                        allocate=allocate,
                        baseline_entry=lambda: adapter.decode_csa_tp1_layer_test(
                            *call.core_args, config=compile_options
                        ),
                        reset=reset,
                        verify=verify,
                        cache_guards=lambda: guard_checks(fixture),
                        compile_options=compile_options,
                        output=args.output,
                        windows=args.l1_windows,
                        wait_backoff=args.wait_backoff,
                        update_stage=update_stage,
                        axis=matrix_axis,
                        reference_source=args.source_matrix_reference,
                    )
                    report["post_l1_protocol"] = all(
                        row["passed"] for row in report[matrix_key]["protocol"].values()
                    )
                else:
                    update_stage("paired_l1")
                    entries = {
                        "dynamic_pto": lambda: adapter.decode_csa_tp1_layer_test(
                            *call.core_args, config=compile_options),
                        "single_spmd": lambda: candidate(config=compile_options),
                    }
                    kernel_configs = {}
                    if static_reference is not None:
                        entries["static_spmd"] = lambda: static_reference(config=compile_options)
                        kernel_configs["static_spmd"] = static_reference.kernel_config

                    def verify_protocol(side, replay_count):
                        if side == "static_spmd":
                            return static_reference.readback(args.launches + replay_count)
                        if side == "single_spmd":
                            signals = signal_view(candidate)
                            wave_protocol = wave_verifier.check(signals, int(before_l1[0]) + replay_count)
                            passed = bool((signals[0, :, 0] == before_l1 + replay_count).all()
                                          and signal_padding_ok(signals) and wave_protocol["passed"])
                            if not passed:
                                raise AssertionError("动态SPMD图重放代际或padding错误")
                            return dict(passed=True, replay_count=replay_count, wave_protocol=wave_protocol)
                        return dict(passed=True, applicable=False)

                    report["l1"] = profile_pair(
                        entries=entries,
                        reset=reset,
                        verify=verify,
                        output=args.output,
                        windows=args.l1_windows,
                        kernel_configs=kernel_configs,
                        verify_protocol=verify_protocol,
                    )
                    signal = signal_view(candidate)
                    expected_generation = before_l1 + report["l1"]["spmd_calls"]
                    report["post_l1_protocol"] = bool(
                        (signal[0, :, 0] == expected_generation).all() and signal_padding_ok(signal)
                    )
                report["post_l1_guards"] = guard_checks(fixture)
                report["post_l1_scratch_guards"] = {
                    name: bool((base[:g].cpu() == 37).all() and (base[-g:].cpu() == 37).all())
                    for name, (base, g) in allocations.items()
                }
                if not (
                    report["post_l1_protocol"]
                    and all(c["status"] == "PASS" for c in report["post_l1_guards"].values())
                    and all(report["post_l1_scratch_guards"].values())
                ):
                    raise AssertionError("L1后保护区或信号代际检查失败")
                report.update(performance_measured=True, graph_replay_validated=True)
            report.update(
                stage="complete",
                passed=True,
                full_csa_validated=True,
                validation_scope="单卡当前B/S/H、正式权重、合成历史；图与性能见独立字段，未覆盖全部动态边界",
            )
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc(), passed=False)
        if args.failure_snapshot and "candidate" in locals():
            try:
                from failure_snapshot import capture

                report["failure_snapshot"] = capture(candidate, args.output / "failure_snapshot.json")
            except Exception as snapshot_error:
                report["failure_snapshot_error"] = str(snapshot_error)
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()
