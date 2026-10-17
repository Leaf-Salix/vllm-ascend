# SPDX-License-Identifier: Apache-2.0
"""冻结d253的Attention逐tick诊断，仅128K/B16 baseline-only，三次A/B/A重放。

复用正式 run_device 的权重、fixture、原 task 数值 oracle 和保护区检查。
同一 Native 根地址、每入口独立 workspace/signal；先 eager/capture 验证，
再按 A→B→A 输入交替顺序重放三轮。没有 DFX/L1 性能窗口；全部时钟只做归因。
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

MAIN = Path(
    "/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/"
    "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
)
HERE = Path(__file__).resolve().parent


def save(output, report):
    temporary = output / "result.tmp.json"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output / "result.json")
    print(json.dumps({k: report.get(k) for k in ("stage", "passed", "device_execution")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    if "TASK_DEVICE" not in os.environ or args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("必须通过 task-submit --device auto 使用分配到的设备")
    if bool(args.candidate_source) == args.baseline_only:
        raise ValueError("配对模式提供 --candidate-source；仅基线提供 --baseline-only")
    if not args.baseline_only or args.candidate_source or (args.history, args.batch) != (131072, 16):
        raise ValueError("本诊断仅开放冻结d253的128K/B16 baseline-only；不扩档")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        stage="initializing",
        passed=False,
        device_execution=False,
        full_csa_validated=False,
        eligible_for_performance=False,
        performance_measured=False,
        native_attention_run=False,
        same_process=True,
        same_root_addresses=True,
        baseline_only=args.baseline_only,
        history=args.history,
        batch=args.batch,
        query_tokens=6,
        rounds=3,
        input_sequence=["A", "B", "A"],
        wait_backoff_ticks=100,
        ffts_meaning="wait_flag_dev原位Scalar起止；A3语义须结合既有核对，仍不把混合阶段区间当纯数学或DMA完成",
        other_is_pure_compute=False,
        checks=[],
        clock_rounds=[],
    )
    try:
        sys.path.insert(0, str(MAIN))
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
        from dsv4_csa_validation import compare_tensor
        from offline_pd.run import decode_additional_config
        from attention_tick_clock import compilation_probe, export_clock
        from pypto import passes
        from pypto.runtime import CompileOptions
        from source_loader import load_binding
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        current_platform.pre_register_and_update()
        device_id = args.device
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("构造 Native fixture 需要既有自定义算子包")
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

        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

        pypto_root = Path(os.environ["CSA_STATIC_PYPTO_ROOT"]).resolve()
        simpler_root = Path(os.environ["HCA_SIMPLER_ROOT"]).resolve()
        for module in (pypto_core, _torch_npu):
            assert pypto_root in Path(module.__file__).resolve().parents
        for module in (simpler, _task_interface):
            assert simpler_root in Path(module.__file__).resolve().parents
        report["environment"] = dict(
            pypto=pypto_core.__file__,
            adapter=_torch_npu.__file__,
            simpler=_task_interface.__file__,
            device=device_id,
        )
        sources = {"spmd_reference": args.baseline_source}
        if not args.baseline_only:
            sources["spmd_candidate"] = args.candidate_source
        bindings = {name: load_binding(path, name + "_clock") for name, path in sources.items()}
        contract = json.loads((MAIN / "source_interface.json").read_text())
        phase_count = contract["phase_count"]
        phase_names = {v: k for k, v in contract["phases"].items()}
        report["sources"] = {name: str(binding.source) for name, binding in bindings.items()}
        for name, binding in bindings.items():
            assert binding.phase_count == phase_count
            destination = args.output / "sources" / name
            destination.mkdir(parents=True)
            for path in binding.source.glob("*.py"):
                (destination / path.name).write_text(path.read_text())
        for path in (
            Path(__file__),
            HERE / "attention_tick_clock.py",
            HERE / "run_device.sh",
            MAIN / "compile_diagnostics.py",
            MAIN / "source_loader.py",
            MAIN / "source_interface.json",
        ):
            (args.output / path.name).write_text(path.read_text())
        with native_session(config, device_id), torch.inference_mode():
            device = torch.device(f"npu:{device_id}")
            layer, report["weights"] = make_layer(config, Path(config.model_config.model), device, 4)
            report["stage"] = "formal_weights_loaded"
            save(args.output, report)
            fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 1024, device)
            idx_group = fixture["groups"]["indexer"]
            scale = idx_group["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            idx_group["initial"] = idx_group["allocation"].cpu()
            pypto.torch.init(
                device=device_id,
                platform="a2a3",
                runtime="tensormap_and_ringbuffer",
                enable_chip_swimlane=0,
                enable_dep_gen=False,
                output_dir=None,
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

            entries = {}
            for name, binding in bindings.items():
                policy = binding.select_policy(
                    tokens=tokens,
                    requests=args.batch,
                    max_query_tokens=6,
                    max_seq_len=args.history + 6,
                )
                entries[name] = binding.prepared(
                    call.args,
                    max_query_tokens=6,
                    max_seq_len=args.history + 6,
                    policy=policy,
                    allocator=lambda key, shape, dtype, device, prefix=name: allocate(
                        prefix + "/" + key,
                        shape,
                        dtype,
                        device,
                    ),
                )
                assert set(entries[name].buffers) == set(contract["runtime_tensors"])
                for key, value in call.args.items():
                    if isinstance(value, torch.Tensor):
                        assert entries[name].native[key].data_ptr() == value.data_ptr()
            report["policies"] = {name: item.policy.constexprs() for name, item in entries.items()}
            report["workspace_bytes"] = {
                name: sum(v.numel() * v.element_size() for v in item.allocations.values())
                for name, item in entries.items()
            }
            compile_options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
            input_a = fixture["hidden"].clone()
            inputs = {"A": input_a, "B": -input_a}
            baseline = {}
            for label, hidden in inputs.items():
                restore(fixture)
                fixture["hidden"].copy_(hidden)
                report["stage"] = "dynamic_oracle_" + label
                save(args.output, report)
                adapter.decode_csa_tp1_layer_test(*call.core_args, config=compile_options)
                torch.npu.synchronize()
                baseline[label] = collect_state(fixture, output, call.args["idx_topk"])
                baseline[label]["idx_topk_scores"] = call.args["idx_topk_scores"].cpu()
                assert all(c["status"] == "PASS" for c in guard_checks(fixture).values())
            generations = dict.fromkeys(entries, 0)

            def reset(name, label):
                restore(fixture)
                fixture["hidden"].copy_(inputs[label])
                output.fill_(float("nan"))
                call.args["idx_topk"].fill_(-777)
                call.args["idx_topk_scores"].fill_(float("nan"))
                for key, value in entries[name].allocations.items():
                    if key not in ("signal", "ffts"):
                        value.fill_(37)

            def checked(name, label, stage):
                state = collect_state(fixture, output, call.args["idx_topk"])
                state["idx_topk_scores"] = call.args["idx_topk_scores"].cpu()
                numerical = {key: compare_tensor(value, baseline[label][key], 0, 0) for key, value in state.items()}
                guards = guard_checks(fixture)
                scratch = {
                    key: bool((base[:guard].cpu() == 37).all() and (base[-guard:].cpu() == 37).all())
                    for key, (base, guard) in allocations.items()
                }
                signal = entries[name].buffers["signal"].cpu().view(phase_count, 72, 128)
                padding = bool((signal[:, :, 1:16] == 0).all() and (signal[:, :, 20:32] == 0).all() and (signal[:, :, 64:] == 0).all())
                protocol = bool((signal[0, :, 0] == generations[name]).all() and padding)
                check = dict(
                    side=name,
                    label=label,
                    stage=stage,
                    generation=generations[name],
                    numerical=numerical,
                    native_guards=guards,
                    scratch_guards=scratch,
                    protocol=protocol,
                    observed_phase_slots=signal[:, :, 0].tolist(),
                )
                check["passed"] = bool(
                    all(c["status"] == "PASS" for c in numerical.values())
                    and all(c["status"] == "PASS" for c in guards.values())
                    and all(scratch.values())
                    and protocol
                )
                report["checks"].append(check)
                save(args.output, report)
                if not check["passed"]:
                    torch.save({"actual": state, "oracle": baseline[label]}, args.output / "failure_states.pt")
                    raise AssertionError("完整输出/state/cache/topk/保护区/代际检查失败")
                return signal

            # 两入口各编译一次，hook 仅在该编译上下文生效；不改共享环境。
            for name, item in entries.items():
                reset(name, "A")
                report["stage"] = name + "_compile"
                save(args.output, report)
                build_root = Path(os.environ["PYPTO_PROG_BUILD_DIR"])
                before = set(build_root.glob("decode_csa_spmd_*/kernel_config.py"))
                with compilation_probe(args.output / (name + "_codegen")):
                    item(config=compile_options)
                torch.npu.synchronize()
                generations[name] += 1
                report["device_execution"] = True
                after = set(build_root.glob("decode_csa_spmd_*/kernel_config.py"))
                assert len(after - before) == 1
                table = next(iter(after - before))
                kernels = list(table.parent.glob("kernels/*/*.cpp"))
                assert len(kernels) == 2
                for kernel in kernels:
                    text = kernel.read_text()
                    assert text.count("CSA_PROFILE_NOTIFY(") > 2 and "CSA_ATTN_PROBE_BEGIN" in text
                    assert "csa_dst[csa_i] = csa_at[csa_i]" in text
                report.setdefault("kernel_configs", {})[name] = str(table)
                report.setdefault("codegen_audits", {})[name] = json.loads(
                    (args.output / (name + "_codegen") / "audit.json").read_text()
                )
                checked(name, "A", "eager")

            # 每侧一次 capture/replay 检查兼作预热，阶段诊断随后统一交替三轮。
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            graphs = {}
            with torch.npu.stream(stream):
                for name, item in entries.items():
                    reset(name, "A")
                    stream.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        item(config=compile_options)
                    graph_signal = item.buffers["signal"].cpu().view(phase_count, 72, 128)
                    assert bool((graph_signal[0, :, 0] == generations[name]).all())
                    graphs[name] = graph
                    reset(name, "A")
                    graph.replay()
                    stream.synchronize()
                    generations[name] += 1
                    checked(name, "A", "graph_warmup")
                sides = list(entries)
                for round_index, label in enumerate(("A", "B", "A")):
                    order = sides if round_index % 2 == 0 else list(reversed(sides))
                    for name in order:
                        report["stage"] = f"round_{round_index}_{name}_{label}"
                        save(args.output, report)
                        reset(name, label)
                        stream.synchronize()
                        graphs[name].replay()
                        stream.synchronize()
                        generations[name] += 1
                        signal = checked(name, label, f"round_{round_index}")
                        clock = export_clock(
                            signal,
                            args.output / f"clock_round_{round_index}_{name}",
                            phase_names,
                        )
                        report["clock_rounds"].append(dict(round=round_index, side=name, label=label, clock=clock))
                        save(args.output, report)
            for name, binding in bindings.items():
                snapshot = args.output / "sources" / name
                assert {p.name for p in snapshot.glob("*.py")} == {p.name for p in binding.source.glob("*.py")}
                assert all(
                    (snapshot / path.name).read_text() == path.read_text() for path in binding.source.glob("*.py")
                )
            report.update(
                stage="complete",
                passed=True,
                full_csa_validated=True,
                graph_replay_validated=True,
                source_direct_text_unchanged=True,
                generations=generations,
                validation_scope="单卡当前 B/S/H；正式权重、合成历史；逐次精确对原 task，阶段插桩不能报性能收益",
            )
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc(), passed=False)
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()
