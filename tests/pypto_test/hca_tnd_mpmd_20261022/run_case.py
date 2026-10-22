# SPDX-License-Identifier: Apache-2.0
"""HCA TND 独立 CPU 编译、单卡 eager 功能和 L1 性能入口。

2026-10-22 用户要求：后续不使用 ACLGraph replay。此入口直接调用两侧算子，
固定初态复核重复执行；L1 也只包围直接调用，不捕获图。历史图基准不混用。
"""

import argparse
import importlib
import importlib.util
import json
import os
import statistics
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PACKAGE = "vllm_ascend.ops.pypto.deepseek_v4_flash_hca_tnd"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--source", type=Path, help="冻结的 ops/pypto 完整依赖目录")
    parser.add_argument("--control-source", type=Path, help="同进程配对的冻结 ops/pypto 目录；不改公共依赖")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--queries", help="逗号分隔的真实 query 长度；默认每请求 6 行")
    parser.add_argument("--histories", help="逐请求历史长度，用于 C128 跨界")
    parser.add_argument("--token-capacity", type=int)
    parser.add_argument("--request-capacity", type=int)
    parser.add_argument("--windows", type=int, default=0)
    parser.add_argument("--native-only", action="store_true", help="只采 Native 直接调用外部 profiling，不使用图")
    parser.add_argument("--profile", action="store_true", help="通过功能检查后采两侧直接调用外部 profiling")
    parser.add_argument("--profile-service", action="store_true",
                        help="外部PTO计时使用真实custom op，每次新forward context，包含metadata生产和描述符绑定")
    parser.add_argument("--host-profile", action="store_true", help="独立测真实服务/预绑定/Native的Host函数耗时，不采PyTorch profiling")
    parser.add_argument("--prepared-host-probe", action="store_true", help="仅诊断SDK已预热registration的提交下界，不改正式入口或SDK")
    parser.add_argument("--diagnostic-perf", action="store_true",
                        help="仅在与冻结控制组全部逐bit一致时，保留 Native 误差并诊断性能；不改 passed")
    parser.add_argument("--native-output-math", action="store_true",
                        help="已做同输入诊断的Native输出量化语义修正；性能门禁仍要求cache/state精确及Native输出通过")
    parser.add_argument("--compare-legacy", action="store_true", help="S6 上核实 TND 迁移是否改变旧 HCA 数学")
    parser.add_argument("--compressor-probe-only", action="store_true", help="固定 Native RMS 输出验证 C128；不测性能")
    parser.add_argument("--metadata-sequence", action="store_true", help="固定设备地址，直接调用 S1/S5/混长 A→B→A；不测性能")
    parser.add_argument("--precision-probe", action="store_true", help="同一生产数学的中间值及Native尾链诊断；不测性能")
    parser.add_argument("--service-probe", action="store_true", help="真实custom op服务接管与KV钩子验证；不测性能")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    if args.control_source and args.compare_legacy:
        raise ValueError("控制组只能选冻结 TND 或旧 HCA 中的一种")
    if args.profile_service and (not (args.profile or args.host_profile) or args.native_only):
        raise ValueError("服务性能采集必须与双侧--profile同时使用")
    if args.host_profile and (not args.profile_service or args.windows or args.profile):
        raise ValueError("Host归因使用--profile-service，且不与L1/PyTorch profiling混用")
    if args.prepared_host_probe and not args.host_profile:
        raise ValueError("预绑定SDK对照只用于独立Host诊断")
    if args.native_output_math and (not args.diagnostic_perf or not args.control_source):
        raise ValueError("Native输出数学迁移必须有冻结控制组和显式诊断性能标记")
    if args.precision_probe and (args.windows or args.profile or args.native_only):
        raise ValueError("中间值诊断不与性能采集混用")
    if args.service_probe and (args.windows or args.profile or args.native_only or args.precision_probe):
        raise ValueError("服务接管验证独立执行，不混入计时或中间值诊断")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(args.output / "build")
    os.environ["PYPTO_CACHE_DIR"] = str(args.output / "cache")
    report = {"passed": False, "stage": "initializing", "device_execution": False,
              "execution_mode": "eager_direct", "aclgraph_replay": False,
              "source": str(args.source), "layer": 3, "nz_mode": 2,
              "shared_toolchain": os.environ.get("CSA_SHARED_TOOLCHAIN_RELEASE")}

    def save(stage):
        report["stage"] = stage
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"stage": stage, "passed": report["passed"]}), flush=True)

    try:
        import dsv4_csa_env

        dsv4_csa_env.activate()
        if args.source:
            import vllm_ascend.ops.pypto as package

            package.__path__ = [str(args.source.resolve())]
        from pypto import passes
        from pypto.runtime import CompileOptions

        entry = importlib.import_module(PACKAGE + ".entry")
        control_package = None
        if args.control_source:
            path = args.control_source.resolve() / "deepseek_v4_flash_hca_tnd"
            control_package = "vllm_ascend.ops.pypto._hca_tnd_control"
            spec = importlib.util.spec_from_file_location(
                control_package, path / "__init__.py", submodule_search_locations=[str(path)],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[control_package] = module
            spec.loader.exec_module(module)
            report["control_source"] = str(path)
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        if args.cpu:
            entry.decode_hca_tp1_layer_test.compile(config=options)
            if args.precision_probe:
                probes = importlib.import_module(PACKAGE + ".trace_entry")
                probes.trace_hca.compile(config=options)
                probes.trace_output.compile(config=options)
            if args.compare_legacy:
                legacy_entry = importlib.import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_hca.decode_hca")
                legacy_entry.decode_hca_tp1_layer_test.compile(config=options)
            if control_package:
                importlib.import_module(control_package + ".entry").decode_hca_tp1_layer_test.compile(config=options)
            # .compile完成IR/PTOAS设备代码，但编排C++原先到首次执行才编译。
            # 在CPU阶段也完成它，避免把明显的编排codegen错误留到占卡之后。
            from pypto.runtime.kernel_compiler import KernelCompiler
            from pypto.runtime.pto_isa import ensure_pto_isa_root
            from concurrent.futures import ThreadPoolExecutor

            compiler = KernelCompiler(platform="a2a3")
            sources = sorted((args.output / "build").glob("*/orchestration/*.cpp"))
            if not sources:
                raise RuntimeError("CPU编译未生成编排源码，不能声称完整后端编译通过")
            report["orchestration_compiles"] = []
            for source in sources:
                binary = compiler.compile_orchestration("tensormap_and_ringbuffer", str(source))
                source.with_suffix(".so").write_bytes(binary)
                report["orchestration_compiles"].append({"source": str(source), "bytes": len(binary)})
            isa_root = str(ensure_pto_isa_root())
            kernels = sorted((args.output / "build").glob("*/kernels/*/*.cpp"))
            if not kernels:
                raise RuntimeError("没有生成incore源码")

            def compile_kernel(source):
                binary = compiler.compile_incore(
                    str(source), core_type=source.parent.name, pto_isa_root=isa_root,
                    runtime_name="tensormap_and_ringbuffer",
                )
                source.with_suffix(".o").write_bytes(binary)
                return {"source": str(source), "bytes": len(binary)}

            # 每个源文件使用独立编译目录；纯CPU完成AscendC编译/链接，不初始化Worker。
            with ThreadPoolExecutor(max_workers=8) as pool:
                report["incore_binary_compiles"] = list(pool.map(compile_kernel, kernels))
            report["passed"] = True
            save("cpu_backend_compiled")
            return

        import pypto.torch
        import torch
        import torch_npu
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import guard_checks, make_layer, restore
        from dsv4_csa_validation import compare_tensor
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_hca import native_attention_half
        from vllm_ascend.utils import enable_custom_op

        # 复用已验证的 Native TND fixture；该工具也支持没有 Indexer 的 C128 层。
        from csa_mpmd_tnd_20261021.scenarios import make_fixture

        device_id = int(os.environ["TASK_DEVICE"])
        if args.device is not None and args.device != device_id:
            raise ValueError("设备参数与 task-submit 分配不一致")
        queries = [int(v) for v in args.queries.split(",")] if args.queries else [6] * args.batch
        histories = [int(v) for v in args.histories.split(",")] if args.histories else [args.history] * args.batch
        if args.metadata_sequence:
            if args.history < 128:
                raise ValueError("同址矩阵需要至少128行历史")
            # 初始化时每请求都关闭一个C128组，为后续零/部分/全部输出提供原有容量。
            histories = [args.history // 128 * 128 - 1] * args.batch
        if len(queries) != args.batch or len(histories) != args.batch:
            raise ValueError("queries/histories 必须逐请求提供")
        report.update(device=device_id, query_lengths=queries, history_lengths=histories,
                      token_capacity=args.token_capacity or sum(queries))
        torch.npu.set_device(device_id)
        # 直接调用仍使用CANN现成编译算子；显式记录实际开关，避免依赖进程默认值。
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        report["execution_config"] = {
            "jit_compile": False,
            "is_jit_compile_false": bool(torch_npu.npu.is_jit_compile_false()),
            "aclgraph_capture": False, "aclgraph_replay": False,
            "superkernel": False,
            "superkernel_note": "遵循用户后续直接调用口径，没有图编译/SuperKernel图优化",
            "weight_nz_mode": 2,
        }
        current_platform.pre_register_and_update()
        if not enable_custom_op():
            raise RuntimeError("Native 自定义算子不可用")
        checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
        config = EngineArgs(
            model=str(checkpoint), tokenizer_mode="deepseek_v4", trust_remote_code=True,
            dtype="bfloat16", quantization="ascend", tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128},
            max_model_len=max(16384, max(histories) + (768 if args.metadata_sequence else 128)),
            max_num_seqs=64, max_num_batched_tokens=384, block_size=32,
            enable_prefix_caching=False, enforce_eager=True,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        device = torch.device(f"npu:{device_id}")
        with native_session(config, device_id), torch.inference_mode():
            layer, report["weights"] = make_layer(config, checkpoint, device, layer_index=3)
            layer.self_attn.dsa_attn._pto_hca_layer = (layer,)
            fixture = make_fixture(
                config, layer.self_attn, args.batch, args.history, 1024, device,
                query_lengths=queries, history_lengths=histories,
                token_capacity=args.token_capacity, request_capacity=args.request_capacity,
                table_history=(args.history // 128 * 128 + 512) if args.metadata_sequence else None,
            )
            output = torch.empty_like(fixture["hidden"])
            native_intermediates = {}
            norm_hook = None
            release_native_capture = None
            if args.precision_probe:
                from precision_probe import capture_native

                native_intermediates, release_native_capture = capture_native(layer)
            if args.compressor_probe_only:
                def capture_norm(module, inputs, value):
                    native_intermediates["normalized"] = value.detach().clone()
                norm_hook = layer.input_layernorm.register_forward_hook(capture_norm)

            def collect():
                return {"output": output.cpu(), **{
                    name: [view.cpu() for view in group["views"]]
                    for name, group in fixture["groups"].items()
                }}

            restore(fixture)
            def native_call():
                with set_ascend_forward_context(
                    fixture["metadata"], config, num_tokens=fixture["tokens"],
                    num_actual_tokens=fixture["actual_tokens"],
                ):
                    native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output)

            native_call()
            torch.npu.synchronize()
            if release_native_capture is not None:
                release_native_capture()
            if norm_hook is not None:
                norm_hook.remove()
            native = collect()
            report["native_guards"] = guard_checks(fixture)
            save("native_reference_ready")
            if args.native_only:
                from profile_eager import profile_call

                def reset_native():
                    restore(fixture)
                    output.fill_(float("nan"))

                report["native_profile"] = profile_call(
                    native_call, reset_native, output_dir=args.output / "native_profile",
                    side="native", samples=args.windows or 10,
                )
                report["post_profile_guards"] = guard_checks(fixture)
                current = collect()
                report["repeat_native"] = {"output": compare_tensor(current["output"], native["output"], 0, 0)}
                for name in fixture["groups"]:
                    report["repeat_native"][name] = [compare_tensor(a, b, 0, 0) for a, b in zip(current[name], native[name])]
                flat_checks = [q for value in report["repeat_native"].values()
                               for q in (value if isinstance(value, list) else [value])]
                report["passed"] = (all(q["status"] == "PASS" for q in flat_checks)
                                    and all(q["status"] == "PASS" for q in report["post_profile_guards"].values()))
                report["device_execution"] = True
                save("native_eager_complete")
                if not report["passed"]:
                    raise AssertionError("Native 直接调用复核失败")
                return

            from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

            pypto.torch.init(
                device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer",
                **ring_sizing_kwargs(),
                **({"enable_chip_swimlane": 1, "enable_dep_gen": True,
                    "output_dir": str(args.output / "dfx")} if args.windows else {}),
            )
            adapter = importlib.import_module(PACKAGE + ".native_adapter")
            groups = {name: (fixture["metadata"][group["prefix"]], tuple(group["views"]))
                      for name, group in fixture["groups"].items()}
            if args.compressor_probe_only:
                from compressor_probe import hca_compressor_test

                # 仅复用描述符绑定，不调用整层算子；输入是实际 Native RMS 输出。
                descriptors = adapter.NativeHCACall(
                    adapter.HCAOperators.register(config=options), adapter.prepare_weights(layer.self_attn, layer),
                    fixture["hidden"], fixture["positions"], groups,
                    layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                    compact_metadata=fixture["compact"]["compressed"], output=output,
                ).args
                probe = pypto.torch.register(hca_compressor_test, "hca_tnd_probe::compressor", config=options)
                restore(fixture)
                keys = ("cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w", "state", "state_table",
                        "state_slots", "positions", "query_bounds", "seq_lens", "cmp_cos", "cmp_sin",
                        "cmp_slots", "cmp_cache")
                probe(native_intermediates["normalized"], *(descriptors[key] for key in keys))
                torch.npu.synchronize()
                actual = collect()
                report["device_execution"] = True
                report["compressor_native"] = {
                    name: [compare_tensor(a, b, 1e-5, 1e-5) for a, b in zip(actual[name], native[name])]
                    for name in ("state", "compressed")
                }
                report["pto_guards"] = guard_checks(fixture)
                report["passed"] = (all(check["status"] == "PASS" for group in report["compressor_native"].values()
                                        for check in group)
                                    and all(check["status"] == "PASS" for check in report["pto_guards"].values()))
                save("compressor_probe_complete")
                if not report["passed"]:
                    raise AssertionError("固定 Native RMS 输入的 C128 数值或写保护失败")
                return
            legacy = None
            reference_call = None
            reference_config = None
            if args.compare_legacy or control_package:
                if args.compare_legacy and (queries != [6] * args.batch or args.token_capacity or args.request_capacity):
                    raise ValueError("旧 HCA 对照只能在真实 S6、无 padding 的输入运行")
                reference_package = control_package or "vllm_ascend.ops.pypto.deepseek_v4_flash_hca"
                legacy_adapter = importlib.import_module(reference_package + ".native_adapter")
                reference_module = importlib.import_module(reference_package + (".entry" if control_package else ".decode_hca"))
                decode_hca_tp1_layer_test = reference_module.decode_hca_tp1_layer_test

                legacy_op = pypto.torch.register(
                    decode_hca_tp1_layer_test, "dsv4_hca_legacy_diagnostic::attention", config=options,
                )
                legacy_call = legacy_adapter.NativeHCACall(
                    legacy_adapter.HCAOperators(legacy_op), legacy_adapter.prepare_weights(layer.self_attn, layer),
                    fixture["hidden"], fixture["positions"], groups,
                    layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                    compact_metadata=fixture["compact"]["compressed"], output=output,
                )
                restore(fixture)
                before_reference = set((args.output / "build").glob("*/kernel_config.py"))
                legacy_call()
                torch.npu.synchronize()
                reference_configs = set((args.output / "build").glob("*/kernel_config.py")) - before_reference
                if len(reference_configs) != 1:
                    raise RuntimeError("控制组需要唯一 kernel_config")
                reference_config = reference_configs.pop()
                reference_call = legacy_call
                legacy = collect()
                report["reference_guards"] = guard_checks(fixture)
                save("numerical_reference_ready")
            call = adapter.NativeHCACall(
                adapter.HCAOperators.register(config=options), adapter.prepare_weights(layer.self_attn, layer),
                fixture["hidden"], fixture["positions"], groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
                compact_metadata=fixture["compact"]["compressed"], output=output,
            )
            if args.metadata_sequence:
                from metadata_sequence import validate

                def save_metadata(label, value):
                    report["metadata_validation"] = value
                    save("metadata_" + label)

                report["metadata_validation"] = validate(
                    adapter=adapter, operators=call.operators, weights=adapter.prepare_weights(layer.self_attn, layer),
                    pinned=call, config=config, layer=layer, fixture=fixture, output=output,
                    history=args.history, save=save_metadata,
                )
                report["device_execution"] = True
                report["passed"] = report["metadata_validation"]["passed"]
                save("metadata_sequence_complete")
                if not report["passed"]:
                    raise AssertionError("同址 TND metadata 直接调用验证失败")
                return
            restore(fixture)
            output.fill_(float("nan"))
            previous_configs = set((args.output / "build").glob("*/kernel_config.py"))
            call()
            torch.npu.synchronize()
            eager = collect()
            report["device_execution"] = True
            report["pto_guards"] = guard_checks(fixture)
            actual = fixture["actual_tokens"]
            report["pto_native"] = {"output": compare_tensor(eager["output"][:actual], native["output"][:actual], 0.01, 0.01)}
            for name in fixture["groups"]:
                report["pto_native"][name] = [compare_tensor(a, b, 1e-5, 1e-5)
                                               for a, b in zip(eager[name], native[name])]
            if legacy is not None:
                report["pto_reference"] = {"output": compare_tensor(eager["output"], legacy["output"], 0, 0)}
                report["reference_native"] = {"output": compare_tensor(legacy["output"], native["output"], 0.01, 0.01)}
                for name in fixture["groups"]:
                    report["pto_reference"][name] = [compare_tensor(a, b, 0, 0) for a, b in zip(eager[name], legacy[name])]
                    report["reference_native"][name] = [compare_tensor(a, b, 1e-5, 1e-5) for a, b in zip(legacy[name], native[name])]
            save("eager_compared")

            restore(fixture)
            output.fill_(float("nan"))
            call()
            torch.npu.synchronize()
            replay = collect()
            report["repeat_eager"] = {"output": compare_tensor(replay["output"], eager["output"], 0, 0)}
            for name in fixture["groups"]:
                report["repeat_eager"][name] = [compare_tensor(a, b, 0, 0) for a, b in zip(replay[name], eager[name])]
            report["repeat_guards"] = guard_checks(fixture)
            report["nonfinite_output"] = int((~torch.isfinite(replay["output"][:actual])).sum())
            save("repeat_compared")

            # 精度报告中按既有 compare_tensor 字段判定；绝不把成功执行当作精度通过。
            def all_passed(value):
                if isinstance(value, list):
                    return all(all_passed(v) for v in value)
                if "status" in value:
                    return value["status"] == "PASS"
                return all(all_passed(v) for v in value.values())

            report["passed"] = (not report["nonfinite_output"]
                                and all_passed(report["pto_native"])
                                and all_passed(report["repeat_eager"])
                                and all_passed(report["native_guards"])
                                and all_passed(report["pto_guards"])
                                and all_passed(report["repeat_guards"]))
            if args.precision_probe:
                if not all_passed(report["pto_guards"]) or report["nonfinite_output"]:
                    raise AssertionError("原入口存在越界或非有限值，不继续拼接诊断")
                from precision_probe import run_probe

                report["precision_probe"] = run_probe(
                    package=PACKAGE, options=options, call=call, fixture=fixture, layer=layer, config=config,
                    output=output, baseline=eager, native=native, native_tensors=native_intermediates,
                    restore=restore, collect=collect, output_dir=args.output,
                )
                report["post_probe_guards"] = guard_checks(fixture)
                save("precision_diagnosed")
                if (not report["precision_probe"]["passed_capture_equivalence"]
                        or not all_passed(report["post_probe_guards"])):
                    raise AssertionError("诊断采集改变生产结果或破坏保护区")
                return
            if args.service_probe:
                if not all_passed(report["pto_guards"]) or report["nonfinite_output"]:
                    raise AssertionError("原入口存在越界或非有限值，不继续服务验证")
                from service_probe import validate

                report["service_validation"] = validate(
                    package=PACKAGE, layer=layer, call=call, config=config, fixture=fixture,
                    output=output, baseline=eager, restore=restore, collect=collect,
                )
                report["post_service_guards"] = guard_checks(fixture)
                save("service_validated")
                if (not report["service_validation"]["passed"]
                        or not all_passed(report["post_service_guards"])):
                    raise AssertionError("服务接管、KV钩子或状态比较失败")
                return
            diagnostic_allowed = (
                args.diagnostic_perf and legacy is not None
                and all_passed(report["pto_reference"]) and all_passed(report["reference_guards"])
                and report["pto_native"]["output"]["status"] == "PASS"
                and all_passed(report["repeat_eager"]) and all_passed(report["pto_guards"])
                and all_passed(report["repeat_guards"]) and not report["nonfinite_output"]
            )
            if args.native_output_math:
                state_exact = all(all_passed(checks) for name, checks in report["pto_reference"].items()
                                  if name != "output")
                report["output_math_transition"] = {
                    "from": "八组各自量化、FP32部分和反量化",
                    "to": "Native O_A BF16边界、整token量化、完整K8192 INT32点积",
                    "same_input_evidence": "native_output_h8192_b32_v4/precision_probe.json",
                    "cache_state_exact_to_control": state_exact,
                    "output_bitwise_equal_to_control": all_passed(report["pto_reference"]["output"]),
                    "native_output_with_existing_tolerance": all_passed(report["pto_native"]["output"]),
                }
                diagnostic_allowed = (
                    state_exact and all_passed(report["pto_native"]["output"])
                    and all_passed(report["reference_guards"]) and all_passed(report["pto_guards"])
                    and all_passed(report["repeat_eager"]) and all_passed(report["repeat_guards"])
                    and not report["nonfinite_output"]
                )
            report["preserves_reference"] = all_passed(report["pto_reference"]) if legacy is not None else None
            report["diagnostic_performance_only"] = bool(diagnostic_allowed and not report["passed"])
            if not report["passed"] and not diagnostic_allowed:
                raise AssertionError("输出、cache/state、重复直接调用或写保护检查失败，见 result.json")
            if args.windows:
                from csa_mpmd_tnd_20261021.run_case import export_l1

                configs = list(set((args.output / "build").glob("*/kernel_config.py")) - previous_configs)
                if len(configs) != 1:
                    raise RuntimeError(f"要求唯一整层 kernel_config，得到 {configs}")
                entries = {"candidate": call}
                kernel_configs = {"candidate": configs[0]}
                if reference_call is not None:
                    entries = {"control": reference_call, **entries}
                    kernel_configs["control"] = reference_config
                for warmup in range(3):
                    order = list(entries) if warmup % 2 == 0 else list(reversed(entries))
                    for side in order:
                        restore(fixture)
                        entries[side]()
                        torch.npu.synchronize()
                by_side = {side: [] for side in entries}
                window_index = 0
                save("collecting_l1")
                for index in range(args.windows):
                    order = list(entries) if index % 2 == 0 else list(reversed(entries))
                    for side in order:
                        restore(fixture)
                        torch.npu.synchronize()
                        pypto.torch.begin_dfx()
                        try:
                            entries[side]()
                        finally:
                            pypto.torch.end_dfx()
                        raw = args.output / "dfx"
                        if window_index:
                            raw /= f"window_{window_index}"
                        by_side[side].append(export_l1(
                            raw, args.output / f"l1_{index:02d}_{side}", "mpmd_hca_" + side, kernel_configs[side],
                        ))
                        window_index += 1
                report["paired_l1"] = {}
                for side, windows in by_side.items():
                    samples = [window["incore_span_us"] for window in windows]
                    report["paired_l1"][side] = {
                        "windows": windows, "samples_us": samples, "median_us": statistics.median(samples),
                        "min_us": min(samples), "max_us": max(samples), "mean_us": statistics.mean(samples),
                    }
                report["l1"] = report["paired_l1"]["candidate"]
                if reference_call is not None:
                    report["candidate_over_control"] = (report["l1"]["median_us"]
                                                        / report["paired_l1"]["control"]["median_us"])
                restore(fixture)
                call()
                torch.npu.synchronize()
                report["post_l1_guards"] = guard_checks(fixture)
                after = collect()
                post_checks = {"output": compare_tensor(after["output"], eager["output"], 0, 0)}
                for name in fixture["groups"]:
                    post_checks[name] = [compare_tensor(a, b, 0, 0) for a, b in zip(after[name], eager[name])]
                report["post_l1_eager"] = post_checks
                if not all_passed(post_checks) or not all_passed(report["post_l1_guards"]):
                    report["passed"] = False
                    raise AssertionError("L1 采样后状态或保护区检查失败")
            if args.profile or args.host_profile:
                from profile_eager import profile_call

                def reset_profile():
                    restore(fixture)
                    output.fill_(float("nan"))

                pto_profile_call = call
                if args.profile_service:
                    from service_probe import validate
                    service = importlib.import_module(PACKAGE + ".service")

                    # 先证明确实进入PTO和保持KV生命周期；计时阶段不保留计数包装。
                    report["profile_service_validation"] = validate(
                        package=PACKAGE, layer=layer, call=call, config=config, fixture=fixture,
                        output=output, baseline=eager, restore=restore, collect=collect,
                    )
                    if not report["profile_service_validation"]["passed"]:
                        raise AssertionError("真实服务验证未通过，不继续计时")
                    service.bind_layer(layer, call.operators, max_num_seqs=64, max_num_batched_tokens=384)

                    def pto_profile_call():
                        # 每次都是一个新step，不能把compact metadata跨step缓存掉。
                        with set_ascend_forward_context(
                            fixture["metadata"], config, num_tokens=fixture["tokens"],
                            num_actual_tokens=fixture["actual_tokens"],
                        ):
                            torch.ops.vllm.dsv4_hca_forward(
                                fixture["hidden"], fixture["positions"], output,
                                layer.self_attn.dsa_attn.prefix,
                            )

                report["outer_profile_scope"] = (
                    "两侧每次新forward context；PTO真实custom op含eligibility、KV钩子、compact producer和描述符绑定"
                    if args.profile_service else "Native完整直接调用；PTO预绑定算子，不含每步服务绑定"
                )
                if args.host_profile:
                    from host_probe import profile_host

                    host_calls = {"native": native_call, "pto_prebound": call, "pto_service": pto_profile_call}
                    if args.prepared_host_probe:
                        from host_probe import prepared_operator

                        entry = importlib.import_module(PACKAGE + ".entry")
                        prepared = prepared_operator(entry.decode_hca_tp1_layer_test, call.operators.attention._schema)
                        prepared_runtime = service.HCATNDServiceRuntime(
                            layer.self_attn, adapter.HCAOperators(prepared), 64, layer, 384,
                        )

                        def prepared_call():
                            prepared(*call.core_args)

                        def prepared_service_call():
                            wrapper = layer.self_attn.dsa_attn
                            previous = wrapper._pto_hca_runtime
                            wrapper._pto_hca_runtime = prepared_runtime
                            try:
                                pto_profile_call()
                            finally:
                                wrapper._pto_hca_runtime = previous

                        host_calls.update(prepared=prepared_call, prepared_service=prepared_service_call)
                        prepared_checks = {}
                        for name, run in (("prepared", prepared_call), ("prepared_service", prepared_service_call)):
                            reset_profile()
                            run()
                            torch.npu.synchronize()
                            actual = collect()
                            checks = {"output": compare_tensor(actual["output"], eager["output"], 0, 0),
                                      "guards": guard_checks(fixture)}
                            for group in fixture["groups"]:
                                checks[group] = [compare_tensor(a, b, 0, 0) for a, b in zip(actual[group], eager[group])]
                            prepared_checks[name] = checks
                            if not all_passed(checks):
                                raise AssertionError("预绑定SDK调用改变了输出或状态")
                        report["prepared_probe_checks"] = prepared_checks
                        report["prepared_probe_scope"] = "测试期间源码/constexpr/config冻结；仅跳过JIT分析，SDK参数/别名/设备/生命周期校验均保留"
                        try:
                            prepared(*call.core_args[:-1])
                        except (ValueError, TypeError) as exc:
                            report["prepared_rejected_missing_argument"] = str(exc)
                        else:
                            raise AssertionError("预绑定SDK入口未拒绝缺失参数")
                    report["host_profile"] = profile_host(
                        host_calls,
                        reset_profile, output_dir=args.output,
                    )
                    after = collect()
                    checks = {"output": compare_tensor(after["output"], eager["output"], 0, 0),
                              "guards": guard_checks(fixture)}
                    for name in fixture["groups"]:
                        checks[name] = [compare_tensor(a, b, 0, 0) for a, b in zip(after[name], eager[name])]
                    report["post_host_profile_checks"] = checks
                    save("host_profiled")
                    if not all_passed(checks):
                        raise AssertionError("Host诊断后结果或保护区改变")
                    return
                report["outer_profiles"] = {}
                report["post_profile_checks"] = {}
                for side, run in (("native", native_call), ("pto", pto_profile_call)):
                    report["outer_profiles"][side] = profile_call(
                        run, reset_profile, output_dir=args.output / f"profile_{side}",
                        side=side, samples=args.windows or 10,
                    )
                    after = collect()
                    expected = native if side == "native" else eager
                    checks = {"output": compare_tensor(after["output"], expected["output"], 0, 0),
                              "guards": guard_checks(fixture)}
                    for name in fixture["groups"]:
                        checks[name] = [compare_tensor(a, b, 0, 0) for a, b in zip(after[name], expected[name])]
                    report["post_profile_checks"][side] = checks
                    if not all_passed(checks):
                        raise AssertionError(f"{side}外部采样后的结果或保护区改变")
            save("completed")
    except Exception:
        report["traceback"] = traceback.format_exc()
        save("failed")
        raise


if __name__ == "__main__":
    main()
