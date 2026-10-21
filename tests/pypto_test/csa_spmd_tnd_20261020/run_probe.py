# SPDX-License-Identifier: Apache-2.0
"""TND首轮设备诊断：真实Native半层与新SPMD包的混长输入对照。

复用正式权重、Native builder、页布局和保护区；直接绑定Native device输入。
Dspark上限保持5，仅本轮实际query长度为命令行值。Native只作功能/精度参照，
不采其性能，也不替代七档S6的原动态PTO性能基线。零容差报告如实保留差异；
shape/有限值/写保护与eager/graph稳定性和Native精度分别记账，不混称完整通过。
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


def workspace_padding_checks(entry):
    """Check private FIFO padding outside timing; never read metadata on the hot path."""
    transfer = entry.buffers["indexer_transfer"]
    columns = entry.policy.score_tile
    if transfer.ndim != 2 or transfer.shape[1] < columns:
        return {"indexer_transfer_shape": False}
    # Validate on CPU so this check does not add a temporary NPU allocation.
    checks = {}
    if transfer.shape[1] > columns:
        checks["indexer_transfer_column_padding"] = bool((transfer.cpu()[:, columns:] == 37).all())
    if getattr(entry.policy, "oa_small_supply", False):
        capacity = entry.buffers["positions"].shape[0]
        checks["oa_unwritten_rows"] = bool((entry.buffers["o_r"].cpu()[capacity:] == 37).all())
    if entry.policy.hc_consumer_reduce:
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.runtime_abi import PHASE_COUNT, PHASE_RAW

        checks["hc_unused_raw_untouched"] = bool((entry.buffers["raw"].cpu() == 37).all())
        signal = entry.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)
        checks["hc_raw_phase_unpublished"] = bool((signal[PHASE_RAW] == 0).all())
    return checks


def scale_writer_plan_diagnostic(entry):
    """Read the completed plan for diagnostics, outside the measured sequence."""
    requests = entry.buffers["idx_lengths"].shape[0]
    tokens = entry.buffers["positions"].shape[0]
    group_base = (requests + 16) // 16 * 16
    scale_base = (group_base + 4 * tokens + 15) // 16 * 16
    plan = entry.buffers["group_offsets"].cpu()
    heads = plan[scale_base : scale_base + 48].tolist()
    if len(heads) != 48:
        raise AssertionError("Indexer scale plan does not contain all 48 writer heads")
    result = {
        "scope": "Post-eager diagnostic only; not a production metadata readback",
        "heads": heads,
        "active_owners": [owner for owner, head in enumerate(heads) if head >= 0],
    }
    legacy_words = scale_base + 48 + 4 * tokens + (2 * requests if entry.policy.scale_page_coalescing else 0)
    if plan.numel() == legacy_words + 64:
        count = int(plan[legacy_words])
        if not 0 <= count <= 48:
            raise AssertionError(f"Active scale-writer count outside [0,48]: {count}")
        owners = plan[legacy_words + 1 : legacy_words + 1 + count].tolist()
        if owners != result["active_owners"]:
            raise AssertionError("Active scale-writer list does not match the current linked-list heads")
        if not bool((plan[legacy_words + 49 :] == 37).all()):
            raise AssertionError("Reserved scale-writer workspace padding was overwritten")
        result["compact_owner_list"] = {"passed": True, "count": count, "owners": owners}
    elif plan.numel() != legacy_words:
        raise AssertionError("Unknown scale-plan workspace extent")
    return result


def bind_native_inputs(adapter, layer, fixture, output, torch):
    """首轮设备探针直接使用新TND生产绑定函数。"""
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.native_adapter import bind_native_inputs as bind

    groups = {name: (fixture["metadata"][group["prefix"]], group["views"]) for name, group in fixture["groups"].items()}
    weights = adapter.prepare_weights(layer.self_attn, groups["indexer"][0].hadamard, layer)
    return bind(
        weights,
        fixture["hidden"],
        fixture["positions"],
        groups,
        layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name,
        compact_metadata=fixture["compact"],
        output=output,
        topk=torch.empty((fixture["tokens"], 512), dtype=torch.int32, device=output.device),
        topk_scores=torch.empty((fixture["tokens"], 512), dtype=torch.float32, device=output.device),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--query-tokens", type=int, choices=tuple(range(1, 7)), default=5)
    parser.add_argument("--causal-extension-reference", action="store_true")
    parser.add_argument("--query-lengths", help="逗号分隔的各请求query长度")
    parser.add_argument("--history-lengths", help="逗号分隔的各请求历史长度")
    parser.add_argument("--spmd-source", type=Path)
    parser.add_argument("--baseline-spmd-source", type=Path, help="原单SPMD源码；仅定位TND回归，不充当Native精度golden")
    parser.add_argument("--service-route", action="store_true")
    parser.add_argument("--service-metadata-replay", action="store_true")
    parser.add_argument("--service-timing", action="store_true")
    parser.add_argument("--service-timing-only", action="store_true")
    parser.add_argument("--l1-windows", type=int, default=0)
    parser.add_argument("--l1-shared-workspace", action="store_true")
    parser.add_argument("--diagnose-stages", action="store_true")
    parser.add_argument(
        "--constant-indexer-history", action="store_true", help="历史Indexer key置零，验证同分TopK；不作为性能样本"
    )
    parser.add_argument("--prepare-only", action="store_true", help="只导出CPU编译所需的容量描述符，不运行候选")
    parser.add_argument("--token-capacity", type=int)
    parser.add_argument("--request-capacity", type=int)
    parser.add_argument("--table-history", type=int)
    parser.add_argument("--metadata-matrix", action="store_true")
    parser.add_argument("--b1-metadata-matrix", action="store_true")
    parser.add_argument("--continuous-only", action="store_true")
    parser.add_argument("--raw-wait", action="store_true", help="不启用测试工具的等待退避C++改写")
    parser.add_argument("--framework-graph", action="store_true")
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    if "TASK_DEVICE" not in os.environ:
        raise RuntimeError("必须通过task-submit --device auto提交")
    if args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("设备参数必须与队列分配一致")
    queries = (
        [int(x) for x in args.query_lengths.split(",")] if args.query_lengths else [args.query_tokens] * args.batch
    )
    histories = (
        [int(x) for x in args.history_lengths.split(",")] if args.history_lengths else [args.history] * args.batch
    )
    if len(queries) != args.batch or any(not 1 <= q <= 6 for q in queries):
        raise ValueError("query-lengths须与batch一致且在1..6")
    if len(histories) != args.batch or any(h < 0 for h in histories):
        raise ValueError("history-lengths须与batch一致且非负")
    args.query_tokens = max(queries)
    args.history = max(histories)
    if args.causal_extension_reference:
        raise ValueError("本轮混长以Native TND为参照，不启用S6因果扩展")
    if args.metadata_matrix and (args.service_route or not args.baseline_spmd_source):
        raise ValueError("metadata矩阵需原单SPMD对照，并与服务诊断使用独立fixture任务")
    if args.continuous_only and not args.metadata_matrix:
        raise ValueError("continuous-only须与metadata-matrix配合")
    if args.framework_graph and not args.service_route:
        raise ValueError("framework-graph须与service-route配合")
    if args.service_metadata_replay and not (args.framework_graph and args.baseline_spmd_source):
        raise ValueError("service-metadata-replay需要framework-graph及baseline-spmd-source")
    if args.service_timing and not args.framework_graph:
        raise ValueError("service-timing需要framework-graph")
    if args.service_timing_only and not (args.service_timing and args.framework_graph):
        raise ValueError("service-timing-only需要service-timing及framework-graph")
    if args.l1_windows and (args.l1_windows < 3 or args.service_route or not args.baseline_spmd_source):
        raise ValueError("TND L1至少三个窗口，需要原SPMD对照，服务封装另测")
    if args.l1_shared_workspace and not args.l1_windows:
        raise ValueError("l1-shared-workspace是L1专用归因诊断")
    if args.b1_metadata_matrix and (not args.metadata_matrix or args.continuous_only):
        raise ValueError("b1-metadata-matrix须与metadata-matrix配合，不与continuous-only混用")
    if args.constant_indexer_history and (args.metadata_matrix or args.service_route or args.l1_windows):
        raise ValueError("同分场景只做独立正确性验证，不混用metadata矩阵、服务或性能采样")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        stage="initializing",
        passed=False,
        functional_passed=False,
        precision_exact=False,
        device_execution=False,
        performance_measured=False,
        batch=args.batch,
        history=args.history,
        query_tokens=args.query_tokens,
        query_lengths=queries,
        history_lengths=histories,
        dspark_spec_tokens=5,
        reference="真实Native TND attention半层；原分task固定S6不作为混长golden",
        limitations=["初次混长数值诊断，不含性能", "尚未覆盖连续状态轨迹", "服务路由另行验证"],
        acceptance_scope="2026-10-21用户调整：原有Native差异单列，验收TND语义与无新增回归",
        wait_backoff=not args.raw_wait,
        constant_indexer_history=args.constant_indexer_history,
        rounds=[],
    )

    def save():
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("stage", "passed", "device_execution")}), flush=True)

    try:
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import torch
        import torch_npu
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
        from dsv4_csa_validation import compare_tensor, compare_topk
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_csa import _native_attention_half
        from vllm_ascend.utils import enable_custom_op

        current_platform.pre_register_and_update()
        device_id = int(os.environ["TASK_DEVICE"])
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        if not enable_custom_op():
            raise RuntimeError("Native自定义算子包尚未注册")
        config = EngineArgs(
            model="/data/model/DeepSeek-V4-Flash-0731-w8a8",
            tokenizer_mode="deepseek_v4",
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="ascend",
            tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128},
            max_model_len=min(1048576, max(16384, args.history + 128)),
            max_num_seqs=40,
            max_num_batched_tokens=256,
            block_size=32,
            enable_prefix_caching=False,
            enforce_eager=True,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            additional_config={"weight_nz_mode": 2, "enable_kv_nz": False, "enable_dsa_cp": False},
        ).create_engine_config()
        import pypto.torch
        from compile_diagnostics import compilation_probe
        from pypto import passes
        from pypto.runtime import CompileOptions

        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa import native_adapter as adapter

        if args.spmd_source:
            # 只在本测试进程导入明确版本的正式动态源码，不切换共享工作区。
            package_name = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd"
            if any(name == package_name or name.startswith(package_name + ".") for name in sys.modules):
                raise RuntimeError("SPMD包已导入，不能在进程中途换源码")
            source = args.spmd_source.resolve()
            spec = importlib.util.spec_from_file_location(
                package_name, source / "__init__.py", submodule_search_locations=[str(source)]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            spec.loader.exec_module(module)
            report["isolated_spmd_source"] = str(source)
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.binding import PreparedDynamicCSA
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd.runtime_abi import PHASE_COUNT
        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

        package = Path(importlib.import_module(PreparedDynamicCSA.__module__).__file__).parent
        (args.output / "source").mkdir()
        for path in package.glob("*.py"):
            (args.output / "source" / path.name).write_text(path.read_text())
        (args.output / Path(__file__).name).write_text(Path(__file__).read_text())
        device = torch.device(f"npu:{device_id}")
        with native_session(config, device_id), torch.inference_mode():
            layer, report["weights"] = make_layer(config, Path(config.model_config.model), device, 4)
            fixture = make_fixture(
                config,
                layer.self_attn,
                args.batch,
                args.history,
                1024,
                device,
                query_tokens=args.query_tokens,
                query_lengths=queries,
                history_lengths=histories,
                token_capacity=args.token_capacity,
                request_capacity=args.request_capacity,
                table_history=args.table_history,
            )
            report["token_capacity"] = fixture["tokens"]
            report["actual_tokens"] = fixture["actual_tokens"]
            report["request_capacity"] = fixture["request_capacity"]
            report["stage"] = "fixture_ready"
            save()
            scale = fixture["groups"]["indexer"]["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            if args.constant_indexer_history:
                # Change only fixture data. New current-token keys still come
                # from the complete CSA; all cache updates and guards remain.
                fixture["groups"]["indexer"]["views"][0].zero_()
                scale.fill_(0.00390625)
            fixture["groups"]["indexer"]["initial"] = fixture["groups"]["indexer"]["allocation"].cpu()
            output = torch.empty_like(fixture["hidden"])
            root = bind_native_inputs(adapter, layer, fixture, output, torch)
            reference_fixture, reference_output = fixture, output
            if fixture["actual_tokens"] != fixture["tokens"] or fixture["request_capacity"] != args.batch:
                # Native eager SparseAttn consumes only effective query rows.
                # Preserve the same full physical cache allocation, but do not
                # pass padded Q endpoints beyond that Native query tensor.
                reference_fixture = make_fixture(
                    config,
                    layer.self_attn,
                    args.batch,
                    args.history,
                    1024,
                    device,
                    query_lengths=queries,
                    history_lengths=histories,
                    table_history=args.table_history,
                    physical_request_capacity=fixture["request_capacity"],
                )
                reference_fixture["hidden"].copy_(fixture["hidden"][: fixture["actual_tokens"]])
                for name, group in reference_fixture["groups"].items():
                    initial = fixture["groups"][name]["initial"]
                    if group["allocation"].shape != initial.shape:
                        raise ValueError(f"Native参考物理cache容量不一致：{name}")
                    group["initial"] = initial
                    group["allocation"].copy_(initial)
                reference_output = torch.empty_like(reference_fixture["hidden"])
            report["native_reference_tokens"] = reference_fixture["tokens"]
            allocations = {}

            def allocate(name, shape, dtype, device):
                count = 1
                for n in shape:
                    count *= n
                guard = 512 // torch.empty((), dtype=dtype).element_size()
                storage = torch.full((count + guard * 2,), 37, dtype=dtype, device=device)
                allocations[name] = storage, guard
                return storage[guard:-guard].view(shape)

            candidate = PreparedDynamicCSA(
                root,
                max_query_tokens=args.query_tokens,
                max_seq_len=args.history + args.query_tokens,
                allocator=allocate,
            )
            report["policy"] = candidate.policy.constexprs()
            report["tensor_count"] = len(candidate.buffers)
            (args.output / "buffer_descriptors.json").write_text(
                json.dumps(
                    {
                        "policy": candidate.policy.constexprs(),
                        "buffers": {
                            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                            for name, value in candidate.buffers.items()
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
            if args.prepare_only:
                report.update(stage="descriptors_only", passed=True, device_execution=False)
                save()
                return 0
            if args.service_route:
                from service_probe import prepare

                service_runtime = prepare(layer)
            else:
                pypto.torch.init(
                    device=device_id,
                    platform="a2a3",
                    runtime="tensormap_and_ringbuffer",
                    enable_chip_swimlane=1 if args.l1_windows else 0,
                    enable_dep_gen=bool(args.l1_windows),
                    output_dir=str(args.output / "dfx") if args.l1_windows else None,
                    **ring_sizing_kwargs(),
                )
            options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
            if args.service_timing_only:
                from service_probe import validate

                for name, group in fixture["groups"].items():
                    group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
                restore(fixture)
                with compilation_probe(args.output / "compile_diagnostics", wait_backoff=not args.raw_wait):
                    candidate(config=options)
                torch.npu.synchronize()
                eager = collect_state(fixture, output, root["idx_topk"])
                report["device_execution"] = True
                report["stage"] = "service_cost_only"
                save()
                report["service_route"] = validate(
                    service_runtime,
                    layer,
                    config,
                    fixture,
                    output,
                    eager,
                    args.output,
                    wait_backoff=not args.raw_wait,
                    framework_graph=True,
                    timing=True,
                    timing_only=True,
                )
                report.update(
                    stage="complete",
                    passed=report["service_route"]["passed"],
                    performance_measured=True,
                    limitations=["仅补测服务成本；Native、原SPMD与变metadata图验收复用v13，不重复计数"],
                )
                return 0 if report["passed"] else 1
            captured = {}
            original_qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer
            native_prefix = {}
            original_hc_pre = layer.hc_pre
            norm_hook = None

            def capture_hc_pre(*inputs, **kwargs):
                value = original_hc_pre(*inputs, **kwargs)
                for name, tensor in zip(("mixed", "post", "comb"), value):
                    native_prefix[name] = tensor.detach().cpu().clone()
                return value

            def capture_norm(module, inputs, value):
                native_prefix["normed"] = value.detach().cpu().clone()

            def capture_topk(*inputs, **kwargs):
                value = original_qli(*inputs, **kwargs)
                captured["topk"] = value[0].detach().clone()
                if args.diagnose_stages:
                    captured["indexer_inputs"] = {
                        name: kwargs[name].detach().cpu().clone()
                        for name in ("query", "weights", "query_dequant_scale", "block_table")
                    }
                return value

            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = capture_topk
            if args.diagnose_stages:
                layer.hc_pre = capture_hc_pre
                norm_hook = layer.input_layernorm.register_forward_hook(capture_norm)
            try:
                restore(reference_fixture)
                with set_ascend_forward_context(
                    reference_fixture["metadata"],
                    config,
                    num_tokens=reference_fixture["tokens"],
                    num_actual_tokens=reference_fixture["actual_tokens"],
                ):
                    _native_attention_half(
                        layer.self_attn.dsa_attn,
                        reference_fixture["hidden"],
                        reference_fixture["positions"],
                        reference_output,
                    )
                torch.npu.synchronize()
                reference = collect_state(reference_fixture, reference_output, captured["topk"])
                report["native_guards"] = guard_checks(reference_fixture)
                assert all(v["status"] == "PASS" for v in report["native_guards"].values())
            finally:
                torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
                layer.hc_pre = original_hc_pre
                if norm_hook is not None:
                    norm_hook.remove()
                for name, group in fixture["groups"].items():
                    group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
            report["stage"] = "native_reference_ready"
            save()
            restore(fixture)
            for name, value in candidate.allocations.items():
                if name not in ("signal", "ffts"):
                    value.fill_(37)
            with compilation_probe(args.output / "compile_diagnostics", wait_backoff=not args.raw_wait):
                candidate(config=options)
            torch.npu.synchronize()
            candidate_tables = set(Path(os.environ["PYPTO_PROG_BUILD_DIR"]).glob("decode_csa_spmd_*/kernel_config.py"))
            report["device_execution"] = True
            eager = collect_state(fixture, output, root["idx_topk"])
            eager_scores = root["idx_topk_scores"].cpu().clone()
            if args.diagnose_stages:
                stage_checks = {}
                candidate_prefix = {}
                for name, expected in native_prefix.items():
                    value = candidate.buffers[name][: expected.shape[0]].cpu().reshape(expected.shape)
                    candidate_prefix[name] = value
                    stage_checks[name] = compare_tensor(value, expected, 0, 0)
                for name, buffer in (
                    ("query", "idx_query_i8"),
                    ("weights", "head_weights"),
                    ("query_dequant_scale", "idx_query_scale"),
                ):
                    expected = captured["indexer_inputs"][name]
                    value = candidate.buffers[buffer].cpu().flatten()[: expected.numel()].reshape(expected.shape)
                    stage_checks["indexer." + name] = compare_tensor(value, expected, 0, 0)
                report["native_stage_diagnostic"] = stage_checks
                torch.save(
                    {
                        "native_prefix": native_prefix,
                        "spmd_prefix": candidate_prefix,
                        "native_indexer_inputs": captured["indexer_inputs"],
                        "spmd_indexer_inputs": {
                            name: candidate.buffers[name].cpu()
                            for name in ("idx_query_i8", "head_weights", "idx_query_scale")
                        },
                    },
                    args.output / "stage_diagnostic.pt",
                )
            report["native_comparison"] = {k: compare_tensor(v, reference[k], 0, 0) for k, v in eager.items()}
            report["native_frozen_output"] = compare_tensor(eager["x_out"], reference["x_out"], 0.01, 0.01)
            report["native_topk_selection"] = compare_topk(
                eager["idx_topk"],
                reference["idx_topk"],
                (fixture["positions"][: fixture["actual_tokens"]].cpu() + 1) // 4,
            )
            # 从逐请求整数区间独立枚举C4关闭点，核对主/Indexer的device前缀计划。
            offsets, closed = [], 0
            for query, history in zip(queries, histories):
                offsets.append(closed - history // 4 - 1)
                closed += sum((position + 1) % 4 == 0 for position in range(history, history + query))
            offsets += [0] * (fixture["request_capacity"] - len(queries))
            report["compact_geometry"] = {}
            for branch in ("main", "idx"):
                report["compact_geometry"][branch + ".offsets"] = compare_tensor(
                    candidate.buffers[branch + "_offsets"], torch.tensor(offsets, dtype=torch.int32), 0, 0
                )
                report["compact_geometry"][branch + ".count"] = compare_tensor(
                    candidate.buffers[branch + "_count"], torch.tensor([closed], dtype=torch.int32), 0, 0
                )
            report["indexer_scale_writer_plan"] = scale_writer_plan_diagnostic(candidate)
            report["eager_guards"] = guard_checks(fixture)
            addresses = {k: v.data_ptr() for k, v in candidate.buffers.items()}
            restore(fixture)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                candidate(config=options)
            for generation in range(2):
                restore(fixture)
                for name, value in candidate.allocations.items():
                    if name not in ("signal", "ffts"):
                        value.fill_(37)
                before = candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)[0, :, 0]
                graph.replay()
                torch.npu.synchronize()
                actual = collect_state(fixture, output, root["idx_topk"])
                checks = {k: compare_tensor(v, eager[k], 0, 0) for k, v in actual.items()}
                guards = guard_checks(fixture)
                scratch = {
                    k: bool((v[:g].cpu() == 37).all() and (v[-g:].cpu() == 37).all())
                    for k, (v, g) in allocations.items()
                }
                scratch.update(workspace_padding_checks(candidate))
                signal = candidate.buffers["signal"].cpu().view(PHASE_COUNT, 72, 128)
                protocol = bool((signal[0, :, 0] == before + 1).all() and (signal[:, :, 1:] == 0).all())
                same = addresses == {k: v.data_ptr() for k, v in candidate.buffers.items()}
                passed = (
                    protocol
                    and same
                    and all(scratch.values())
                    and all(v["status"] == "PASS" for v in (*checks.values(), *guards.values()))
                )
                report["rounds"].append(
                    dict(
                        replay=generation + 1,
                        passed=passed,
                        protocol=protocol,
                        same_addresses=same,
                        comparisons=checks,
                        guards=guards,
                        scratch=scratch,
                    )
                )
                assert passed, "实际query的eager/graph或保护区/代际失败"
            report["functional_passed"] = (
                all(v["status"] == "PASS" for v in report["eager_guards"].values())
                and all(v.get("nonfinite") == 0 for v in report["native_comparison"].values())
                and report["native_topk_selection"]["status"] != "FAIL"
            )
            report["precision_exact"] = all(v["status"] == "PASS" for v in report["native_comparison"].values())
            report["passed"] = False  # 无新增回归须有相同真实输入的原单SPMD证据。
            # 先保存真实SQ的数值证据；后续因果参照失败也不丢失已完成的比较。
            if not report["precision_exact"]:
                torch.save(
                    {
                        "native_output": reference["x_out"],
                        "spmd_output": eager["x_out"],
                        "native_topk": reference["idx_topk"],
                        "spmd_topk": eager["idx_topk"],
                    },
                    args.output / "numerical_diagnostic.pt",
                )
            baseline = None
            if args.baseline_spmd_source:
                from source_loader import load_binding

                report["stage"] = "original_spmd_attribution"
                save()
                baseline_package = load_binding(args.baseline_spmd_source, "tnd_original_baseline")
                baseline = baseline_package.prepared(
                    root,
                    max_query_tokens=args.query_tokens,
                    max_seq_len=args.history + args.query_tokens,
                    allocator=(lambda name, shape, dtype, device: allocate("baseline/" + name, shape, dtype, device))
                    if args.l1_windows
                    else None,
                )
                restore(fixture)
                for name, value in baseline.allocations.items():
                    if name not in ("signal", "ffts"):
                        value.fill_(37)
                with compilation_probe(args.output / "baseline_compile_diagnostics", wait_backoff=not args.raw_wait):
                    baseline(config=options)
                torch.npu.synchronize()
                baseline_tables = (
                    set(Path(os.environ["PYPTO_PROG_BUILD_DIR"]).glob("decode_csa_spmd_*/kernel_config.py"))
                    - candidate_tables
                )
                original = collect_state(fixture, output, root["idx_topk"])
                checks = {k: compare_tensor(eager[k], value, 0, 0) for k, value in original.items()}
                checks["idx_topk_scores"] = compare_tensor(eager_scores, root["idx_topk_scores"].cpu(), 0, 0)
                report["original_spmd_attribution"] = {
                    "source": str(baseline_package.source),
                    "scope": "同输入原单SPMD回归归属；不替代Native精度验收",
                    "candidate_exactly_matches": all(v["status"] == "PASS" for v in checks.values()),
                    "comparisons": checks,
                    "baseline_native_frozen_output": compare_tensor(original["x_out"], reference["x_out"], 0.01, 0.01),
                    "baseline_native_topk_selection": compare_topk(
                        original["idx_topk"],
                        reference["idx_topk"],
                        (fixture["positions"][: fixture["actual_tokens"]].cpu() + 1) // 4,
                    ),
                    "baseline_guards": guard_checks(fixture),
                }
                report["passed"] = (
                    report["functional_passed"]
                    and report["original_spmd_attribution"]["candidate_exactly_matches"]
                    and all(value["status"] == "PASS" for value in report["compact_geometry"].values())
                    and all(
                        value["status"] == "PASS"
                        for value in report["original_spmd_attribution"]["baseline_guards"].values()
                    )
                )
            if args.service_route:
                from service_probe import validate

                regression_passed = report["passed"]
                report["stage"] = "real_service_route"
                report["passed"] = False
                save()
                report["service_route"] = validate(
                    service_runtime,
                    layer,
                    config,
                    fixture,
                    output,
                    eager,
                    args.output,
                    wait_backoff=not args.raw_wait,
                    framework_graph=args.framework_graph,
                    metadata_replay=args.service_metadata_replay,
                    baseline=baseline,
                    options=options,
                    timing=args.service_timing,
                )
                report["passed"] = regression_passed and report["service_route"]["passed"]
            if args.l1_windows:
                from tnd_performance import profile

                if len(candidate_tables) != 1 or len(baseline_tables) != 1:
                    raise ValueError("配对L1必须能唯一归属两侧编译配置")
                regression_passed = report["passed"]
                report.update(stage="paired_tnd_l1", passed=False)
                save()
                report["l1"] = profile(
                    candidate=candidate,
                    baseline=baseline,
                    fixture=fixture,
                    output=output,
                    expected=eager,
                    scores=eager_scores,
                    options=options,
                    directory=args.output,
                    windows=args.l1_windows,
                    shared_workspace=args.l1_shared_workspace,
                    padding_checks=workspace_padding_checks,
                    configs={
                        "spmd_tnd": next(iter(candidate_tables)),
                        "spmd_original": next(iter(baseline_tables)),
                    },
                )
                report["performance_measured"] = True
                report["passed"] = regression_passed and report["l1"]["passed"]
            if args.metadata_matrix:
                from metadata_matrix import validate as validate_matrix

                # Same-input exact checks already precede L1. Run the metadata
                # matrix last: it changes captured root contents and does not
                # restore the initial S6 fixture. This also lets one compiled
                # pair cover S6 timing and the separate dynamic-state checks.
                regression_passed = report["passed"]
                report["stage"] = "tnd_metadata_and_continuous_state"
                report["passed"] = False
                save()
                report["metadata_matrix"] = validate_matrix(
                    config,
                    layer,
                    fixture,
                    candidate,
                    baseline,
                    allocations,
                    options,
                    args.output / "metadata_matrix.json",
                    args.history,
                    args.table_history,
                    continuous_only=args.continuous_only,
                    b1_only=args.b1_metadata_matrix,
                    plan_diagnostic=scale_writer_plan_diagnostic,
                )
                report["passed"] = (
                    regression_passed
                    and report["functional_passed"]
                    and report["original_spmd_attribution"]["candidate_exactly_matches"]
                    and report["metadata_matrix"]["passed"]
                )
            report["stage"] = "complete"
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        if getattr(exc, "span", None):
            report["error_span"] = exc.span
    finally:
        save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
