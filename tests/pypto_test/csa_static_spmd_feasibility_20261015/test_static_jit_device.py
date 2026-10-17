# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""完整 CSA 单 SPMD 的 native 直调验证；复用正式权重与冻结动态基线。"""

import argparse
import ast
import importlib
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import dsv4_csa_env

SOURCE = dsv4_csa_env.activate()

import torch
import torch_npu
import pypto.torch
from pypto import passes
from pypto.runtime import CompileOptions
from dsv4_csa_native_case import native_session
from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
from dsv4_csa_validation import compare_tensor, compare_topk
from offline_pd.run import decode_additional_config
from prepared_arguments import DTYPES, PreparedStaticArguments


def write_report(directory, report):
    temporary = directory / "result.tmp.json"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(directory / "result.json")
    print(json.dumps({"stage": report["stage"], "passed": report["passed"]}, ensure_ascii=False), flush=True)


def verify_isolation():
    import _task_interface
    import simpler
    from pypto import _torch_npu, pypto_core
    from pypto._kernel_abi import SIMPLER_KERNEL_REVISION

    pypto_root = Path(os.environ["CSA_STATIC_PYPTO_ROOT"]).resolve()
    simpler_root = Path(os.environ["HCA_SIMPLER_ROOT"]).resolve()
    for module in (pypto_core, _torch_npu):
        assert pypto_root in Path(module.__file__).resolve().parents
    for module in (simpler, _task_interface):
        assert simpler_root in Path(module.__file__).resolve().parents
    assert _torch_npu.simpler_revision == _task_interface.__build_commit__ == SIMPLER_KERNEL_REVISION
    return {"pypto_binding": pypto_core.__file__, "torch_adapter": _torch_npu.__file__,
            "simpler_binding": _task_interface.__file__, "simpler_revision": SIMPLER_KERNEL_REVISION,
            "torch": torch.__version__, "torch_npu": torch_npu.__version__}


def progress_expectation(layout, generation):
    """只检查实际需要发布的末次进度；未发布的独立尾任务不虚构额外信号。"""
    protocol = layout["protocol"]
    expected = [generation * protocol["generation_stride"]] * protocol["core_count"]
    for batches in layout["batches"].values():
        for batch in batches:
            for member in batch["members"]:
                if member["publish_completion"]:
                    core = member["core"]
                    expected[core] = max(expected[core], generation * protocol["generation_stride"]
                                         + member["dispatch_seq"])
    if generation > protocol.get("max_generation", 2**63 - 1):
        raise ValueError("代际超出当前信号协议的整数范围；必须先排空再重新初始化")
    dtype = DTYPES[layout["workspace"]["sections"]["progress"]["dtype"]]
    return torch.tensor(expected, dtype=dtype)


def exact_checks(actual, expected):
    return {name: compare_tensor(value, expected[name], 0, 0) for name, value in actual.items()}


def checks_passed(checks):
    return all(value["status"] == "PASS" for value in checks.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    parser.add_argument("--graph-replay", action="store_true")
    args = parser.parse_args()
    if args.device < 0 or "TASK_DEVICE" not in os.environ:
        raise RuntimeError("必须通过 task-submit --device auto 分配设备")
    args.output.mkdir(parents=True, exist_ok=True)
    match = re.fullmatch(r"h(\d+)_b(\d+)", args.case.name)
    if match is None:
        raise ValueError("case 应为已捕获的 h<history>_b<batch> 模板")
    history, batch = map(int, match.groups())
    tokens = batch * 6
    report = {"stage": "initializing", "passed": False, "device_execution": False,
              "full_csa_validated": False, "scope": "完整 attention 半层；正式权重、合成历史，无 MoE/EP16",
              "entry": "JITFunction.__call__ → native prepare/enqueue，无 torch.library 注册",
              "batch": batch, "query_tokens": 6, "history": history, "device": args.device,
              "case": str(args.case.resolve()), "source": str(SOURCE), "launches": []}
    original_qli = None
    try:
        report["environment"] = verify_isolation()
        # 与静态排程提取时完全同源；不能误用仍在演进的生产算子目录。
        import vllm_ascend.ops.pypto as operator_package

        frozen = Path(os.environ["HC_PRE_TEST_OPERATOR_SOURCE"]).resolve()
        operator_package.__path__ = [str(frozen)]
        report["frozen_operator_source"] = str(frozen)
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform

        current_platform.pre_register_and_update()
        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("Native 自定义算子不可用")
        additional = decode_additional_config(SimpleNamespace(
            graph_mode="full_decode_only", capture_sizes=[tokens], recompute_scheduler=False))
        additional.update(weight_nz_mode=2, enable_kv_nz=False, enable_dsa_cp=False,
                          pto_csa_ring_config={"heap_mb": [256, 128, 256, 32], "task_window": 4096})
        config = EngineArgs(
            model="/data/model/DeepSeek-V4-Flash-0731-w8a8", tokenizer_mode="deepseek_v4",
            trust_remote_code=True, dtype="bfloat16", quantization="ascend", tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128}, max_model_len=max(16384, history + 128),
            max_num_seqs=40, max_num_batched_tokens=256, block_size=32, enable_prefix_caching=False,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [tokens],
                                "cache_dir": str(args.output / "compile_cache")},
            additional_config=additional,
        ).create_engine_config()
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_csa import _native_attention_half
        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs, variant_package

        adapter = importlib.import_module(f"{variant_package()}.native_adapter")
        assert frozen in Path(adapter.__file__).resolve().parents
        with native_session(config, args.device), torch.inference_mode():
            device = torch.device(f"npu:{args.device}")
            layer, weights = make_layer(config, Path(config.model_config.model), device, 4)
            report["weights"] = weights
            report["stage"] = "formal_weights_loaded"
            write_report(args.output, report)
            fixture = make_fixture(config, layer.self_attn, batch, history, 1024, device)
            group = fixture["groups"]["indexer"]
            scale = group["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            group["initial"] = group["allocation"].cpu()
            output = torch.empty_like(fixture["hidden"])
            native_topk = {}
            original_qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer

            def record_qli(*inputs, **kwargs):
                value = original_qli(*inputs, **kwargs)
                native_topk["value"] = value[0]
                return value

            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = record_qli
            restore(fixture)
            with set_ascend_forward_context(fixture["metadata"], config, num_tokens=tokens,
                                           num_actual_tokens=tokens):
                _native_attention_half(layer.self_attn.dsa_attn, fixture["hidden"], fixture["positions"], output)
            torch.npu.synchronize()
            native_reference = collect_state(fixture, output, native_topk["value"])
            report["native_guards"] = guard_checks(fixture)
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
            original_qli = None
            report["stage"] = "native_reference_complete"
            write_report(args.output, report)

            pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer",
                             **ring_sizing_kwargs(additional))
            groups = {name: (fixture["metadata"][group["prefix"]], group["views"])
                      for name, group in fixture["groups"].items()}
            hadamard = groups["indexer"][0].hadamard
            pto_weights = adapter.prepare_weights(layer.self_attn, hadamard, layer)
            dynamic_entry = adapter.decode_csa_tp1_layer_test
            # 动态参考沿用已有 56 参数注册合同，排除源码签名中的 constexpr 默认值。
            # 132/133 参数静态入口仍直接调用 JIT，不经过 Torch dispatcher。
            call = adapter.NativeCSACall(
                adapter.CSAOperators.register(), pto_weights, fixture["hidden"], fixture["positions"], groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name, compact_metadata=fixture["compact"],
                buffers={"idx_topk_scores": torch.empty((tokens, 512), dtype=torch.float32, device=device),
                         "idx_topk": torch.empty((tokens, 512), dtype=torch.int32, device=device), "x_out": output})
            layout = json.loads((args.case / "layout.json").read_text())
            progress_expectation(layout, 5 if args.graph_replay else 3)
            prepared = PreparedStaticArguments(args.case, call.args,
                                               workspace_guard_bytes=layout["workspace"]["alignment"])
            prepared.require_core_capacity(256)
            assert all(value.is_contiguous() for value in prepared.core_args)
            report.update(native_inputs=len(prepared.native_args), tensor_arguments=len(prepared.core_args),
                          core_tensor_arguments=prepared.contract["leaf_tensor_count"],
                          derived_native_views=prepared.contract.get("derived_native_views", []),
                          workspace_bytes=prepared.workspace.numel())
            snapshot = args.output / "source.py"
            snapshot.write_text((args.case / "entry_jit.py").read_text())
            if layout["protocol"].get("signal_api") == "temporary_local_set_ge":
                adapter_snapshot = args.output / "local_signal_adapter.py"
                adapter_snapshot.write_text(Path(__file__).with_name("local_signal_adapter.py").read_text())
                report["signal_adapter_snapshot"] = str(adapter_snapshot)
            spec = importlib.util.spec_from_file_location("csa_static_full_device", snapshot)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            syntax = ast.parse(snapshot.read_text())
            entry_node = next(node for node in syntax.body
                              if isinstance(node, ast.FunctionDef) and node.name == "entry")
            return_names = [node.id for node in entry_node.body[-1].value.elts]
            compile_options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
            restore(fixture)
            dynamic_entry(*call.core_args, config=compile_options)
            torch.npu.synchronize()
            dynamic_reference = collect_state(fixture, output, call.args["idx_topk"])
            report["dynamic_guards"] = guard_checks(fixture)
            report["dynamic_native_exact"] = exact_checks(dynamic_reference, native_reference)
            report["stage"] = "dynamic_reference_complete"
            write_report(args.output, report)

            # 所有更新与直调均放在同一非默认流，验证 adapter 的当前流传递。
            # 进度区仅在 PreparedStaticArguments 初始化一次，后续不清零。
            stream = torch.npu.Stream(device=device)
            stream.wait_stream(torch.npu.current_stream())
            guard = prepared.workspace_guard_bytes

            def static_call():
                returned = module.entry(*prepared.core_args, config=compile_options)
                assert len(returned) == len(return_names)
                assert all(value is prepared.core_args[prepared.param_names.index(name)]
                           for value, name in zip(returned, return_names))

            def inspect_launch(generation, kind):
                actual = collect_state(fixture, output, call.args["idx_topk"])
                checks = exact_checks(actual, dynamic_reference)
                guards = guard_checks(fixture)
                progress = prepared.sections["progress"].cpu()[::layout["protocol"]["progress_words_per_core"]]
                protocol_check = compare_tensor(progress, progress_expectation(layout, generation), 0, 0)
                workspace_ok = bool((prepared.workspace_allocation[:guard].cpu() == 37).all()
                                    and (prepared.workspace_allocation[-guard:].cpu() == 37).all())
                launch = {"generation": generation, "kind": kind, "against_dynamic": checks,
                          "native_topk": compare_topk(actual["idx_topk"], native_reference["idx_topk"],
                                                       (fixture["positions"].cpu() + 1) // 4),
                          "guards": guards, "protocol": protocol_check, "progress": progress.tolist(),
                          "workspace_guards_passed": workspace_ok}
                launch["passed"] = checks_passed(checks) and checks_passed(guards) and workspace_ok \
                    and protocol_check["status"] == "PASS"
                report["launches"].append(launch)
                report["stage"] = f"{kind}_{generation}_complete"
                write_report(args.output, report)
                # 首轮有差异就保留完整状态用于定位，不继续多轮掩盖问题。
                if not launch["passed"]:
                    torch.save({"static": actual, "dynamic": dynamic_reference, "native": native_reference},
                               args.output / "failure_states.pt")
                    raise AssertionError("完整静态 CSA 与动态基线、进度或保护区不一致")

            with torch.npu.stream(stream):
                for generation in range(1, 4):
                    restore(fixture)
                    output.fill_(float("nan"))
                    call.args["idx_topk"].fill_(-777)
                    report["stage"] = f"static_launch_{generation}_enqueue"
                    write_report(args.output, report)
                    static_call()
                    stream.synchronize()
                    report["device_execution"] = True
                    inspect_launch(generation, "eager")
                if args.graph_replay:
                    restore(fixture)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        static_call()
                    stream.synchronize()
                    # ACLGraph capture 只记录调用，第一次 replay 才是第 4 次设备执行。
                    for generation in (4, 5):
                        restore(fixture)
                        output.fill_(float("nan"))
                        call.args["idx_topk"].fill_(-777)
                        graph.replay()
                        stream.synchronize()
                        inspect_launch(generation, "replay")
            report.update(passed=True, full_csa_validated=True, stage="complete", current_stream=int(stream.stream_id),
                          graph_replay_validated=bool(args.graph_replay), precision_scope="对冻结动态 PTO 逐元素精确比较")
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if original_qli is not None:
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
        write_report(args.output, report)


if __name__ == "__main__":
    main()
