# SPDX-License-Identifier: Apache-2.0
"""静态 CSA 旧/新预编译二进制的同卡配对诊断。

复用既有正式 W8A8 fixture 和 ops 参数准备；两侧使用相同根 Tensor、工作区、
输入、图重放和 L1 统计。旧/新完整输出、Top-K、cache/state、保护区及信号
代际交叉检查。直接 load_prebuilt，不重新编译，不运行 Native 或多 task。
"""

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "csa_static_reference"))


def save(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("stage", "passed")}, ensure_ascii=False), flush=True)


def progress_check(prepared, generation):
    """测试侧检查进度、所有保留字、只读控制表与arena首尾保护区。"""
    import torch

    layout = prepared.layout
    protocol = layout["protocol"]
    progress = prepared.sections["progress"].cpu().view(protocol["core_count"], protocol["progress_words_per_core"])
    expected = torch.tensor(layout["per_core_final_dispatch_seq"], dtype=progress.dtype)
    expected += generation * protocol["generation_stride"]
    guard = prepared.workspace_guard_bytes
    allocation = prepared.workspace_allocation
    tables = {name: torch.equal(prepared.sections[name].cpu(), torch.tensor(layout["workspace"][name],
                       dtype=torch.int64).view(prepared.sections[name].shape)) for name in ("records", "waits")}
    report = dict(generation=generation, progress=torch.equal(progress[:, 0], expected),
                  padding_zero=bool((progress[:, 1:] == 0).all()), controls=tables,
                  guard=bool((allocation[:guard].cpu() == 37).all() and (allocation[-guard:].cpu() == 37).all()),
                  observed_progress=progress[:, 0].tolist())
    report["passed"] = report["progress"] and report["padding_zero"] and report["guard"] and all(tables.values())
    if not report["passed"]:
        raise AssertionError("静态SPMD进度或保护检查失败：" + str(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l1-windows", type=int, default=6)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    if "TASK_DEVICE" not in os.environ or args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("必须经task-submit --device auto分配设备")
    if args.l1_windows != 6:
        raise ValueError("二进制诊断固定六个窗口，AB/BA各三次")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(stage="initializing", passed=False, history=args.history, batch=args.batch,
                  query_tokens=6, reference_only=True, eligible_for_dynamic_goal=False, launches=[])
    result_path = args.output / "result.json"
    try:
        import dsv4_csa_env
        dsv4_csa_env.activate()
        import pypto.torch
        import torch
        import torch_npu
        from pypto import pypto_core
        from dsv4_csa_native_case import native_session
        from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
        from dsv4_csa_validation import compare_tensor
        from offline_pd.run import decode_additional_config
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_static_reference.reference import prepare_static_csa

        current_platform.pre_register_and_update()
        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        from vllm_ascend.utils import enable_custom_op
        if not enable_custom_op():
            raise RuntimeError("Native fixture需要已安装的DSV4自定义算子包")
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa import native_adapter as adapter
        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs

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
        report["environment"] = dict(pypto=pypto.__file__, core=pypto_core.__file__, device=args.device)
        with native_session(config, args.device), torch.inference_mode():
            device = torch.device(f"npu:{args.device}")
            layer, report["weights"] = make_layer(config, Path(config.model_config.model), device, 4)
            fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 1024, device)
            group = fixture["groups"]["indexer"]
            scale = group["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            group["initial"] = group["allocation"].cpu()
            groups = {name: (fixture["metadata"][item["prefix"]], item["views"])
                      for name, item in fixture["groups"].items()}
            weights = adapter.prepare_weights(layer.self_attn, groups["indexer"][0].hadamard, layer)
            report["nz_weight_formats"] = {name: int(torch_npu.get_npu_format(weights[name]))
                                           for name in ("wq_a", "wq_b", "wo_a", "wo_b")}
            if set(report["nz_weight_formats"].values()) != {29}:
                raise RuntimeError("静态SPMD参考的根权重必须实际为NZ29")
            output = torch.empty_like(fixture["hidden"])
            call = adapter.NativeCSACall(
                adapter.CSAOperators.register(), weights, fixture["hidden"], fixture["positions"], groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name, compact_metadata=fixture["compact"],
                buffers={"idx_topk_scores": torch.empty((tokens, 512), dtype=torch.float32, device=device),
                         "idx_topk": torch.empty((tokens, 512), dtype=torch.int32, device=device), "x_out": output})
            prepared = prepare_static_csa(call.args, history=args.history, batch=args.batch, workspace_guard_bytes=512)
            # 同一根对象；静态数学和工作区从ops包独立加载，不读取历史生成目录。
            assert all(tensor is call.args[name] for name, tensor in prepared.native_args.items())
            report["operator_source"] = str(prepared.case_directory)
            report["reference"] = prepared.reference
            pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer",
                             enable_chip_swimlane=1 if args.l1_windows else 0,
                             enable_dep_gen=bool(args.l1_windows),
                             output_dir=str(args.output / "dfx") if args.l1_windows else None,
                             **ring_sizing_kwargs(additional))
            inputs = {"A": fixture["hidden"].clone()}
            inputs["B"] = -inputs["A"]
            baseline = {}

            def state():
                result = collect_state(fixture, output, call.args["idx_topk"])
                result["idx_topk_scores"] = call.args["idx_topk_scores"].cpu()
                return result

            def reset(label="A"):
                restore(fixture)
                fixture["hidden"].copy_(inputs[label])
                output.fill_(float("nan"))
                call.args["idx_topk"].fill_(-777)
                call.args["idx_topk_scores"].fill_(float("nan"))

            def compare(label="A"):
                checked = {key: compare_tensor(value, baseline[label][key], 0, 0) for key, value in state().items()}
                guards = guard_checks(fixture)
                if not all(row["status"] == "PASS" for row in (*checked.values(), *guards.values())):
                    raise AssertionError("完整CSA输出/TopK/cache/state或Native保护区与旧二进制基准不符：" + str(checked))
                return dict(tensors=checked, guards=guards)

            # 直接装载当时保存的 prebuilt；不调用 JIT 编译入口，也不重建二进制。
            from pypto._kernel_abi import KernelABI
            from pypto.jit._artifact_manifest import BuildKind
            from pypto.runtime._kernel_artifact import KernelArtifact
            from pypto.runtime._prebuilt import load_prebuilt
            from pypto.runtime.kernel.context import get_process_kernel_state
            from pypto.torch.launch import describe_eager_call, invoke
            from profiling import export_window
            import statistics

            manifest = json.loads((Path(__file__).parent / "artifacts.json").read_text())
            selected = manifest[f"h{args.history}_b{args.batch}"]
            artifacts = {}
            abi_reference = None
            for side, path in selected.items():
                directory = Path(path)
                meta = json.loads((directory / "compiled_meta.json").read_text())
                abi = KernelABI.from_record(meta["kernel_abi"])
                if abi_reference is not None:
                    abi.require_compatible(abi_reference)
                abi_reference = abi
                artifact = KernelArtifact(directory, abi)
                # load_prebuilt 保留运行时原有的 ABI/产物校验，无编译、锁或文件写入。
                artifact._loaded = load_prebuilt(directory, abi.platform, BuildKind.SINGLE_CHIP,
                                                kernel_abi=abi)["."]
                artifacts[side] = artifact
            report["artifact_paths"] = selected
            report["recompiled"] = False
            report["identical_arguments_and_workspace"] = True
            report["abi_parameter_count"] = len(abi_reference.parameters)
            bound = get_process_kernel_state().require_config()
            actual_calls = 0

            def launch(side):
                artifact = artifacts[side]
                frame = describe_eager_call(artifact.kernel_abi, tuple(prepared.core_args), bound)
                return invoke(artifact, frame, bound)

            # 同一组根 Tensor 和同一份工作区，旧、新二进制共用递增代际。
            # 先保存旧二进制 A/B 结果，逐 bit 核对新二进制，不引入已被别会话更新的多 task。
            for label in ("A", "B", "A"):
                for side in artifacts:
                    reset(label)
                    for tensor in prepared.temporaries.values():
                        tensor.fill_(37)
                    launch(side)
                    torch.npu.synchronize()
                    actual_calls += 1
                    if side == "spmd_old" and label not in baseline:
                        baseline[label] = state()
                    checked = compare(label)
                    checked["protocol"] = progress_check(prepared, actual_calls)
                    report["launches"].append(dict(side=side, label=label, checks=checked))
            report["stage"] = "binary_pair_functional_passed"
            save(result_path, report)

            def order(sample):
                sides = list(artifacts)
                return sides if sample % 2 == 0 else list(reversed(sides))

            records, graphs = [], {}
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for side in artifacts:
                    reset()
                    stream.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        launch(side)
                    graphs[side] = graph
                    reset()
                    graph.replay()
                    stream.synchronize()
                    actual_calls += 1
                    compare()
                    progress_check(prepared, actual_calls)
                # 统一预热条件；核内测量仍只统计最早 incore 开始至最后结束。
                for sample in range(6):
                    for side in order(sample):
                        reset()
                        graphs[side].replay()
                        stream.synchronize()
                        actual_calls += 1
                window = 0
                for sample in range(args.l1_windows):
                    for side in order(sample):
                        reset()
                        stream.synchronize()
                        pypto.torch.begin_dfx()
                        try:
                            graphs[side].replay()
                        finally:
                            pypto.torch.end_dfx()
                        actual_calls += 1
                        source = args.output / "dfx"
                        if window:
                            source = source / f"window_{window}"
                        records.append(export_window(source, args.output / f"l1_{sample:02d}_{side}", side,
                                       kernel_config=Path(selected[side]) / "kernel_config.py"))
                        window += 1
                        if sample == args.l1_windows - 1:
                            compare()
                            progress_check(prepared, actual_calls)
            samples = {side: [r["incore_span_us"] for r in records if r["side"] == side] for side in artifacts}
            report["l1"] = dict(windows=records, sample_order=[order(i) for i in range(args.l1_windows)],
                                statistics_us={side: dict(samples_us=values, median_us=statistics.median(values),
                                                         min_us=min(values), max_us=max(values))
                                               for side, values in samples.items()})
            report.update(stage="complete", passed=True, graph_replay_passed=True,
                          total_protocol_generations=actual_calls,
                          scope="同卡、同进程、同根地址、同工作区旧新二进制配对；未重新编译")
    except Exception as exc:
        report.update(stage="failed", passed=False, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        save(result_path, report)


if __name__ == "__main__":
    main()
