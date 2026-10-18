# SPDX-License-Identifier: Apache-2.0
"""从ops静态参考包独立复现完整CSA：正式权重、精度、图重放及可选L1。

算子入口、同步helper、参数和调度合同全部来自ops/pypto下的静态包；本测试
不读取csa_multishape_spmd中的历史源码、调度生成器或结果目录。复用既有通用
Native fixture/权重加载和原多task PTO基线，仅在tests中完成比较及profiling。
"""

import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace


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
    parser.add_argument("--l1-windows", type=int, default=5)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TASK_DEVICE", "-1")))
    args = parser.parse_args()
    if "TASK_DEVICE" not in os.environ or args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("必须经task-submit --device auto分配设备")
    if args.l1_windows not in (0, 3, 5):
        raise ValueError("L1使用0、3或5个窗口；0只验证功能与图重放")
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
        from pypto import passes, pypto_core
        from pypto.runtime import CompileOptions
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
            options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
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
                    raise AssertionError("完整CSA输出/TopK/cache/state或Native保护区与原多task不符：" + str(checked))
                return dict(tensors=checked, guards=guards)

            for label in ("A", "B"):
                reset(label)
                report["stage"] = f"multitask_reference_{label}"
                save(result_path, report)
                adapter.decode_csa_tp1_layer_test(*call.core_args, config=options)
                torch.npu.synchronize()
                baseline[label] = state()
                assert all(item["status"] == "PASS" for item in guard_checks(fixture).values())
            report["stage"] = "archive_static_compile"
            save(result_path, report)
            prepared.compile(config=options)
            report["compile"] = prepared.compile_report
            for generation, label in enumerate(("A", "B", "A"), 1):
                reset(label)
                for tensor in prepared.temporaries.values():
                    tensor.fill_(37)
                prepared(config=options)
                torch.npu.synchronize()
                checked = compare(label)
                checked["protocol"] = progress_check(prepared, generation)
                report["launches"].append(dict(generation=generation, label=label, passed=True, checks=checked))
                report["stage"] = f"archive_static_{generation}_passed"
                save(result_path, report)
            if args.l1_windows:
                from profiling import profile_pair

                report["l1"] = profile_pair(
                    entries={"dynamic_pto": lambda: adapter.decode_csa_tp1_layer_test(*call.core_args, config=options),
                             "static_spmd": lambda: prepared(config=options)},
                    reset=reset, verify=compare, output=args.output, windows=args.l1_windows,
                    kernel_configs={"static_spmd": Path(prepared.compile_report["kernel_config"])},
                    verify_protocol=lambda side, calls: progress_check(prepared, 3 + calls)
                        if side == "static_spmd" else dict(applicable=False, passed=True))
                report["graph_replay_passed"] = True
            else:
                reset()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    prepared(config=options)
                reset()
                graph.replay()
                torch.npu.synchronize()
                report["graph_checks"] = compare()
                report["graph_protocol"] = progress_check(prepared, 4)
                report["graph_replay_passed"] = True
            report.update(stage="complete", passed=True,
                          scope="当前H/B/S6；正式第4层权重、固定seed合成历史；独立ops静态参考复现")
    except Exception as exc:
        report.update(stage="failed", passed=False, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        save(result_path, report)


if __name__ == "__main__":
    main()
