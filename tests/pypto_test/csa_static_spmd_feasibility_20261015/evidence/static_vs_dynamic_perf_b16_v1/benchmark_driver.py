# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""仅比较冻结动态 PTO 与已通过验证的单 SPMD；不执行 Native forward。"""

import argparse
import ast
import importlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from test_static_jit_device import (
    CompileOptions, PreparedStaticArguments, checks_passed, collect_state,
    decode_additional_config, exact_checks, guard_checks, make_fixture, make_layer,
    native_session, passes, progress_expectation, pypto, torch, torch_npu, verify_isolation,
)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def statistics_us(values):
    ordered = sorted(values)
    return {"samples_us": values, "count": len(values), "median_us": statistics.median(values),
            "mean_us": statistics.mean(values), "min_us": ordered[0], "max_us": ordered[-1],
            "p95_us": ordered[math.ceil(0.95 * len(values)) - 1]}


def export_window(source, destination, side):
    """保留原始六列；第 4/5 列是 incore 起止，不能把收任务时间作为开始。"""
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("window_*"))
    raw = json.loads((destination / "chip_swimlane_records.json").read_text())
    frequency = raw["metadata"]["clock_freq_hz"]
    rows = raw["aicore_tasks"]
    if raw["chip_swimlane_level"] != 1 or not rows or frequency <= 0:
        raise ValueError("需要有效 L1 核内任务时间记录")
    if any(len(row) != 6 or row[4] < row[3] or row[3] <= 0 for row in rows):
        raise ValueError("原始泳道时间戳异常")
    if len({row[0] for row in rows}) != 72:
        raise ValueError("泳道未覆盖 24 AIC + 48 AIV")
    if side == "single_spmd" and (len(rows) != 72 or len({row[1] for row in rows}) != 1):
        raise ValueError("单 SPMD 不应产生多个 incore task")
    tables = list(Path(os.environ["PYPTO_PROG_BUILD_DIR"]).glob("*/kernel_config.py"))
    names = {}
    for table in tables:
        if (side == "single_spmd") != ("decode_csa" not in table.parent.name):
            continue
        tree = ast.parse(table.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KERNELS"
                                                    for t in node.targets):
                for item in node.value.elts:
                    values = {ast.literal_eval(k): v for k, v in zip(item.keys, item.values)}
                    names[str(ast.literal_eval(values["func_id"]))] = ast.literal_eval(values["name"])
        shutil.copy2(table, destination / (table.parent.name + "_kernel_config.py"))
    save(destination / "name_map.json", {"callable_id_to_name": names})
    converted = subprocess.run(
        [sys.executable, "-m", "simpler_setup.tools.swimlane_converter",
         str(destination / "chip_swimlane_records.json"), "--func-names",
         str(destination / "name_map.json"), "-o", str(destination / "swimlane.json")],
        text=True, capture_output=True)
    (destination / "converter.log").write_text(converted.stdout + converted.stderr)
    converted.check_returncode()
    return {"incore_span_us": (max(row[4] for row in rows) - min(row[3] for row in rows)) * 1e6 / frequency,
            "clock_freq_hz": frequency, "core_records": len(rows),
            "task_count": len({row[1] for row in rows}), "cores": len({row[0] for row in rows}),
            "raw": str(destination / "chip_swimlane_records.json"),
            "trace": str(destination / "swimlane.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--windows", type=int, default=3)
    args = parser.parse_args()
    if "TASK_DEVICE" not in os.environ or args.device != int(os.environ["TASK_DEVICE"]):
        raise RuntimeError("必须通过统一队列分配单卡")
    if args.case.name != "h131072_b16":
        raise ValueError("本轮仅测试用户指定的 128K/B16/S6")
    report = {"passed": False, "stage": "initializing", "history": 131072, "batch": 16,
              "query_tokens": 6, "tokens": 96, "device": args.device, "native_forward_executed": False,
              "scope": "正式第4层权重，完整CSA attention半层，合成历史；固定输入图重放",
              "fixture_restored_per_call": True, "signal_reset_between_calls": False,
              "warmup": args.warmup, "iterations": args.iters, "l1_windows_per_side": args.windows,
              "consecutive_decode_trajectory_validated": False, "l1_windows": []}

    def stage(value):
        report["stage"] = value
        save(args.output / "result.json", report)
        print(value, flush=True)

    try:
        report["environment"] = verify_isolation()
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
            raise RuntimeError("fixture 初始化所需自定义算子未加载")
        additional = decode_additional_config(SimpleNamespace(
            graph_mode="full_decode_only", capture_sizes=[96], recompute_scheduler=False))
        additional.update(weight_nz_mode=2, enable_kv_nz=False, enable_dsa_cp=False,
                          pto_csa_ring_config={"heap_mb": [256, 128, 256, 32], "task_window": 4096})
        config = EngineArgs(
            model="/data/model/DeepSeek-V4-Flash-0731-w8a8", tokenizer_mode="deepseek_v4",
            trust_remote_code=True, dtype="bfloat16", quantization="ascend", tensor_parallel_size=1,
            hf_overrides={"sliding_window": 128}, max_model_len=131200,
            max_num_seqs=40, max_num_batched_tokens=256, block_size=32, enable_prefix_caching=False,
            speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [96],
                                "cache_dir": str(args.output / "compile_cache")},
            additional_config=additional,
        ).create_engine_config()
        from vllm_ascend.ops.pypto.variant import ring_sizing_kwargs, variant_package
        adapter = importlib.import_module(f"{variant_package()}.native_adapter")
        assert frozen in Path(adapter.__file__).resolve().parents
        with native_session(config, args.device), torch.inference_mode():
            device = torch.device(f"npu:{args.device}")
            layer, report["weights"] = make_layer(config, Path(config.model_config.model), device, 4)
            fixture = make_fixture(config, layer.self_attn, 16, 131072, 1024, device)
            group = fixture["groups"]["indexer"]
            scale = group["views"][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            group["initial"] = group["allocation"].cpu()
            output = torch.empty_like(fixture["hidden"])
            stage("formal_weights_and_fixture_ready")
            pypto.torch.init(device=args.device, platform="a2a3", runtime="tensormap_and_ringbuffer",
                             enable_chip_swimlane=1, enable_dep_gen=True,
                             output_dir=str(args.output / "dfx"), **ring_sizing_kwargs(additional))
            groups = {name: (fixture["metadata"][group["prefix"]], group["views"])
                      for name, group in fixture["groups"].items()}
            weights = adapter.prepare_weights(layer.self_attn, groups["indexer"][0].hadamard, layer)
            call = adapter.NativeCSACall(
                adapter.CSAOperators.register(), weights, fixture["hidden"], fixture["positions"], groups,
                layer_name=layer.self_attn.dsa_attn.dsa_attn.layer_name, compact_metadata=fixture["compact"],
                buffers={"idx_topk_scores": torch.empty((96, 512), dtype=torch.float32, device=device),
                         "idx_topk": torch.empty((96, 512), dtype=torch.int32, device=device), "x_out": output})
            layout = json.loads((args.case / "layout.json").read_text())
            progress_expectation(layout, 2 + args.warmup + args.iters + args.windows)
            prepared = PreparedStaticArguments(args.case, call.args, workspace_guard_bytes=512)
            prepared.require_core_capacity(256)
            for name in ("entry_jit.py", "layout.json", "entry.bindings.json"):
                shutil.copy2(args.case / name, args.output / name)
            shutil.copy2(Path(__file__).with_name("local_signal_adapter.py"), args.output / "local_signal_adapter.py")
            spec = importlib.util.spec_from_file_location("csa_static_perf_entry", args.output / "entry_jit.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
            entries = {
                "dynamic_pto": lambda: adapter.decode_csa_tp1_layer_test(*call.core_args, config=options),
                "single_spmd": lambda: module.entry(*prepared.core_args, config=options),
            }
            initial = {name: group["initial"].to(device) for name, group in fixture["groups"].items()}

            def reset():
                # 恢复在计时及 DFX 窗口外；两边共用权重、metadata、输入和输出地址。
                for name, group in fixture["groups"].items():
                    group["allocation"].copy_(initial[name])
                output.fill_(float("nan"))
                call.args["idx_topk"].fill_(-777)

            def check(reference):
                actual = collect_state(fixture, output, call.args["idx_topk"])
                checks = exact_checks(actual, reference)
                if not checks_passed(checks):
                    raise AssertionError("性能路径输出/状态与动态 eager 参考不一致：" + str(checks))
                return checks

            stream = torch.npu.Stream(device=device)
            stream.wait_stream(torch.npu.current_stream())
            graphs, generations = {}, 0
            with torch.npu.stream(stream):
                reset()
                entries["dynamic_pto"]()
                stream.synchronize()
                reference = collect_state(fixture, output, call.args["idx_topk"])
                reset()
                stage("single_spmd_first_compile_and_launch")
                entries["single_spmd"]()
                stream.synchronize()
                generations += 1
                report["eager_exact"] = check(reference)
                stage("both_eager_entries_passed")
                # 预编译完成后只捕获 PTO 调用。reset 不进图，不执行 Native 对照。
                for side in entries:
                    reset()
                    stream.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        entries[side]()
                    graphs[side] = graph
                    reset()
                    graph.replay()
                    stream.synchronize()
                    if side == "single_spmd":
                        generations += 1
                    report[side + "_graph_exact"] = check(reference)
                stage("both_graph_replays_passed")
                start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                samples = {side: [] for side in entries}
                stamps = []
                for iteration in range(args.warmup + args.iters):
                    # AB/BA 交替，避免先测/后测顺序与温度趋势始终偏向一侧。
                    order = list(entries) if iteration % 2 == 0 else list(reversed(entries))
                    for side in order:
                        reset()
                        start.record()
                        graphs[side].replay()
                        end.record()
                        stream.synchronize()
                        if side == "single_spmd":
                            generations += 1
                        stamp = start.recorded_time()
                        if stamps and stamp <= stamps[-1]:
                            raise ValueError("设备事件时间戳没有更新")
                        stamps.append(stamp)
                        elapsed = start.elapsed_time(end) * 1000
                        if not math.isfinite(elapsed) or elapsed <= 0:
                            raise ValueError("设备事件耗时无效")
                        if iteration >= args.warmup:
                            samples[side].append(elapsed)
                report["event_replay"] = {side: statistics_us(values) for side, values in samples.items()}
                report["event_scope"] = "图外NPU Event包住单次图重放；不含fixture恢复，DFX窗口关闭"
                stage("event_timing_complete")
                window = 0
                for sample in range(args.windows):
                    order = list(entries) if sample % 2 == 0 else list(reversed(entries))
                    for side in order:
                        reset()
                        stream.synchronize()
                        pypto.torch.begin_dfx()
                        try:
                            graphs[side].replay()
                        finally:
                            pypto.torch.end_dfx()
                        if side == "single_spmd":
                            generations += 1
                        source = args.output / "dfx"
                        if window:
                            source /= f"window_{window}"
                        destination = args.output / "l1" / side / f"step_{sample + 1}"
                        record = export_window(source, destination, side)
                        record.update(side=side, step=sample + 1, dfx_window=window)
                        report["l1_windows"].append(record)
                        window += 1
                        if sample == args.windows - 1:
                            report[side + "_final_exact"] = check(reference)
                        stage(f"l1_{side}_{sample + 1}_complete")
                progress = prepared.sections["progress"].cpu()[::layout["protocol"]["progress_words_per_core"]]
                expected = progress_expectation(layout, generations)
                report["signal_generations"] = generations
                report["progress_passed"] = torch.equal(progress, expected)
                report["guards"] = guard_checks(fixture)
                guard = prepared.workspace_guard_bytes
                report["workspace_guards_passed"] = bool(
                    (prepared.workspace_allocation[:guard].cpu() == 37).all()
                    and (prepared.workspace_allocation[-guard:].cpu() == 37).all())
                if not (report["progress_passed"] and report["workspace_guards_passed"]
                        and checks_passed(report["guards"])):
                    raise AssertionError("性能结束后的信号或保护区检查失败")
            report["l1_incore"] = {
                side: statistics_us([r["incore_span_us"] for r in report["l1_windows"] if r["side"] == side])
                for side in entries}
            report["l1_scope"] = "min(incore start)至max(incore end)，保留核内等待和中间空隙，排除外围调度首尾"
            report["comparison"] = {}
            for metric in ("l1_incore", "event_replay"):
                baseline = report[metric]["dynamic_pto"]["median_us"]
                candidate = report[metric]["single_spmd"]["median_us"]
                report["comparison"][metric] = {"dynamic_pto_us": baseline, "single_spmd_us": candidate,
                                                "speedup": baseline / candidate,
                                                "latency_change_pct": (candidate / baseline - 1) * 100}
            report.update(passed=True, performance_measured=True, graph_replay_validated=True)
            stage("complete")
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        stage("failed")
        raise


if __name__ == "__main__":
    main()
