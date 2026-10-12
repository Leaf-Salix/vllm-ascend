# SPDX-License-Identifier: Apache-2.0
"""正式 HC 权重、合成 BF16 输入的单卡 HC_pre 数值与性能对照。"""

import argparse
import importlib.util
import itertools
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path

from dsv4_csa_env import activate, write_json


def load_weights(checkpoint, layer, branch, device):
    """只读取三份 HC 权重，不构造模型、不加载 attention/MoE 或 KV cache。"""
    import torch
    from safetensors import safe_open

    from vllm_ascend.ops.pypto.hc_pre import D, HC_DIM, HC_EPS, HC_MULT, HC_SINKHORN_ITER, MIX_HC, NORM_EPS

    config = json.loads((checkpoint / "config.json").read_text())
    expected = {"hidden_size": D, "hc_mult": HC_MULT, "hc_sinkhorn_iters": HC_SINKHORN_ITER,
                "rms_norm_eps": NORM_EPS, "hc_eps": HC_EPS}
    for name, value in expected.items():
        if config[name] != value:
            raise ValueError(f"HC_pre 专用配置不匹配：{name}={config[name]}，预期 {value}")
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    weights, records = [], []
    for suffix, shape in (("fn", (MIX_HC, HC_DIM)), ("scale", (3,)), ("base", (MIX_HC,))):
        name = f"layers.{layer}.hc_{branch}_{suffix}"
        with safe_open(checkpoint / index[name], framework="pt", device="cpu") as reader:
            value = reader.get_tensor(name)
        if value.dtype != torch.float32 or tuple(value.shape) != shape:
            raise ValueError(f"正式 HC 参数不符合接口：{name} {value.dtype} {tuple(value.shape)}")
        weights.append(value.to(device))
        records.append({"name": name, "shard": index[name], "shape": list(value.shape), "dtype": str(value.dtype)})
    return weights, records


def compare(actual, reference, *, exact=False):
    """逐元素门禁；BF16 和 FP32 分别使用固定容差，另保留零容差差异。"""
    import torch

    result = {}
    for name, value, expected in zip(("mixed", "post", "comb"), actual, reference, strict=True):
        value, expected = value.detach().cpu(), expected.detach().cpu()
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"{name} 输出 shape/dtype 不匹配")
        rtol, atol = (0.0, 0.0) if exact else ((0.01, 0.01) if value.dtype == torch.bfloat16 else (1e-4, 1e-5))
        diff = (value.float() - expected.float()).abs()
        finite = bool(torch.isfinite(value).all() and torch.isfinite(expected).all())
        bad = diff > atol + rtol * expected.float().abs()
        result[name] = {
            "passed": finite and not bool(bad.any()), "finite": finite,
            "shape": list(value.shape), "dtype": str(value.dtype), "rtol": rtol, "atol": atol,
            "max_abs": float(diff.max()), "rmse": float(diff.square().mean().sqrt()),
            "different": int((value != expected).sum()), "out_of_tolerance": int(bad.sum()),
        }
    return result


def require_pass(result):
    if not all(item["passed"] for item in result.values()):
        raise AssertionError(f"HC_pre 数值检查失败：{result}")


def capture(run):
    import torch

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = run()
    graph.replay()
    torch.npu.synchronize()
    return graph, outputs


def measure_pair(graphs, warmup, iters):
    """交替测两侧，图外 Event 包住一次 replay，不含编译、分配、权重加载和比较。"""
    import torch

    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    samples = {name: [] for name in graphs}
    last_stamp = None
    for iteration in range(warmup + iters):
        names = list(graphs) if iteration % 2 == 0 else list(reversed(graphs))
        for name in names:
            start.record()
            graphs[name].replay()
            end.record()
            torch.npu.synchronize()
            stamp = start.recorded_time()
            if last_stamp is not None and stamp <= last_stamp:
                raise RuntimeError("图外 Event 时间戳未更新")
            last_stamp = stamp
            if iteration >= warmup:
                samples[name].append(start.elapsed_time(end) * 1000)
    return {
        name: {"p50_us": statistics.median(values), "p95_us": sorted(values)[math.ceil(0.95 * iters) - 1],
               "min_us": min(values), "samples_us": values}
        for name, values in samples.items()
    }


def profile_graph(graph, output):
    """额外采三个重放 step；profiler 不进入性能统计区间。"""
    import torch
    import torch_npu

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output)),
    ) as trace:
        for _ in range(3):
            graph.replay()
            torch.npu.synchronize()
            trace.step()
    paths = list(output.glob("*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    if len(paths) != 1:
        raise RuntimeError(f"预期唯一 PyTorch trace，实到 {len(paths)}：{output}")
    return paths[0]


def measure_incore(graphs, args, case_name, window, variant=None):
    """Native 取设备 HcPre，PTO 取指定档位泳道内 incore 起止；两侧各三个重放。"""
    import pypto.torch
    import torch
    from dsv4_csa_single_card_bench import _export_swimlane
    from dsv4_hc_pre_timing import summarize_native, summarize_swimlane

    for _ in range(args.warmup):
        for graph in graphs.values():
            graph.replay()
    torch.npu.synchronize()
    native_trace = profile_graph(graphs["native"], args.output / "profile" / case_name / "native")
    pypto.torch.begin_dfx()
    try:
        if args.profile:
            pto_trace = profile_graph(graphs["pto"], args.output / "profile" / case_name / "pto")
        else:
            for _ in range(3):
                graphs["pto"].replay()
                torch.npu.synchronize()
    finally:
        pypto.torch.end_dfx()
    directory = args.output / "dfx"
    if window:
        directory /= f"window_{window}"
    exported = _export_swimlane(directory, kernel_pattern="_jit_hc_pre_*/kernel_config.py")
    if not exported["exported"]:
        raise RuntimeError(f"泳道导出失败：{exported}")
    actual_level = json.loads((directory / "chip_swimlane_records.json").read_text())["chip_swimlane_level"]
    if actual_level != args.swimlane_level:
        raise ValueError(f"泳道实际档位 {actual_level} 与请求档位 {args.swimlane_level} 不一致")
    traces = args.output / "traces"
    traces.mkdir(exist_ok=True)
    native_copy = traces / f"{case_name}_Native_PyTorch_3steps.json"
    swimlane_copy = traces / f"{case_name}_PTO_L{actual_level}_3steps.json"
    shutil.copyfile(native_trace, native_copy)
    shutil.copyfile(exported["merged_swimlane"], swimlane_copy)
    native = summarize_native(native_copy, 3)
    pto = summarize_swimlane(swimlane_copy, 3, level=actual_level)
    # 融合后阶段数会变化，以本进程实际编译的任务表检查每次重放完整性。
    names = json.loads(Path(exported["name_map"]).read_text())["callable_id_to_name"]
    required = {name.removesuffix("_spmd") for name in names.values()}
    if variant is not None:
        # 所有分支会一起编译，但一次重放只应执行当前形状的 AIC/AIV 对。
        expected_compiled = set(variant["all_compiled_tasks"])
        if required != expected_compiled:
            raise ValueError(f"HC_pre 分支任务表不匹配：实际 {required}，预期 {expected_compiled}")
        required = set(variant["required_tasks"])
    for sample in pto["samples"]:
        if not required.issubset(sample["tasks"]):
            raise ValueError(f"HC_pre 阶段缺失：{required - set(sample['tasks'])}")
        if variant is not None and set(sample["tasks"]) != required:
            raise ValueError(f"HC_pre 执行了其他分支：{set(sample['tasks']) - required}")
    result = {"scope": "Native HcPre 设备事件与 PTO incore 起止；不计调用外层 AICPU 区间",
              "native": native, "pto": pto, "dfx": exported,
              "native_over_pto_span": native["p50_us"] / pto["span_p50_us"],
              "native_over_pto_active_union": native["p50_us"] / pto["active_union_p50_us"]}
    if variant is not None:
        result["variant"] = variant
    if args.profile:
        target = traces / f"{case_name}_PTO_PyTorch_3steps.json"
        shutil.copyfile(pto_trace, target)
        result["pto_pytorch"] = str(target)
    return result


def run(args, report):
    from pypto.runtime import RunConfig

    from vllm_ascend.ops.pypto.hc_pre import D, HC_EPS, HC_MULT, HC_SINKHORN_ITER, NORM_EPS, hc_pre

    module = sys.modules["vllm_ascend.ops.pypto.hc_pre"]
    variants = getattr(module, "HC_PRE_VARIANTS", ())

    if args.lower_only or args.build_only:
        config = RunConfig(platform="a2a3", save_kernels=True,
                           save_kernels_dir=str(args.output / "build"))
        program = hc_pre.warmup(config=config).program if args.build_only else hc_pre.lower(config=config)
        (args.output / "hc_pre_lowered.py").write_text(str(program))
        report.update(status="COMPILED" if args.build_only else "LOWERED", device_execution=False)
        return
    if not os.environ.get("TASK_DEVICE"):
        raise RuntimeError("真机用例必须通过 task-submit --device auto 提交")

    import pypto.torch
    import torch
    import torch_npu  # noqa: F401

    if args.native_backend == "recipes":
        if args.native_library is None or args.native_opp is None:
            raise ValueError("recipes 基线必须提供 --native-library 和 --native-opp，禁止静默回退旧版本")
        torch.ops.load_library(str(args.native_library.resolve(strict=True)))
        reference = args.native_library.parent / "reference.json"
        report["native_reference"] = json.loads(reference.read_text())
    else:
        import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    weights, records = load_weights(args.checkpoint, args.layer, args.branch, device)
    report.update(weights=records, device_execution=True, physical_device=os.environ["TASK_DEVICE"],
                  pypto_path=pypto.__file__, torch_version=torch.__version__, torch_npu_version=torch_npu.__version__)
    pypto.torch.init(device=0, platform="a2a3", runtime="tensormap_and_ringbuffer",
                     **({"enable_chip_swimlane": args.swimlane_level, "enable_dep_gen": True,
                         "output_dir": str(args.output / "dfx")} if args.timing == "incore" else {}))
    op = pypto.torch.register(hc_pre, "dsv4_hc_pre_bench::hc_pre")
    for window, (batch_size, seqlen) in enumerate(itertools.product(args.batch_size, args.seqlen)):
        tokens = batch_size * seqlen
        case_name = f"B{batch_size:02d}_S{seqlen}_T{tokens:03d}"
        generator = torch.Generator().manual_seed(args.seed + batch_size * 100 + seqlen)
        row = {"batch_size": batch_size, "seqlen": seqlen, "tokens": tokens, "name": case_name}
        variant = None
        if variants:
            selected = next(item for item in variants if item[0] is None or tokens <= item[0])
            _, name, m_tile, k_tile = selected
            variant = {"name": name, "m_tile": m_tile, "k_tile": k_tile,
                       "selection": "动态 T=B×S；由 tensor shape 分派",
                       "required_tasks": [f"{name}_aic", f"{name}_aiv"],
                       "all_compiled_tasks": [f"{item[1]}_{core}" for item in variants for core in ("aic", "aiv")]}
            row["variant"] = variant
        report["cases"].append(row)
        x = torch.randn(tokens, HC_MULT, D, generator=generator, dtype=torch.bfloat16).to(device)
        # B/S 只在入口用零拷贝 view 合并；Native 接收真实 [B,S,hc,D] 四维输入。
        x_native = x.view(batch_size, seqlen, HC_MULT, D)
        input_before = x.cpu()
        allocations = (
            torch.full((tokens + 2, D), 37, device=device, dtype=torch.bfloat16),
            torch.full((tokens + 2, HC_MULT), 37, device=device, dtype=torch.float32),
            torch.full((tokens + 2, HC_MULT, HC_MULT), 37, device=device, dtype=torch.float32),
        )
        outputs = tuple(value[1:-1] for value in allocations)
        for value in outputs:
            value.fill_(float("nan"))

        def native():
            if args.native_backend == "recipes":
                values = torch.ops.custom.npu_hc_pre(
                    x_native, *weights, hc_mult=HC_MULT, hc_sinkhorn_iters=HC_SINKHORN_ITER,
                    norm_eps=NORM_EPS, hc_eps=HC_EPS)
            else:
                values = torch.ops._C_ascend.npu_hc_pre_v2(
                    x_native, *weights, HC_MULT, HC_SINKHORN_ITER, NORM_EPS, HC_EPS)
            return tuple(value.flatten(0, 1) for value in values)

        def pto():
            op(x, *weights, *outputs)
            return outputs

        native_eager = tuple(value.cpu() for value in native())
        if args.native_backend == "recipes" and "native_loaded_api" not in report:
            expected_api = (args.native_opp / "op_api/lib/libcust_opapi.so").resolve()
            loaded = {Path(line.split()[-1]).resolve() for line in Path("/proc/self/maps").read_text().splitlines()
                      if line.endswith("/libcust_opapi.so")}
            if expected_api not in loaded:
                raise RuntimeError(f"未加载指定上游 HcPre API：{expected_api}；实际 {loaded}")
            report["native_loaded_api"] = str(expected_api)
        pto_eager = tuple(value.cpu() for value in pto())
        row["pto_native"] = compare(pto_eager, native_eager)
        require_pass(row["pto_native"])
        graphs, graph_outputs = {}, {}
        for name, runner, expected in (("native", native, native_eager), ("pto", pto, pto_eager)):
            graphs[name], graph_outputs[name] = capture(runner)
            row[f"{name}_graph_eager"] = compare(graph_outputs[name], expected, exact=True)
            require_pass(row[f"{name}_graph_eager"])
        if args.timing == "incore":
            row["incore_timing"] = measure_incore(graphs, args, case_name, window, variant)
        else:
            row["replay_timing"] = measure_pair(graphs, args.warmup, args.iters)
            if args.profile:
                for name, graph in graphs.items():
                    profile_graph(graph, args.output / "profile" / case_name / name)
        if not torch.equal(x.cpu(), input_before):
            raise AssertionError("HC_pre 改写了输入")
        # 同址换输入并毒化输出，确认捕获图真正重算，而非读到旧结果。
        x.copy_(torch.randn(tokens, HC_MULT, D, generator=generator, dtype=torch.bfloat16).to(device))
        for values in graph_outputs.values():
            for value in values:
                value.fill_(float("nan"))
        expected = tuple(value.cpu() for value in native())
        for name, graph in graphs.items():
            graph.replay()
            row[f"{name}_updated_input"] = compare(graph_outputs[name], expected, exact=(name == "native"))
            require_pass(row[f"{name}_updated_input"])
        row["output_guards"] = all(bool((value[0].cpu() == 37).all() and (value[-1].cpu() == 37).all())
                                   for value in allocations)
        if not row["output_guards"]:
            raise AssertionError("PTO 输出越界改写保护行")
        row["status"] = "PASS"
        write_json(args.output / "report.json", report)
        measured = row.get("incore_timing")
        print(json.dumps({"case": case_name, "status": row["status"], "timing": args.timing,
                          **({"native_kernel_p50_us": measured["native"]["p50_us"],
                              "pto_incore_span_p50_us": measured["pto"]["span_p50_us"],
                              "pto_active_union_p50_us": measured["pto"]["active_union_p50_us"]}
                             if measured else row["replay_timing"])}), flush=True)
    report["status"] = "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/model/DeepSeek-V4-Flash-0731-w8a8"))
    parser.add_argument("--batch-size", type=int, nargs="+", default=[2, 4, 8, 16, 20, 24, 28])
    parser.add_argument("--seqlen", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--timing", choices=("incore", "replay"), default="incore")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--branch", choices=("attn", "ffn"), default="attn")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20, help="仅 replay 模式使用；incore 固定采三个 step")
    parser.add_argument("--profile", action="store_true", help="额外采集 PTO PyTorch 外部 trace，三个 step")
    parser.add_argument("--swimlane-level", type=int, choices=(1, 4), default=1,
                        help="incore 模式的泳道采集档位；L4 额外记录 AICPU 派发、调度与编排")
    parser.add_argument("--pypto-core", type=Path, help="仅本进程使用的本地调试扩展，须与当前 PyPTO 源码一致")
    parser.add_argument("--kernel-file", type=Path, help="独立 HC_pre 源码快照，供优化候选隔离编译和采样")
    parser.add_argument("--native-backend", choices=("recipes", "release"), default="recipes")
    parser.add_argument("--native-library", type=Path, help="由指定 cann-recipes-infer 源码构建的 HC_pre 绑定")
    parser.add_argument("--native-opp", type=Path, help="对应算子包的 vendors/customize 目录")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--lower-only", action="store_true")
    mode.add_argument("--build-only", action="store_true")
    # 队列可能追加物理卡号；实际运行统一由外层脚本映射到逻辑卡 0。
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if min(args.batch_size + args.seqlen) < 1 or args.iters < 1 or args.warmup < 0 or args.layer < 0:
        parser.error("batch-size/seqlen/iters 必须为正，warmup/layer 不能为负")
    if len(set(args.batch_size)) != len(args.batch_size) or len(set(args.seqlen)) != len(args.seqlen):
        parser.error("batch-size/seqlen 不得重复")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        parser.error("请使用新的输出目录，避免覆盖历史证据")
    activate()
    if args.native_opp is not None:
        opp = args.native_opp.resolve(strict=True)
        if not (opp / "op_api/lib/libcust_opapi.so").is_file():
            raise ValueError(f"Native 算子包不完整：{opp}")
        # 在 NPU 初始化之前只选择本进程基线包，不覆盖共用安装。
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(opp)
    if args.pypto_core is not None:
        core = args.pypto_core.resolve(strict=True)
        # 共用环境的 editable 安装可能落后于已编译源码；只重绑定当前进程，不覆盖共享安装。
        finders = [finder for finder in sys.meta_path
                   if "pypto.pypto_core" in getattr(finder, "known_wheel_files", {})]
        if len(finders) != 1:
            raise RuntimeError("无法唯一定位 PyPTO editable 扩展映射")
        finders[0].known_wheel_files["pypto.pypto_core"] = str(core)
    if args.kernel_file is not None:
        source = args.kernel_file.resolve(strict=True)
        name = "vllm_ascend.ops.pypto.hc_pre"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    report = {"status": "RUNNING", "scope": "单卡独立 HC_pre；正式 HC 权重、合成输入；不代表整模型性能",
              "implementation": "standalone_local_hc_pre",
              "native_backend": args.native_backend,
              "checkpoint": str(args.checkpoint), "layer": args.layer, "branch": args.branch,
              "batch_size": args.batch_size, "seqlen": args.seqlen,
              "seed": args.seed, "case_seed_rule": "seed + batch_size * 100 + seqlen",
              "warmup": args.warmup, "replay_iters": args.iters, "timing": args.timing,
              "swimlane_level": args.swimlane_level if args.timing == "incore" else 0,
              "timing_scope": (f"PTO L{args.swimlane_level} incore 首次开始至最后结束；另列并行区间并集和无 incore 空隙；Native HcPre 设备事件"
                               if args.timing == "incore" else "图外 NPU Event 包含一次完整图重放"),
              "runtime": "tensormap_and_ringbuffer", "cases": []}
    if args.kernel_file is not None:
        report["kernel_file"] = str(source)
    if args.native_opp is not None:
        report["native_opp"] = str(opp)
    if args.pypto_core is not None:
        report["pypto_core"] = str(core)
    try:
        run(args, report)
    except Exception as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
