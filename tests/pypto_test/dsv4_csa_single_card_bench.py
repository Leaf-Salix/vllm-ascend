"""单卡 schema=2 快照回放与 PTO 泳道诊断。

执行当前 HC_pre→norm→CSA→HC_post 根入口；输入可来自正式层权重的单卡 case，
也可来自真实模型调用。这里的 eager 墙钟包含主机派发，只用于诊断；
两侧完整设备区间使用 dsv4_csa_single_layer.py，最终以 16 卡整模型验收为准。
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

from dsv4_csa_env import activate, write_json
from dsv4_csa_replay import (
    SCHEMA_VERSION,
    argument_roles,
    convert_weight_layouts,
    materialize,
    restore_mutable_storages,
)
from dsv4_csa_validation import validate_outputs


def _export_swimlane(directory: Path) -> dict:
    """把本次 DFX 记录转成带真实任务名的泳道，任务名取自本进程实际生成的 kernel_config。"""
    import ast
    import json as _json
    import subprocess

    records = directory / "chip_swimlane_records.json"
    deps = directory / "deps.json"
    if not records.is_file() or not deps.is_file():
        return {"exported": False, "reason": "DFX 未产出记录或依赖"}
    # 在新的输出目录执行，唯一完整层编译产物才可作为任务名称依据。
    builds = list(Path("build_output").glob("_jit__decode_csa_tp1_layer_*/kernel_config.py"))
    if len(builds) != 1:
        return {"exported": False, "reason": f"需要唯一完整层 kernel_config.py，实到 {len(builds)} 份"}
    table = builds[0]
    recorded_table = directory / "kernel_config_source.py"
    shutil.copyfile(table, recorded_table)
    tree = ast.parse(table.read_text())
    tables = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "KERNELS" for t in node.targets)]
    if len(tables) != 1:
        return {"exported": False, "reason": f"{table} 里的 KERNELS 表不唯一"}
    names = {}
    for node in tables[0].elts:
        values = {ast.literal_eval(k): v for k, v in zip(node.keys, node.values)}
        names[str(ast.literal_eval(values["func_id"]))] = ast.literal_eval(values["name"])
    name_map = directory / "name_map.json"
    name_map.write_text(_json.dumps({"callable_id_to_name": names}, ensure_ascii=False, indent=2))
    merged = directory / "merged_swimlane.json"
    proc = subprocess.run([sys.executable, "-m", "simpler_setup.tools.swimlane_converter", str(records),
                           "--func-names", str(name_map), "-o", str(merged)],
                          capture_output=True, text=True)
    (directory / "converter_output.txt").write_text(proc.stdout + proc.stderr)
    return {"exported": proc.returncode == 0, "kernel_config": str(recorded_table),
            "merged_swimlane": str(merged), "name_map": str(name_map)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-dir", type=Path, required=True, help="run.py argdump 落盘目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="同输入/初态的参考输出与状态张量字典 .pt")
    parser.add_argument("--tolerances", type=Path, help="JSON：每个浮点输出分别声明 atol/rtol")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20, help="计时轮数（另有 3 轮预热）")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--swimlane", type=int, default=0, choices=range(5),
                        help="非 0 时开 DFX 芯片泳道；4 才有逐任务 kernel 时长")
    parser.add_argument("--windows", type=int, default=1,
                        help="采几个泳道窗口。单次调用的 makespan 受设备侧调度次序影响，"
                             "run 与 run 之间能差出几十 us；看改动效果要取多窗口的中位数")
    parser.add_argument("--pmu", type=int, default=0,
                        help="AICore PMU 事件：2=PIPE_UTILIZATION、4=MEMORY。走编译程序路径，"
                             "与 pypto.torch.init 互斥，所以开了它就不采泳道")
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 0:
        parser.error("--iters 必须大于 0，--warmup 不得为负")
    if (args.reference is None) != (args.tolerances is None):
        parser.error("--reference 与 --tolerances 必须同时提供")
    report = {"status": "RUNNING", "args_dir": str(args.args_dir),
              "reference": str(args.reference) if args.reference else None}
    try:
        _run_benchmark(args, report)
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        write_json(args.output / "report.json", report)
        raise
    write_json(args.output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "samples_us"}, ensure_ascii=False, indent=2))


def _run_benchmark(args, report):
    activate()

    import pypto.torch
    import torch
    import torch_npu  # noqa: F401  加载 NPU 后端
    from pypto.runtime import RunConfig

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.reduction import validate_reduction_mode
    from vllm_ascend.ops.pypto.variant import selected_variant, variant_package

    validate_reduction_mode()
    package = variant_package()
    # 当前入口覆盖 mHC 残差流之间的完整 HC_pre + norm + CSA + HC_post。
    entry = "decode_csa_tp1_layer_test"
    module = __import__(f"{package}.decode_csa", fromlist=[entry])
    kernel = getattr(module, entry)

    meta = json.loads((args.args_dir / "csa_args_meta.json").read_text())
    names = list(kernel.param_names)
    if names != meta["param_names"]:
        raise ValueError("落盘入参的参数表与当前 kernel 不一致，需重新采集快照")
    root = getattr(module, kernel.__name__)
    roles = argument_roles(root)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("旧快照缺少布局/初态/别名信息，必须重新采集 schema=2 快照")
    if any(meta["tensors"][name]["role"] != roles[name] for name in names):
        raise ValueError("快照读写角色与当前根签名不一致")
    blob = torch.load(args.args_dir / "csa_args.pt", map_location="cpu", weights_only=True)

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.nz_mode import root_weight_layouts, root_weight_shapes
    layouts = root_weight_layouts(root)
    device = f"npu:{args.device}"
    torch.npu.set_device(args.device)
    tensors, backings = materialize(meta, blob, device)
    tensors, converted = convert_weight_layouts(tensors, meta, layouts, root_weight_shapes(root))
    for name in converted:
        # 转换仅允许独占的只读权重，释放已被新布局替换的设备存储。
        sid = meta["tensors"][name]["storage"]
        del backings[sid]
    report.update(snapshot_source=meta["source"], root_layouts=layouts, converted_weights=converted,
                  state_reset="每次调用前恢复相同初态；恢复与同步均在计时/DFX 窗口之外")
    call_args = tuple(tensors[name] for name in names)

    args.output.mkdir(parents=True, exist_ok=True)
    if args.pmu:
        # PMU 只能从 RunConfig 走编译程序路径；它与 pypto.torch.init 在同一进程里互斥。
        compiled = kernel.compile(*call_args, config=RunConfig(
            platform="a2a3", device_id=args.device, enable_pmu=args.pmu,
            save_kernels=True, save_kernels_dir=str((args.output / "pmu").resolve())))
        run = lambda: compiled(*call_args)  # noqa: E731
    else:
        # 执行目标一次性定在进程上：JIT 调用本身不接 RunConfig。
        pypto.torch.init(device=args.device, platform="a2a3",
                         enable_chip_swimlane=args.swimlane,
                         enable_dep_gen=args.swimlane >= 4,
                         output_dir=str((args.output / "dfx").resolve()) if args.swimlane else None)
        run = lambda: kernel(*call_args)  # noqa: E731

    report.update({"variant": selected_variant(), "package": package, "device": args.device,
              "layer_index": meta["layer_index"], "tokens": meta["tokens"],
              "iters": args.iters, "warmup": args.warmup, "swimlane_level": args.swimlane,
              "scope": "单算子单卡回放，不含 MoE 与通信；绝对耗时不代表端到端性能"})
    # 正确性检查单独执行，不把 CPU 比较、拷贝或同步开销混入计时。
    reference = torch.load(args.reference, map_location="cpu", weights_only=True) if args.reference else None
    tolerances = json.loads(args.tolerances.read_text()) if args.tolerances else None
    restore_mutable_storages(meta, blob, backings)
    run()
    torch.npu.synchronize()
    report["validation"] = validate_outputs(tensors, kernel.output_param_names, reference, tolerances)
    if report["validation"]["status"] == "FAIL":
        raise ValueError(f"CSA 输出检查失败：{report['validation']['errors']}")
    for _ in range(args.warmup):
        restore_mutable_storages(meta, blob, backings)
        run()
    torch.npu.synchronize()

    samples = []
    for _ in range(args.iters):
        restore_mutable_storages(meta, blob, backings)
        torch.npu.synchronize()
        start = time.perf_counter()
        run()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    if args.swimlane:
        # 每个窗口只包一次调用：不然一个窗口里记好几遍，产物没法逐任务对照。
        # 墙钟里绝大部分是 eager 下的主机侧派发开销（PyPTO 的 _resolve_compiled
        # 按调用次数计费），要量 kernel 本身必须看泳道的 kernel-duration。
        # 第一个窗口写进 dfx/，之后的依次写进 dfx/window_1、dfx/window_2 ……
        for _ in range(max(1, args.windows)):
            restore_mutable_storages(meta, blob, backings)
            torch.npu.synchronize()
            pypto.torch.begin_dfx()
            try:
                run()
            finally:
                pypto.torch.end_dfx()
            torch.npu.synchronize()
    report.update(status=report["validation"]["status"],
                  us_min=samples[0], us_p50=statistics.median(samples),
                  us_p90=samples[int(len(samples) * 0.9) - 1], us_max=samples[-1],
                  samples_us=samples,
                  wallclock_scope="墙钟含 eager 主机侧派发开销，不是 kernel 时间；"
                                  "逐任务 kernel 时长看 --swimlane 4 的产物")
    if args.swimlane >= 4 and not args.pmu:
        report["swimlane"] = _export_swimlane(args.output / "dfx")
        report["swimlane_windows"] = [
            _export_swimlane(args.output / "dfx" / f"window_{index}")
            for index in range(1, max(1, args.windows))
        ]
    report["final_finite_checks"] = validate_outputs(tensors, kernel.output_param_names)
    if report["final_finite_checks"]["status"] == "FAIL":
        raise ValueError(f"CSA 计时后输出检查失败：{report['final_finite_checks']['errors']}")
    out_name = "x_out"
    out = tensors[out_name].detach().float().cpu()
    report[out_name] = {"finite": bool(torch.isfinite(out).all()),
                          "absmax": float(out.abs().max()), "mean": float(out.mean())}


if __name__ == "__main__":
    main()
