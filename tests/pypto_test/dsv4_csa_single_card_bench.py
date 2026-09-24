"""单卡回放一次真实 CSA 调用：验正确性、量单次耗时、可选采泳道。

存在的理由：改一处 kernel 就起 16 卡整模型（75 个权重分片、约 4.5 分钟加载）去看
它崩不崩、快不快，代价太高。本脚本吃 `offline_pd/run.py argdump` 落盘的真实根入参，
在**一张卡**上直接调 `decode_csa_tp1_attention_test`，一轮几十秒。

口径限制：
- 只跑 CSA 这一个算子，不含 MoE 与通信，绝对耗时不能当端到端性能；
  它衡量的是"这次 kernel 改动让该算子本身快了还是慢了"。
- 输入是某一步、某一层的快照，不随步数变化；用于对照而非覆盖所有档位。
- aicore 故障、越界这类问题会当场暴露，这是它相对 in-core 模拟器的价值。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from dsv4_csa_env import activate, write_json


def _export_swimlane(directory: Path) -> dict:
    """把本次 DFX 记录转成带真实任务名的泳道，任务名取自本进程实际生成的 kernel_config。"""
    import ast
    import json as _json
    import subprocess

    records = directory / "chip_swimlane_records.json"
    deps = directory / "deps.json"
    if not records.is_file() or not deps.is_file():
        return {"exported": False, "reason": "DFX 未产出记录或依赖"}
    # 单卡进程只编一份 kernel，按 mtime 取最新的即可，不像多 rank 那样有歧义。
    builds = sorted(Path("build_output").glob("_jit__decode_csa_tp1_attention_*/kernel_config.py"),
                    key=lambda p: p.stat().st_mtime)
    if not builds:
        return {"exported": False, "reason": "未找到 kernel_config.py"}
    table = builds[-1]
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
    return {"exported": proc.returncode == 0, "kernel_config": str(table),
            "merged_swimlane": str(merged), "name_map": str(name_map)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-dir", type=Path, required=True, help="run.py argdump 落盘目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20, help="计时轮数（另有 3 轮预热）")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--swimlane", type=int, default=0, choices=range(5),
                        help="非 0 时开 DFX 芯片泳道；4 才有逐任务 kernel 时长")
    args = parser.parse_args()
    activate()

    import torch
    import torch_npu  # noqa: F401  加载 NPU 后端

    import pypto.torch
    from vllm_ascend.ops.pypto.variant import selected_variant, variant_package

    package = variant_package()
    kernel = __import__(f"{package}.decode_csa", fromlist=["decode_csa_tp1_attention_test"]).decode_csa_tp1_attention_test

    meta = json.loads((args.args_dir / "csa_args_meta.json").read_text())
    blob = torch.load(args.args_dir / "csa_args.pt", map_location="cpu")
    names = list(kernel.param_names)
    if names != meta["param_names"]:
        raise ValueError("落盘入参的参数表与当前 kernel 不一致，需重新 argdump")

    device = f"npu:{args.device}"
    torch.npu.set_device(args.device)
    tensors = {name: blob[name].to(device) for name in names}
    call_args = tuple(tensors[name] for name in names)

    args.output.mkdir(parents=True, exist_ok=True)
    # 执行目标一次性定在进程上：JIT 调用本身不接 RunConfig。
    pypto.torch.init(device=args.device, platform="a2a3",
                     enable_chip_swimlane=args.swimlane,
                     enable_dep_gen=args.swimlane >= 4,
                     output_dir=str((args.output / "dfx").resolve()) if args.swimlane else None)

    report = {"variant": selected_variant(), "package": package, "device": args.device,
              "layer_index": meta["layer_index"], "tokens": meta["tokens"],
              "iters": args.iters, "warmup": args.warmup, "swimlane_level": args.swimlane,
              "scope": "单算子单卡回放，不含 MoE 与通信；绝对耗时不代表端到端性能"}
    try:
        for _ in range(args.warmup):
            kernel(*call_args)
        torch.npu.synchronize()

        samples = []
        for _ in range(args.iters):
            torch.npu.synchronize()
            start = time.perf_counter()
            kernel(*call_args)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1e6)
        samples.sort()
        if args.swimlane:
            # 只给一次调用开窗口：不然每轮都记一遍，产物没法逐任务对照。
            # 墙钟里绝大部分是 eager 下的主机侧派发开销（PyPTO 的 _resolve_compiled
            # 按调用次数计费），要量 kernel 本身必须看泳道的 kernel-duration。
            torch.npu.synchronize()
            pypto.torch.begin_dfx()
            try:
                kernel(*call_args)
            finally:
                pypto.torch.end_dfx()
            torch.npu.synchronize()
        report.update(status="PASS",
                      us_min=samples[0], us_p50=statistics.median(samples),
                      us_p90=samples[int(len(samples) * 0.9) - 1], us_max=samples[-1],
                      samples_us=samples,
                      wallclock_scope="墙钟含 eager 主机侧派发开销，不是 kernel 时间；"
                                      "逐任务 kernel 时长看 --swimlane 4 的产物")
        if args.swimlane >= 4:
            report["swimlane"] = _export_swimlane(args.output / "dfx")
        out = tensors["attn_out"].detach().float().cpu()
        report["attn_out"] = {"finite": bool(torch.isfinite(out).all()),
                              "absmax": float(out.abs().max()), "mean": float(out.mean())}
    except BaseException as exc:  # 故障也要留证据
        report.update(status="FAIL", error=repr(exc))
        write_json(args.output / "report.json", report)
        raise
    write_json(args.output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "samples_us"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
